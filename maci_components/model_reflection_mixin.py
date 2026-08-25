"""Reflection context, consultation, and prompt synthesis logic for MACI models."""

from .agent_support import *


class ModelReflectionMixin:
    def build_reflection_context(self):
        """Builds the shared post-round context for optimization reflection."""
        return {
            "target": list(getattr(self, "target_pos", [])),
            "steps": max((getattr(agent, "turns", 0) for agent in self.agents), default=0),
            "token_usage": self.token_summary(),
            "communication_log": list(getattr(self, "communication_log", []))[-40:],
            "gates_open": bool(getattr(self, "gates_open", False)),
            "open_gates": [list(pos) for pos in sorted(getattr(self, "open_gates", set()))],
            "switches": [list(pos) for pos in getattr(self, "switches", [])],
            "gates": [list(pos) for pos in getattr(self, "gates", [])],
            "agents": [
                {
                    "label": self._agent_label(agent),
                    "model": getattr(agent, "model_name", ""),
                    "position": list(agent.pos),
                    "done": bool(getattr(agent, "is_done", False)),
                    "turns": int(getattr(agent, "turns", 0)),
                    "tokens": int(getattr(agent, "total_tokens", 0)),
                    "recent_path": list(getattr(agent, "action_history", [])[-16:]),
                    "last_decision": getattr(agent, "last_decision", {}),
                    "memory": getattr(agent, "memory", ""),
                    "landmarks": getattr(agent, "structured_memory", {}).get("landmarks", {}),
                    "blocked_pos": getattr(agent, "structured_memory", {}).get("blocked_pos", []),
                    "inventory": getattr(agent, "structured_memory", {}).get("inventory", []),
                    "communication_space": getattr(agent, "structured_memory", {}).get("communication_space", {})
                }
                for agent in self.agents
            ]
        }

    def run_reflection_consultation(self, reflection_context, max_rounds=2):
        """Lets agents discuss the finished round in natural language wrapped in JSON."""
        discussion = []
        for round_idx in range(1, max_rounds + 1):
            round_items = []
            for agent in self.agents:
                label = self._agent_label(agent)
                prompt = f"""
The optimization round has ended. You are Agent {label}.
Discuss with the other agents in natural language, but return ONLY JSON so the system can parse it.

Normal gameplay broadcasts use ONLY base codes F/S/G/K/D/H/X/N. Do not propose or use numbered code forms.

Shared round context:
{json.dumps(reflection_context, ensure_ascii=False)}

Previous reflection discussion:
{json.dumps(discussion, ensure_ascii=False)}

Return ONLY JSON:
{{
  "message": "natural-language message to the other agents",
  "lessons": ["short lesson"],
  "proposed_next_round_rule": "short rule for the next round",
  "need_more_discussion": true/false
}}

Keep message and lessons concrete: mention movement, switch/gate timing, frontier choice, and communication only when relevant.
"""
                try:
                    kwargs = {
                        "model": agent.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    if getattr(self, "json_response_format_supported", True):
                        kwargs["response_format"] = {"type": "json_object"}
                    resp = agent.llm_client.chat.completions.create(**kwargs)
                    usage_record = {}
                    if hasattr(resp, "usage") and resp.usage:
                        usage_record = agent._record_token_usage(resp.usage, source="reflection_consultation")
                    raw_text_full = resp.choices[0].message.content.strip()
                    provider_reasoning = extract_provider_reasoning(resp.choices[0].message)
                    inline_reasoning, raw_text = extract_reasoning_and_answer(raw_text_full)
                    reasoning_content = "\n\n".join(part for part in [provider_reasoning, inline_reasoning] if part)
                    if hasattr(self, "log_llm_io"):
                        self.log_llm_io({
                            "agent": label,
                            "phase": "reflection_consultation",
                            "round": round_idx,
                            "model": agent.model_name,
                            "provider": getattr(self, "provider", ""),
                            "messages_sent": [{"role": "user", "content": prompt}],
                            "raw_response": raw_text_full,
                            "reasoning_content": reasoning_content,
                            "cleaned_response": raw_text,
                            "token_usage": usage_record,
                        })
                    data = json.loads(raw_text)
                    if not isinstance(data.get("lessons", []), list):
                        data["lessons"] = [str(data.get("lessons", ""))]
                    data["raw_response"] = raw_text_full
                    data["reasoning_content"] = reasoning_content
                except Exception as e:
                    data = {
                        "message": f"Reflection unavailable for Agent {label}: {e}",
                        "lessons": [],
                        "proposed_next_round_rule": "",
                        "need_more_discussion": False,
                        "error": str(e),
                        "raw_response": ""
                    }
                data["agent"] = label
                data["round"] = round_idx
                round_items.append(data)
                discussion.append(data)
            if not any(bool(item.get("need_more_discussion", False)) for item in round_items):
                break
        return discussion

    def _fallback_reflection_synthesis(self, raw_text="", error=""):
        return {
            "global_system_patch": "Avoid repeated branches, use base-code communication only, and coordinate switch/gate support explicitly.",
            "agent_patches": {},
            "communication_guideline": "Use only base codes F/S/G/K/D/H/X/N; do not append digits. Use G when staged at a gate, S when actively providing switch support, H when help is needed, F when the exit is reachable, X for unusable routes, and N otherwise.",
            "summary": "Fallback reflection patch applied because synthesis was unavailable.",
            "raw_response": raw_text,
            "error": error
        }

    def synthesize_reflection_prompt_patch(self, reflection_context, discussion):
        """Synthesizes parsed consultation into temporary next-round prompt patches."""
        synthesis_agent = next((agent for agent in self.agents if getattr(agent, "is_done", False)), None)
        if synthesis_agent is None and self.agents:
            synthesis_agent = self.agents[0]
        if synthesis_agent is None:
            return self._fallback_reflection_synthesis(error="No agents available for synthesis")

        labels = [self._agent_label(agent) for agent in self.agents]
        prompt = f"""
Synthesize the parsed natural-language reflection discussion into temporary next-round system prompt additions.

Normal gameplay broadcasts use ONLY base codes F/S/G/K/D/H/X/N. Numeric suffixes are disabled. Do not write rules that use numbered code forms.

Shared round context:
{json.dumps(reflection_context, ensure_ascii=False)}

Parsed reflection discussion:
{json.dumps(discussion, ensure_ascii=False)}

Agent labels:
{json.dumps(labels, ensure_ascii=False)}

Return ONLY JSON:
{{
  "global_system_patch": "shared next-round temporary instruction",
  "agent_patches": {{"A": "agent-specific temporary instruction"}},
  "communication_guideline": "base-code communication rule, no numeric suffixes",
  "summary": "short human-readable summary"
}}

Rules:
- Patches must be short, tactical, and suitable to append to the agents' system prompts for only the next round.
- Do not include the original base prompt, output schema, or JSON formatting instructions.
- Preserve base-code communication only. Use no numbered code forms.
"""
        raw_text = ""
        try:
            kwargs = {
                "model": synthesis_agent.model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            if getattr(self, "json_response_format_supported", True):
                kwargs["response_format"] = {"type": "json_object"}
            resp = synthesis_agent.llm_client.chat.completions.create(**kwargs)
            usage_record = {}
            if hasattr(resp, "usage") and resp.usage:
                usage_record = synthesis_agent._record_token_usage(resp.usage, source="reflection_synthesis")
            raw_text_full = resp.choices[0].message.content.strip()
            provider_reasoning = extract_provider_reasoning(resp.choices[0].message)
            inline_reasoning, raw_text = extract_reasoning_and_answer(raw_text_full)
            reasoning_content = "\n\n".join(part for part in [provider_reasoning, inline_reasoning] if part)
            if hasattr(self, "log_llm_io"):
                self.log_llm_io({
                    "agent": self._agent_label(synthesis_agent),
                    "phase": "reflection_synthesis",
                    "model": synthesis_agent.model_name,
                    "provider": getattr(self, "provider", ""),
                    "messages_sent": [{"role": "user", "content": prompt}],
                    "raw_response": raw_text_full,
                    "reasoning_content": reasoning_content,
                    "cleaned_response": raw_text,
                    "token_usage": usage_record,
                })
            data = json.loads(raw_text)
            if not isinstance(data, dict):
                raise ValueError("Reflection synthesis JSON was not an object")
            agent_patches = data.get("agent_patches", {})
            if not isinstance(agent_patches, dict):
                agent_patches = {}
            result = {
                "global_system_patch": strip_numeric_suffix_prompt_text(str(data.get("global_system_patch", "")).strip(), include_default=False),
                "agent_patches": {
                    str(label): strip_numeric_suffix_prompt_text(str(patch).strip(), include_default=False)
                    for label, patch in agent_patches.items()
                    if str(patch).strip()
                },
                "communication_guideline": strip_numeric_suffix_prompt_text(str(data.get("communication_guideline", "")).strip(), include_default=False),
                "summary": str(data.get("summary", "")).strip(),
                "raw_response": raw_text
            }
            if not result["global_system_patch"]:
                result["global_system_patch"] = self._fallback_reflection_synthesis()["global_system_patch"]
            if not result["communication_guideline"]:
                result["communication_guideline"] = self._fallback_reflection_synthesis()["communication_guideline"]
            return result
        except Exception as e:
            return self._fallback_reflection_synthesis(raw_text=raw_text, error=str(e))

    def apply_reflection_prompt_patch(self, synthesis):
        global_patch = strip_numeric_suffix_prompt_text(synthesis.get("global_system_patch", ""), include_default=False)
        agent_patches = synthesis.get("agent_patches", {}) if isinstance(synthesis.get("agent_patches", {}), dict) else {}
        for agent in self.agents:
            label = self._agent_label(agent)
            agent_patch = strip_numeric_suffix_prompt_text(agent_patches.get(label, ""), include_default=False)
            combined = "\n".join(part for part in [global_patch, agent_patch] if str(part).strip()).strip()
            agent.prompt_addition = combined
            agent.prompt_addition_source = json.dumps({
                "global_system_patch": global_patch,
                "agent_patch": agent_patch,
                "summary": synthesis.get("summary", "")
            }, ensure_ascii=False)

        communication_guideline = strip_numeric_suffix_prompt_text(synthesis.get("communication_guideline", ""), include_default=False)
        if communication_guideline:
            self.current_communication_guideline = communication_guideline[:1200]
        self.communication_improvements.append({
            "session": len(self.communication_improvements) + 1,
            "items": [],
            "applied_guideline": self.current_communication_guideline,
            "source": "reflection_synthesis",
            "token_usage": self.token_summary()
        })
