"""Success debrief conversation logic for MACI models."""

from .agent_support import *


class ModelSuccessDebriefMixin:
    def run_success_debrief(self, max_rounds=3):
        """Lets agents briefly consult after success, then produces up to 8 keywords with definitions."""
        if self.success_debrief_done:
            return self.success_debrief

        self.success_debrief_done = True
        discussion = []
        shared_context = {
            "target": list(getattr(self, "target_pos", [])),
            "steps": max((getattr(agent, "turns", 0) for agent in self.agents), default=0),
            "token_usage": self.token_summary(),
            "communication_log": self.communication_log[-20:],
            "agents": [
                {
                    "label": self._agent_label(agent),
                    "model": getattr(agent, "model_name", ""),
                    "position": list(agent.pos),
                    "done": bool(getattr(agent, "is_done", False)),
                    "inventory": list(getattr(agent, "structured_memory", {}).get("inventory", [])),
                    "memory": getattr(agent, "memory", ""),
                    "recent_path": getattr(agent, "action_history", [])[-10:],
                    "landmarks": getattr(agent, "structured_memory", {}).get("landmarks", {})
                }
                for agent in self.agents
            ]
        }

        for round_idx in range(1, max_rounds + 1):
            round_notes = []
            for agent in self.agents:
                prompt = f"""
The maze exploration succeeded. You are Agent {self._agent_label(agent)}.
Consult with the other agents only as much as needed to agree on the important lessons and vocabulary from this run.

Shared run context:
{json.dumps(shared_context, ensure_ascii=False)}

Previous debrief discussion:
{json.dumps(discussion, ensure_ascii=False)}

Return ONLY JSON:
{{
  "need_more_discussion": true/false,
  "message": "one concise contribution for the other agents",
  "candidate_keywords": [
    {{"keyword": "short term", "definition": "one sentence"}}
  ]
}}
Keep candidate_keywords to the terms that are actually useful for explaining this run.
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
                        usage_record = agent._record_token_usage(resp.usage, source="success_debrief_discussion")
                    raw_text_full = resp.choices[0].message.content.strip()
                    provider_reasoning = str(getattr(resp.choices[0].message, "reasoning_content", "") or "").strip()
                    inline_reasoning, text = extract_reasoning_and_answer(raw_text_full)
                    if hasattr(self, "log_llm_io"):
                        self.log_llm_io({
                            "agent": self._agent_label(agent),
                            "phase": "success_debrief_discussion",
                            "round": round_idx,
                            "model": agent.model_name,
                            "provider": getattr(self, "provider", ""),
                            "messages_sent": [{"role": "user", "content": prompt}],
                            "raw_response": raw_text_full,
                            "reasoning_content": "\n\n".join(part for part in [provider_reasoning, inline_reasoning] if part),
                            "cleaned_response": text,
                            "token_usage": usage_record,
                        })
                    data = json.loads(text)
                    data["raw_response"] = raw_text_full
                except Exception as e:
                    data = {
                        "need_more_discussion": False,
                        "message": f"Debrief unavailable for Agent {self._agent_label(agent)}: {e}",
                        "candidate_keywords": []
                    }
                data["agent"] = self._agent_label(agent)
                data["round"] = round_idx
                round_notes.append(data)
                discussion.append(data)

            if not any(bool(item.get("need_more_discussion", False)) for item in round_notes):
                break

        synthesis_agent = next((agent for agent in self.agents if getattr(agent, "is_done", False)), None)
        if synthesis_agent is None and self.agents:
            synthesis_agent = self.agents[0]

        keywords = []
        if synthesis_agent is not None:
            prompt = f"""
The agents have finished consulting after a successful maze exploration.

Discussion:
{json.dumps(discussion, ensure_ascii=False)}

Produce the final glossary of necessary keywords for this successful run.
Rules:
- Return ONLY JSON.
- Include at most 8 items.
- Each item must have "keyword" and "definition".
- Definitions must be concise and specific to this maze run.
- Do not include duplicate or decorative terms.

JSON shape:
{{"keywords":[{{"keyword":"...", "definition":"..."}}]}}
"""
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
                    usage_record = synthesis_agent._record_token_usage(resp.usage, source="success_debrief_keywords")
                raw_text_full = resp.choices[0].message.content.strip()
                provider_reasoning = str(getattr(resp.choices[0].message, "reasoning_content", "") or "").strip()
                inline_reasoning, cleaned_text = extract_reasoning_and_answer(raw_text_full)
                if hasattr(self, "log_llm_io"):
                    self.log_llm_io({
                        "agent": self._agent_label(synthesis_agent),
                        "phase": "success_debrief_keywords",
                        "model": synthesis_agent.model_name,
                        "provider": getattr(self, "provider", ""),
                        "messages_sent": [{"role": "user", "content": prompt}],
                        "raw_response": raw_text_full,
                        "reasoning_content": "\n\n".join(part for part in [provider_reasoning, inline_reasoning] if part),
                        "cleaned_response": cleaned_text,
                        "token_usage": usage_record,
                    })
                data = json.loads(cleaned_text)
                raw_keywords = data.get("keywords", [])
                for item in raw_keywords[:8]:
                    if isinstance(item, dict) and item.get("keyword") and item.get("definition"):
                        keywords.append({
                            "keyword": str(item["keyword"])[:60],
                            "definition": str(item["definition"])[:240]
                        })
            except Exception as e:
                keywords = [{"keyword": "debrief_error", "definition": f"Final keyword synthesis failed: {e}"}]

        self.success_debrief = {
            "discussion": discussion,
            "keywords": keywords[:8],
            "communication_improvements": list(getattr(self, "communication_improvements", [])),
            "current_communication_guideline": getattr(self, "current_communication_guideline", ""),
            "token_usage": self.token_summary()
        }
        self.log("\n--- SUCCESS DEBRIEF ---\n" + json.dumps(self.success_debrief, ensure_ascii=False, indent=2))
        return self.success_debrief
