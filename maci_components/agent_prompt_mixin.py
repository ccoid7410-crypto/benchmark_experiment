"""Prompt rendering and optimization helpers for MACI agents."""

from .agent_support import *


class AgentPromptMixin:
    def _render_prompt_template(self):
        template = self.custom_system_prompt.strip()
        if not template:
            template = DEFAULT_PROMPT_TEMPLATES.get(self.prompt_profile, DEFAULT_PROMPT_TEMPLATES["gpt"])
        try:
            return template.format(
                agent_id=self.unique_id,
                model_name=self.model_name,
                speed_limit=self.speed_limit,
                vision_range=self.vision_range,
                prompt_profile=self.prompt_profile,
            )
        except Exception:
            return template

    def _build_system_prompt(self, include_self_improvement=True):
        base_prompt = self._render_prompt_template()
        communication_guideline = getattr(self.model, "current_communication_guideline", "").strip()
        improvement_section = f"""
=== SELF-IMPROVED PROTOCOL ===
{communication_guideline if communication_guideline else "- No self-improved guideline has been learned yet."}
"""
        parts = [base_prompt.strip(), COMMON_PROMPT_APPENDIX.strip()]
        if include_self_improvement:
            parts.append(improvement_section.strip())
        return "\n\n".join(part for part in parts if part)

    def _remember_optimization_base_prompt(self):
        """Keep the non-optimization system prompt so later reflection output can be reduced to a delta."""
        if not self.optimization_base_prompt:
            self.optimization_base_prompt = self._build_system_prompt(include_self_improvement=False)
        return self.optimization_base_prompt

    def _extract_prompt_addition(self, raw_text, base_prompt=None):
        """Accept a guideline, JSON wrapper, or full prompt and return only the optimization delta."""
        original = str(raw_text or "").strip()
        if not original:
            return ""

        text = original.replace("```json", "").replace("```", "").strip()

        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            try:
                parsed = json.loads(text[first_brace:last_brace + 1])
                for key in [
                    "guideline",
                    "prompt_addition",
                    "optimization_guideline",
                    "optimized_guideline",
                    "improvement",
                    "strategy",
                    "optimized_prompt",
                    "system_prompt"
                ]:
                    value = parsed.get(key)
                    if isinstance(value, str) and value.strip():
                        text = value.strip()
                        break
            except Exception:
                pass

        opt_match = re.search(
            r"\[Optimization Mode Active\]\s*(?:Guidelines?:)?\s*(.*)",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
        if opt_match:
            text = opt_match.group(1).strip()

        removable_chunks = [
            base_prompt,
            self.optimization_base_prompt,
            self._build_system_prompt(include_self_improvement=False),
            self._build_system_prompt(include_self_improvement=True),
            self._render_prompt_template(),
            COMMON_PROMPT_APPENDIX.strip()
        ]
        for chunk in removable_chunks:
            chunk = str(chunk or "").strip()
            if chunk and chunk in text:
                text = text.replace(chunk, " ")

        text = re.sub(r"(?is)<role>.*?</role>", " ", text)
        text = re.sub(r"(?is)<output_contract>.*?</output_contract>", " ", text)
        text = re.sub(r"(?is)<task>.*?</task>", " ", text)
        text = re.sub(r"(?is)===\s*SELF-IMPROVED PROTOCOL\s*===.*?(?====|\Z)", " ", text)
        text = re.sub(r"(?im)^\s*(Guidelines?|Prompt Additions?|Optimization Mode Active)\s*:\s*", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if not text:
            text = original
        return text[:4000]
