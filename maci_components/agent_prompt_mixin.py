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
                allowed_broadcast_codes="/".join(getattr(self.model, "allowed_broadcast_codes", DEFAULT_BROADCAST_CODES))
            )
        except Exception:
            return template

    def _build_system_prompt(self, include_self_improvement=True):
        base_prompt = self._render_prompt_template()
        allowed_codes = "/".join(getattr(self.model, "allowed_broadcast_codes", DEFAULT_BROADCAST_CODES))
        if getattr(self.model, "coded_communication", False):
            base_prompt = re.sub(
                r'(?im)^\s*-?\s*"broadcast_message".*$',
                f'- "broadcast_message": one base code only, no numeric suffix. Valid: {allowed_codes}.',
                base_prompt
            )
            base_prompt = re.sub(
                r'(?im)^.*broadcast_message must be one of .*$',
                f'broadcast_message must be one of {allowed_codes}; do not append numbers.',
                base_prompt
            )
            base_prompt = re.sub(r'(?im)^.*Valid examples:.*N0.*$', 'Valid examples: "N", "S", "H", "G".', base_prompt)
            base_prompt = re.sub(r'(?im)^.*Invalid examples:.*$', 'Invalid examples: "", any code with digits, natural-language sentences.', base_prompt)
            base_prompt = strip_numeric_suffix_prompt_text(base_prompt, include_default=False)
        else:
            base_prompt = re.sub(
                r'(?im)^\s*-?\s*"broadcast_message".*$',
                '- "broadcast_message": optional short natural-language status for partners; use "" if no useful update.',
                base_prompt
            )
            base_prompt = re.sub(
                r'(?im)^.*broadcast_message must be one of .*$', 
                'broadcast_message is optional plain text. Do not use numbered compact codes for now.',
                base_prompt
            )
            base_prompt = re.sub(r'(?im)^.*Valid examples:.*N0.*$', 'Valid examples: "", "holding switch", "at gate, need switch", "exit reachable".', base_prompt)
            base_prompt = re.sub(r'(?im)^.*Invalid examples:.*$', '', base_prompt)
            base_prompt = base_prompt.replace("compact symbolic broadcasts", "short natural-language updates")
            base_prompt = base_prompt.replace("compact symbolic message", "short natural-language status")
        symbol_space = self.symbol_space_prompt.strip()
        communication_guideline = getattr(self.model, "current_communication_guideline", "").strip()
        if getattr(self.model, "coded_communication", False):
            symbol_section = f"""
=== EXPERIMENTER SYMBOL SPACE ===
- Allowed broadcast base codes for this run: {allowed_codes}.
- Every turn MUST broadcast one allowed base code only.
- Numeric suffixes are disabled. Use S, G, H, F, K, D, X, or N; do not append digits.
- If no urgent cooperative event exists, use N for navigation/frontier status when N is allowed.
- Use the experimenter-defined symbol notes below only when they do not conflict with the no-numeric-suffix rule.
{symbol_space if symbol_space else "- No extra experimenter symbol notes were provided."}
"""
        else:
            symbol_section = f"""
=== COMMUNICATION MODE ===
- Coded communication is currently DISABLED.
- Do not use numbered compact codes.
- broadcast_message may be empty or a short natural-language status for partners.
- Use plain terms like "holding switch", "at gate", "exit reachable", or "no help needed".
- structured_memory.communication_space is not required while coded communication is disabled.
"""
        improvement_section = f"""
=== SELF-IMPROVED COMMUNICATION PROTOCOL ===
{communication_guideline if communication_guideline else "- No self-improved communication guideline has been learned yet."}
"""
        appendix = COMMON_PROMPT_APPENDIX.strip()
        if not getattr(self.model, "coded_communication", False):
            appendix = re.sub(r"(?is)=== COMMUNICATION CODEBOOK ===.*?(?=\n=== |\Z)", "", appendix).strip()
            appendix = re.sub(r"(?is)=== COMMUNICATION RULES ===.*?(?=\n=== |\Z)", "", appendix).strip()
        else:
            appendix = strip_numeric_suffix_prompt_text(appendix, include_default=False)
        parts = [base_prompt.strip(), appendix, symbol_section.strip()]
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
        text = re.sub(r"(?is)===\s*COMMON RULES\s*===.*?(?====|\Z)", " ", text)
        text = re.sub(r"(?is)===\s*EXPERIMENTER SYMBOL SPACE\s*===.*?(?====|\Z)", " ", text)
        text = re.sub(r"(?is)===\s*SELF-IMPROVED COMMUNICATION PROTOCOL\s*===.*?(?====|\Z)", " ", text)
        text = re.sub(r"(?im)^\s*(Guidelines?|Prompt Additions?|Optimization Mode Active)\s*:\s*", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if not text:
            text = original
        return text[:4000]
