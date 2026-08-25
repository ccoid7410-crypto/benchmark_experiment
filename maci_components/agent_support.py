import os
import mesa
from openai import OpenAI
import json
import random
import re
import datetime

DEFAULT_BROADCAST_CODES = ["F", "S", "G", "K", "D", "H", "X", "N"]

# Default OpenAI-compatible endpoints per provider. An explicitly supplied
# base_url always overrides these, so "openai" or "custom" can also be
# pointed at a local llama.cpp server (llama-server exposes /v1 by default).
PROVIDER_DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "llamacpp": "http://localhost:8080/v1",
}

# Local/self-hosted OpenAI-compatible servers (llama.cpp, Ollama, arbitrary
# custom endpoints) don't reliably support strict json_object response
# formatting or the reasoning_effort parameter, so those are only sent for
# providers known to accept them.
RELAXED_API_PROVIDERS = {"ollama", "llamacpp", "custom"}


def build_llm_client(provider, api_key, base_url):
    """Builds an OpenAI-compatible client for the selected provider.

    Never falls back to a bundled/hardcoded key - only the explicitly
    passed api_key or a <PROVIDER>_API_KEY / OPENAI_API_KEY environment
    variable are used.
    """
    provider = str(provider or "openai").lower()
    resolved_base_url = base_url or PROVIDER_DEFAULT_BASE_URLS.get(provider)
    resolved_api_key = (
        api_key
        or os.environ.get(f"{provider.upper()}_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    if provider in ("ollama", "llamacpp") and not resolved_api_key:
        resolved_api_key = "not-needed"
    if resolved_base_url:
        client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
    else:
        client = OpenAI(api_key=resolved_api_key)
    return client, resolved_base_url, resolved_api_key


def extract_reasoning_and_answer(raw_text):
    """Splits <think>...</think>-style inline reasoning out of a raw response.

    Local reasoning models served through llama.cpp/Ollama often emit their
    chain-of-thought inline in <think> tags. This preserves that content
    (for logging) instead of discarding it, while returning a cleaned
    answer string for JSON parsing.
    """
    text = str(raw_text or "")
    reasoning_parts = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    reasoning = "\n\n".join(part.strip() for part in reasoning_parts if part.strip())
    return reasoning, cleaned
NO_NUMERIC_SYMBOL_SPACE_PROMPT = """Communication mode:
- Coded communication is enabled, but numeric suffixes are disabled.
- Use only base codes: F, S, G, K, D, H, X, N.
- Do not append digits to broadcast codes.
- If no cooperative event is active, use N.

Experimenter notes:
- Add temporary meanings or hypotheses here when needed."""

def strip_numeric_suffix_prompt_text(value, include_default=True):
    text = str(value or "")
    if not text.strip():
        return ""
    cleaned_lines = []
    for line in text.splitlines():
        lowered = line.lower()
        mentions_numeric = any(token in lowered for token in [
            "numeric suffix space",
            "integer suffix",
            "optional integer suffix",
            "optionally followed",
            "code+number",
            "0-10",
            "n-slot",
            "s3",
            "s7",
            "h10",
            "g0",
            "s0",
            "n0",
        ])
        is_no_numeric_rule = (
            "disabled" in lowered
            or "do not use" in lowered
            or "invalid examples" in lowered
            or "no numeric suffix" in lowered
            or "with no numeric suffix" in lowered
        )
        if mentions_numeric and not is_no_numeric_rule:
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if include_default and "Communication mode:" not in text:
        text = f"{NO_NUMERIC_SYMBOL_SPACE_PROMPT}\n\n{text}".strip()
    return text

def parse_broadcast_codes(value):
    if isinstance(value, list):
        raw = "".join(str(item) for item in value)
    else:
        raw = str(value or "")
    codes = []
    for token in re.findall(r"[A-Za-z]", raw.upper()):
        if token not in codes:
            codes.append(token)
    return codes or DEFAULT_BROADCAST_CODES[:]

def usage_value(obj, key, default=0):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default) or default
    return getattr(obj, key, default) or default

def extract_token_usage(usage):
    completion_details = usage_value(usage, "completion_tokens_details", {}) or {}
    prompt_details = usage_value(usage, "prompt_tokens_details", {}) or {}
    return {
        "prompt_tokens": int(usage_value(usage, "prompt_tokens", 0)),
        "completion_tokens": int(usage_value(usage, "completion_tokens", 0)),
        "reasoning_tokens": int(usage_value(completion_details, "reasoning_tokens", 0)),
        "cached_tokens": int(usage_value(prompt_details, "cached_tokens", 0)),
        "total_tokens": int(usage_value(usage, "total_tokens", 0))
    }

def merge_token_usage(target, addition):
    for key in ["prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens", "total_tokens"]:
        target[key] = int(target.get(key, 0)) + int(addition.get(key, 0))
    return target

DEFAULT_PROMPT_TEMPLATES = {
    "gpt": """
<role>
You are Agent {agent_id}, a cooperative maze-running agent in Project MACI.
Your goal is to reach the exit symbol 'F' in as few turns and tokens as possible while coordinating through compact symbolic broadcasts.
</role>

<output_contract>
Reply ONLY with one valid JSON object. No markdown, no prose outside JSON.
Required keys:
- "reason": concise tactical rationale, 1-3 sentences. Do not expose hidden chain-of-thought.
- "action": one of "UP", "DOWN", "LEFT", "RIGHT".
- "blocks": integer from 1 to {speed_limit}.
- "broadcast_message": one base code only, no numeric suffix. Valid: F/S/G/K/D/H/X/N.
- "notes": short memory update for the next turn.
- "structured_memory": compact memory update; communication_space may define base-code meanings.
</output_contract>
""",
    "gemini": """
<role>
You are a cooperative maze-running agent in Project MACI.
You control exactly one agent: Agent {agent_id}.
Your goal is to reach the exit F efficiently while coordinating with partner agents under limited communication.
</role>

<task>
At each turn, inspect the provided map, memory, interaction rules, inbox messages, and recent path.
Choose exactly one movement action and broadcast a compact base code.
</task>

<output_contract>
Return ONLY one valid JSON object. No markdown. No extra prose.
Required keys:
{{
  "reason": "1-3 concise tactical sentences. Do not reveal hidden chain-of-thought.",
  "action": "UP | DOWN | LEFT | RIGHT",
  "blocks": integer from 1 to {speed_limit},
  "broadcast_message": "one of F/S/G/K/D/H/X/N, with no numeric suffix",
  "notes": "short memory update for next turn",
  "structured_memory": {{"communication_space": {{}}}}
}}
</output_contract>
""",
    "kimi": """
You are Agent {agent_id}, a cooperative maze-running agent in Project MACI.

You must act as one grid-world agent. This is tactical navigation with compact communication.
Be decisive. Keep reasoning short. Preserve and reuse compact communication conventions.
Do not output anything except the requested JSON object.

Output exactly one JSON object:
{{
  "reason": "concise tactical rationale, 1-3 sentences",
  "action": "UP | DOWN | LEFT | RIGHT",
  "blocks": 1,
  "broadcast_message": "N",
  "notes": "short next-turn memory",
  "structured_memory": {{"communication_space": {{}}}}
}}

broadcast_message must be one of F/S/G/K/D/H/X/N with no numeric suffix.
Valid examples: "N", "S", "H", "G".
Invalid examples: "", any code with digits, "Switch here".
"""
}

COMMON_PROMPT_APPENDIX = """
=== DECISION PRIORITIES ===
1. If 'F' is visible or known and reachable, move toward it immediately.
2. If a useful interactive tile is visible, handle it: stand on 'S' to open 'G', collect 'K' before 'D'.
3. Otherwise move toward a frontier: a known open tile next to unknown space.
4. Avoid recent loops, confirmed walls, and blocked positions.
5. If a shortest known path to a landmark is needed, you may request:
   {{"tool_call": {{"name": "dijkstra", "target_x": X, "target_y": Y}}}}

=== FRONTIER MEMORY ===
- Use frontier_memory as your exploration queue.
- Prefer the nearest frontier that does not repeat the recent path.
- When blocked, remember that coordinate as blocked and pick another frontier.
- If no frontier is listed, pick a safe unexplored-looking direction from immediate neighbors.

=== COMMUNICATION CODEBOOK ===
S = switch-hold. You are on, moving to, or requesting continued control of an active switch.
G = gate-staging. You are waiting at or blocked by a relevant gate and ready to pass when it opens.
H = help-signal. You need partner support now, usually switch-hold or continued switch cycling.
F = finish-confirm. Exit is known/reachable; keep cooperation active until the finisher is through or done.
K = key-control. Key is visible, collected, or needed for a locked door.
D = door-block. Locked door is visible or currently blocking progress.
X = bad-route. Confirmed blocked/dead-end/hazard route.
N = navigation. No urgent cooperative event; broadcasting exploration/frontier movement status.

=== COMMUNICATION RULES ===
- broadcast_message must be one standard base code on every turn.
- Empty broadcast_message is invalid. If no F/S/G/K/D/H/X event is useful, send N.
- Numeric suffixes are disabled. Bare codes like S, G, H, F, and N are valid.
- Remember useful base-code meanings in structured_memory.communication_space and reuse them consistently.
- Broadcast a standard base code every turn. Use F/S/G/K/D/H/X for confirmed related events; otherwise use N for current navigation/frontier status.
- Interpret inbox codes using the same codebook.
- If an inbox message includes an old number form, ignore the number and interpret the base code.
- If you receive H/G/F from a partner near a gate or exit, keep holding/cycling useful S tiles until the partner passes or finishes.
- If you are blocked by a closed gate near F, broadcast H or G, stage adjacent to that gate, and retry when partner switch support is active.
- Secret symbols must not conflict with F, S, G, K, D, H, or X.

=== INTERACTIVE TILES ===
- 'S' & 'G': Stand on Switch 'S' to open linked Gates 'G'. Gates stay open only while an allowed agent is on a switch.
- 'K' & 'D': Step on Key 'K' to pick it up. The key holder can pass through Door 'D', consuming one key.
- Obey Interaction Rules from the current status. Some switches/gates may only work for specific agents.

=== COOPERATION KEYWORDS ===
- switch-hold: The switch agent stays on or cycles within active S tiles to keep linked gates open until the partner passes or finishes.
- gate-staging: The blocked agent waits adjacent to the target gate so they can pass immediately when the switch opens it.
- help-signal: H means the agent is blocked by a cooperative gate/obstacle and needs the partner to act or keep acting.
- finish-confirm: F near a gate means the exit is reachable, but switch support must continue until the finisher is through or done.

Legend: ' ': unknown, '.': path, '#': wall, '@': YOU, numbers: other agents, letters: symbols.
"""
