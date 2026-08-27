import os
import sys
import mesa
from openai import OpenAI, APIStatusError, APIConnectionError
import json
import random
import re
import time
import datetime

# Model output can contain arbitrary Unicode (math symbols, emoji, etc.), and
# on Windows the console's default codepage (e.g. cp949) often can't encode
# it - printing such text would otherwise crash mid-turn with a
# UnicodeEncodeError. Reconfigure stdout/stderr to UTF-8 once, defensively,
# since every entry point (gui_app.py, maci.py, maci_judge.py) imports this
# module early.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DEFAULT_BROADCAST_CODES = ["F", "S", "G", "K", "D", "H", "X", "N"]

# Default OpenAI-compatible endpoints per provider. An explicitly supplied
# base_url always overrides these, so "openai" or "custom" can also be
# pointed at a local llama.cpp server (llama-server exposes /v1 by default).
PROVIDER_DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "llamacpp": "http://localhost:8080/v1",
    "openrouter": "https://openrouter.ai/api/v1",
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


CONTEXT_OVERFLOW_MARKERS = (
    "exceed_context_size_error",
    "context size",
    "exceeds the available context",
    "context_length_exceeded",
    "maximum context length",
)


def is_context_overflow_error(exc) -> bool:
    """True if `exc` looks like a 'prompt too long for this model's context
    window' error, across the different wordings local servers (llama.cpp)
    and hosted routers (OpenRouter/OpenAI) use for it. Distinguishing this
    from other errors matters because retrying the identical oversized
    request just fails the same way again - the caller needs to shrink the
    prompt first."""
    if not isinstance(exc, APIStatusError):
        return False
    if getattr(exc, "status_code", None) not in (400, 413):
        return False
    return any(marker in str(exc).lower() for marker in CONTEXT_OVERFLOW_MARKERS)


def extract_provider_reasoning(message) -> str:
    """Pulls whatever reasoning text a non-streaming response message carries,
    in whichever shape the provider uses: a plain `reasoning_content` or
    `reasoning` string, or `reasoning_details` - a list of blocks each
    carrying a "text" field (OpenRouter's shape for some models).
    """
    piece = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    if piece:
        return str(piece).strip()
    details = getattr(message, "reasoning_details", None) or []
    parts = []
    for item in details:
        text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
        if text:
            parts.append(text)
    return "".join(parts).strip()


def with_retry(fn, *args, retries=3, delay=2.0, **kwargs):
    """Retry on transient network/5xx/429 errors. 4xx errors other than 429
    (e.g. a too-long prompt) are deterministic - retrying the identical
    request just fails the same way again, so those are raised immediately
    instead of wasting the retry budget."""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except APIStatusError as e:
            status = getattr(e, "status_code", None)
            if status is not None and status < 500 and status != 429:
                raise
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))
        except APIConnectionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))


def extract_reasoning_delta(delta) -> str:
    """Streaming counterpart to extract_provider_reasoning(): pulls whatever
    reasoning text a single streamed delta carries, in whichever shape the
    provider uses (plain `reasoning` string, or `reasoning_details` - a list
    of blocks each carrying a "text" field)."""
    piece = getattr(delta, "reasoning", None)
    if piece:
        return piece
    details = getattr(delta, "reasoning_details", None) or []
    parts = []
    for item in details:
        text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


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
You are Agent {agent_id}, a cooperative maze-running agent in Project MACI. Speak and act decisively.
</role>
""",
    "gemini": """
<role>
You are a cooperative maze-running agent in Project MACI.
You control exactly one agent: Agent {agent_id}. Coordinate with your partner agent(s) when it actually helps.
</role>
""",
    "kimi": """
You are Agent {agent_id}, a cooperative maze-running agent in Project MACI.
You must act as one grid-world agent. Be decisive and concise.
"""
}

COMMON_PROMPT_APPENDIX = """
=== GOAL ===
Reach the exit symbol 'F' in as few turns as possible, cooperating with your
partner agent(s) only when it actually helps.

=== OUTPUT CONTRACT: SUBMIT CODE VIA TOOL CALL, NOT PLAIN TEXT ===
You do not act by writing text. The ONLY way to act this turn is to call
the submit_policy_code tool with a `code` argument that is valid, directly-
executable Python - plain-text replies are ignored outright, not treated as
your answer. Your code is checked for valid syntax before it's accepted: if
it fails to compile, you get the exact error back and must call
submit_policy_code again with a fix - it costs you a round, not the turn.
Optionally start the code with a single "# " comment line explaining your
choice in a few words - that comment is the only place your reasoning is
shown to the experimenter, so make it count.

This is real Python, not a restricted mini-language - use whatever the
language gives you to make your decision: if/elif/else, loops, list/dict
comprehensions, helper variables, sorting with a custom key, string
formatting, arithmetic, your own local helper function (define it and call
it - or write a top-level `def decide(state): ...` and it will be called
automatically), etc. The only things unavailable are imports and file/
network access (no stdlib modules) - everything else in the language is
fair game. Whichever style you write, `auto_move()` must actually get
called by the time your code finishes running.

Before submitting, you may call the test_policy_code tool (up to a few
times) to dry-run a draft and see what it would actually do - including
Python errors - without committing to it. Use it to debug, not to explore
every option; once you're confident, call submit_policy_code with your
final code.

Your code runs against a `state` object and these functions - nothing else
is available (no imports, no file/network access):
  auto_move(direction, blocks=1, until=None)
                               # REQUIRED, call exactly once by the time your
                               # code finishes running.
                               # direction: "UP" | "DOWN" | "LEFT" | "RIGHT"
                               # blocks: integer 1..state.speed_limit
                               # until: OPTIONAL - a Python boolean expression
                               #   as a string, e.g. "state.standing_on == 'S'"
                               #   or "'F' in state.landmarks". Leave it out
                               #   for a normal single-turn move (you'll be
                               #   asked again next turn either way). Give it
                               #   a condition to keep moving the same
                               #   direction automatically on later turns,
                               #   with NO further LLM calls, until either
                               #   that condition becomes true or you can no
                               #   longer move that direction - whichever
                               #   happens first triggers a fresh call back
                               #   to you with the current state. Use this
                               #   for "keep going until something relevant
                               #   happens" instead of re-deciding every
                               #   single tile - it costs no extra tokens.
                               #   The expression is re-evaluated fresh each
                               #   turn against the state at that turn, so
                               #   only reference state fields, not local
                               #   variables from this turn's code.
  broadcast(text)              # OPTIONAL. Call at most once, only when you
                               # have something worth telling your partner.
                               # Do not call it just to say nothing useful -
                               # silence is fine when there is nothing to add.
                               # Only fires on turns the LLM is actually
                               # called - not during an auto-move run.

There is no memory API and no persistent notes - your only memory is the
conversation itself. Each turn you can see your own past turns (the code you
wrote and what happened) accumulated in context, so refer back to that
instead of re-storing facts anywhere.

state fields available to your code:
  state.pos, state.standing_on, state.speed_limit
  state.neighbors            # {{"UP": "OPEN" | "BLOCKED: ...", ...}}
  state.frontier_memory      # list of [x, y] known-open tiles worth exploring
  state.landmarks            # {{"symbol": [x, y]}}
  state.blocked_pos          # list of [x, y] previously confirmed blocked
  state.visible_agents       # ["Agent <id> at (x, y)", ...]
  state.inbox                # list of messages received since your last turn
  state.inventory            # list of held items (e.g. "Key")
  state.interaction_rules    # which agents may use which switches/gates
  state.recent_path          # your last few actions
  state.coordination_hint    # usually "" - see COORDINATION below

=== DECISION PRIORITIES ===
1. If 'F' is visible or known and reachable, move toward it immediately.
2. If a useful interactive tile is visible, handle it: stand on 'S' to open
   linked 'G', collect 'K' before 'D'.
3. Otherwise move toward a frontier from state.frontier_memory.
4. Avoid recent loops, confirmed walls, and state.blocked_pos.

=== INTERACTIVE TILES ===
- 'S' & 'G': Stand on Switch 'S' to open linked Gates 'G'. Gates stay open
  only while an allowed agent is on a switch.
- 'K' & 'D': Step on Key 'K' to pick it up. The key holder can pass through
  Door 'D', consuming one key.
- Obey state.interaction_rules - some switches/gates only work for specific agents.

=== COMMUNICATION ===
Communication is free-form text and optional every turn - there is no fixed
code list to pick from. Call broadcast(text) when telling your partner
something would actually change what they do (e.g. "holding the switch,
go through the gate now"); don't call it otherwise, and don't pad it with
filler just to say something.

=== COORDINATION ===
state.coordination_hint is normally empty. If you've been blocked and silent
for a while, it will contain a short nudge like "you have been blocked for
N turns without saying anything to your partner - consider whether telling
them would help." Treat it as a hint, not a command: broadcast only if it
actually applies, and it is fine to conclude it doesn't and stay silent again.

Legend: ' ': unknown, '.': path, '#': wall, '@': YOU, numbers: other agents, letters: symbols.
"""
