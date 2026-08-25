import math
import os
import re
import time

import pandas as pd
from openai import OpenAI, APIConnectionError, APIStatusError

HERE = os.path.dirname(os.path.abspath(__file__))


def read_secret(env_var: str, filename: str) -> str:
    """Reads a secret from an env var first, falling back to a local file
    (e.g. key.txt) resolved relative to this script, not the cwd."""
    value = os.environ.get(env_var, "").strip()
    if value:
        return value
    path = os.path.join(filename)
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise SystemExit(
            f"Missing credential: set the {env_var} environment variable, "
            f"or create {path} containing just the key."
        )


OPENROUTER_API_KEY = read_secret("OPENROUTER_API_KEY", "key.txt")
HF_TOKEN = read_secret("HF_TOKEN", "hf_key.txt")
os.environ.setdefault("HF_TOKEN", HF_TOKEN)

client_openrouter = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
client_llama_cpp = client_openrouter
df = pd.read_parquet("hf://datasets/cais/hle/data/test-00000-of-00001.parquet")

default_generation_model = 'upstage/solar-pro4'
default_evaluation_model = 'nvidia/nemotron-3.5-lightning'

# same sampling params on every call (generation and evaluation) so runs are comparable
GEN_PARAMS = dict(temperature=0.0, top_p=1.0, max_tokens=4000, seed=0)
# The local llama.cpp server (ling-3.0-tiny) has no working reasoning-effort/
# reasoning-budget control - reasoning_effort, extra_body reasoning.effort,
# reasoning.max_tokens, and chat_template_kwargs.thinking_budget were all
# tested directly and silently ignored (byte-identical output every time).
# The only lever that actually works is this hard max_tokens cap, which caps
# the evaluator's <think> budget at 2000 tokens. score_prefix() grows this
# (up to 4x) and retries if the model gets cut off before </think> closes,
# and returns NaN instead of guessing if it's still truncated after that.
EVAL_PARAMS = dict(temperature=0.0, top_p=1.0, max_tokens=2000, seed=0)

N_SAMPLES = 3  # how many dataset rows to run; raise once the pipeline looks right

def with_retry(fn, *args, retries=3, delay=2.0, **kwargs):
    """Retry on transient network/5xx/429 errors, which OpenRouter hits often on
    long batch runs. 4xx errors other than 429 (e.g. a too-long prompt) are
    deterministic - retrying the identical request just fails the same way
    again, so those are raised immediately instead of wasting the retry budget."""
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

def generate_cot(question: str) -> str:
    """Generation model solves the task, numbering each reasoning step so it can be split later."""
    resp = with_retry(
        client_openrouter.chat.completions.create,
        model=default_generation_model,
        messages=[
            {"role": "system", "content": (
                "Solve the task. Think step by step and prefix every reasoning step with "
                "'Step N:' on its own line before giving a final answer."
            )},
            {"role": "user", "content": question},
        ],
        extra_body={"reasoning": {"enabled": False}},
        **GEN_PARAMS,
    )
    choice = resp.choices[0]
    content = choice.message.content or getattr(choice.message, "reasoning", None)
    if not content:
        raise RuntimeError(f"empty generation response (finish_reason={choice.finish_reason!r})")
    return content

def split_cot_by_meaning(cot: str) -> list[str]:
    """Divide the CoT into semantically distinct parts, one per numbered reasoning step."""
    cot = cot.strip()
    parts = re.split(r"\n(?=#{0,3}\s*Step\s*\d+\s*:)", cot)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        return parts
    parts = re.split(r"\n\s*\n", cot)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else [cot]

CONTEXT_OVERFLOW_MARKERS = ("exceed_context_size_error", "context size", "exceeds the available context")


def _is_context_overflow(exc) -> bool:
    return isinstance(exc, APIStatusError) and getattr(exc, "status_code", None) == 400 and any(
        marker in str(exc).lower() for marker in CONTEXT_OVERFLOW_MARKERS
    )


def _extract_reasoning_delta(delta) -> str:
    """Pulls whatever reasoning text a streamed delta carries, in whichever
    shape this provider/model uses: a plain `reasoning` string, or
    `reasoning_details` - a list of blocks each carrying a "text" field."""
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


def score_prefix(row: pd.Series, prefix: str) -> float:
    """Evaluator model rates (0-10) how close the reasoning-so-far is to a correct
    solution. Streamed so the grader's output is visible as it is generated instead
    of only after the full response lands. Thinking is capped at a 2000-token
    budget (EVAL_PARAMS['max_tokens']) - see the comment above EVAL_PARAMS for
    why that's a hard cutoff rather than a real effort/budget setting, and the
    grow-and-retry-then-NaN handling below for what happens if it's not enough.
    `prefix` (the accumulated CoT so far) is sent as-is; the local llama.cpp
    server already enforces its own context-window limit and reports a 400 when
    a prompt is too long, so that's the trigger for trimming rather than
    guessing a character budget upfront - see _is_context_overflow below."""
    question = row['question']
    answer = row['answer']

    prefix_budget = None  # None = send the full prefix; only set once the server rejects it as too long
    token_budget = EVAL_PARAMS.get("max_tokens", 2000)
    max_token_budget = token_budget * 4  # hard cap so a stuck grader can't run forever
    eval_kwargs = {k: v for k, v in EVAL_PARAMS.items() if k != "max_tokens"}

    while True:
        if prefix_budget is None or len(prefix) <= prefix_budget:
            trimmed = prefix
        else:
            trimmed = "...(earlier reasoning trimmed)..." + "\n" + prefix[-prefix_budget:]
        try:
            stream = with_retry(
                client_llama_cpp.chat.completions.create,
                model=default_evaluation_model,
                messages=[
                    {"role": "system", "content": (
                        "You grade partial reasoning toward solving a task. Given the task, the "
                        "correct answer, and the reasoning so far, reply with only a single number "
                        "from 0 to 10 for how close the reasoning is to that correct answer."
                    )},
                    {"role": "user", "content": f"Task:\n{question}\n\nCorrect answer to this task:\n{answer}\n\nReasoning so far:\n{trimmed}\n\nScore real number (0-10):"},
                ],
                extra_body={"reasoning": {"enabled": True}},
                stream=True,
                max_tokens=token_budget,
                **eval_kwargs,
            )
            content_chunks = []
            reasoning_chunks = []
            reasoning_started = False
            content_started = False
            finish_reason = None
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta

                # OpenRouter-routed reasoning models stream their thinking
                # separately from the final answer, either as a plain
                # `reasoning` string or as `reasoning_details` (a list of
                # blocks, each with a "text" field) - check both, since
                # which one a given model/provider uses varies.
                reasoning_piece = _extract_reasoning_delta(delta)
                if reasoning_piece:
                    if not reasoning_started:
                        print("[reasoning] ", end="", flush=True)
                        reasoning_started = True
                    print(reasoning_piece, end="", flush=True)
                    reasoning_chunks.append(reasoning_piece)

                content_piece = delta.content
                if content_piece:
                    if reasoning_started and not content_started:
                        print("\n[answer] ", end="", flush=True)
                    content_started = True
                    print(content_piece, end="", flush=True)
                    content_chunks.append(content_piece)
            print()
            text = "".join(content_chunks).strip()
            reasoning_text = "".join(reasoning_chunks).strip()

            # If the token budget cut generation off before an answer came
            # through - either mid <think> block (models that inline
            # reasoning in `content`) or mid separate `reasoning`/
            # `reasoning_details` stream with `content` still empty - the
            # strip-then-take-last-number logic below can't tell reasoning
            # from the real verdict, so it would silently grab a stray digit
            # from the unfinished reasoning again. Grow the budget and retry
            # instead of trusting that.
            unclosed_inline_think = "<think>" in text and "</think>" not in text
            no_answer_yet = bool(reasoning_text) and not re.sub(r"(?is)<think>.*?</think>", "", text).strip()
            truncated_mid_think = finish_reason == "length" and (unclosed_inline_think or no_answer_yet)
            if truncated_mid_think and token_budget < max_token_budget:
                token_budget = min(token_budget * 2, max_token_budget)
                print(f"[score_prefix] cut off mid-<think> (finish_reason=length); retrying with max_tokens={token_budget}")
                continue
            break
        except APIStatusError as e:
            if not _is_context_overflow(e):
                raise
            next_budget = (prefix_budget // 2) if prefix_budget is not None else (len(prefix) // 2)
            if next_budget < 500:
                raise
            prefix_budget = next_budget
            continue

    answer_only = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    if ("<think>" in text and "</think>" not in text) or (not answer_only and reasoning_text):
        print(f"[score_prefix] WARNING: reasoning still truncated after retries (finish_reason={finish_reason!r}); returning NaN instead of guessing.")
        return float("nan")

    # Take the LAST number in the answer text - the model states its verdict
    # at the end, and grabbing the first number instead previously picked up
    # unrelated digits from its own reasoning.
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer_only or text)
    return float(numbers[-1]) if numbers else 0.0

def run_example(row: pd.Series) -> dict:
    cot = generate_cot(row['question'])
    parts = split_cot_by_meaning(cot)

    cumulative = ""
    scores = []
    for part in parts:
        cumulative = (cumulative + "\n" + part).strip()
        scores.append(score_prefix(row, cumulative))

    diffs = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
    lengths = [len(parts[i]) for i in range(1, len(parts))]
    weighted = [d * l for d, l in zip(diffs, lengths)]
    
    logged = [math.log(abs(w) + 1e-9) for w in weighted]

    return {
        "cot": cot,
        "parts": parts,
        "scores": scores,
        "diffs": diffs,
        "lengths": lengths,
        "weighted": weighted,
        "logged": logged,
        "mean_logged": sum(logged) / len(logged) if logged else float("nan"),
    }

def main():
    sample = df.head(N_SAMPLES)
    
    results = []
    for i, row in sample.iterrows():
        print(f"[{i}] {row['id']}")
        r = run_example(row)
        print(f"    parts={len(r['parts'])} scores={r['scores']} mean_logged={r['mean_logged']:.4f}")
        results.append(r)

    overall_mean = sum(r["mean_logged"] for r in results) / len(results)
    print(f"\noverall mean(log(|delta_score * length|)) over {len(results)} examples: {overall_mean:.4f}")

if __name__ == "__main__":
    main()