import math
import os
import re
import time

import pandas as pd
from openai import OpenAI, APIConnectionError, APIStatusError

HERE = os.path.dirname(os.path.abspath(__file__))
OPENROUTER_API_KEY = open(os.path.join(HERE, "key.txt")).read().strip()
HF_TOKEN = open(os.path.join(HERE, "hf_key.txt")).read().strip()
os.environ.setdefault("HF_TOKEN", HF_TOKEN)

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
client_llama_cpp = OpenAI(base_url="http://127.0.0.1:7654")
df = pd.read_parquet("hf://datasets/datacurve/deep-swe/data/test-00000-of-00001.parquet")

default_generation_model = 'qwen/qwen3.8-27b'
default_evaluation_model = 'ling-3.0-tiny'

# same sampling params on every call (generation and evaluation) so runs are comparable
GEN_PARAMS = dict(temperature=0.0, top_p=1.0, max_tokens=4000, seed=0)
EVAL_PARAMS = dict(temperature=0.0, top_p=1.0, max_tokens=20, seed=0)

N_SAMPLES = 3  # how many dataset rows to run; raise once the pipeline looks right


def with_retry(fn, *args, retries=3, delay=2.0, **kwargs):
    """Retry on transient network/5xx errors, which OpenRouter hits often on long batch runs."""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except (APIConnectionError, APIStatusError):
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))


def generate_cot(problem_statement: str) -> str:
    """Generation model solves the task, numbering each reasoning step so it can be split later."""
    resp = with_retry(
        client.chat.completions.create,
        model=default_generation_model,
        messages=[
            {"role": "system", "content": (
                "Solve the task. Think step by step and prefix every reasoning step with "
                "'Step N:' on its own line before giving a final answer."
            )},
            {"role": "user", "content": problem_statement},
        ],
        # many OpenRouter models are reasoning-capable and otherwise spend the token
        # budget on hidden `reasoning` before ever writing to `content`
        extra_body={"reasoning": {"enabled": False}},
        **GEN_PARAMS,
    )
    choice = resp.choices[0]
    content = choice.message.content or getattr(choice.message, "reasoning", None)
    if not content:
        raise RuntimeError(f"empty generation response (finish_reason={choice.finish_reason!r})")
    return content


def split_cot_by_meaning(cot: str) -> list[str]:
    """Divide the CoT into semantically distinct parts, one per numbered reasoning step.

    The model doesn't always follow the "Step N:" formatting instruction, so fall
    back to blank-line paragraph breaks when no numbered steps are found.
    """
    cot = cot.strip()
    parts = re.split(r"\n(?=#{0,3}\s*Step\s*\d+\s*:)", cot)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        return parts
    parts = re.split(r"\n\s*\n", cot)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else [cot]


def score_prefix(problem_statement: str, prefix: str) -> float:
    """Evaluator model rates (0-10) how close the reasoning-so-far is to a correct solution."""
    resp = with_retry(
        client.chat.completions.create,
        model=default_evaluation_model,
        messages=[
            {"role": "system", "content": (
                "You grade partial reasoning toward solving a task. Given the task and the "
                "reasoning so far, reply with only a single number from 0 to 10."
            )},
            {"role": "user", "content": f"Task:\n{problem_statement}\n\nReasoning so far:\n{prefix}\n\nScore (0-10):"},
        ],
        # the free nemotron model is a reasoning model; without this it burns the
        # whole token budget "thinking" and never emits the final number
        extra_body={"reasoning": {"enabled": False}},
        **EVAL_PARAMS,
    )
    text = resp.choices[0].message.content.strip()
    match = re.search(r"-?\d+(\.\d+)?", text)
    return float(match.group()) if match else 0.0


def run_example(problem_statement: str) -> dict:
    cot = generate_cot(problem_statement)
    parts = split_cot_by_meaning(cot)

    cumulative = ""
    scores = []
    for part in parts:
        cumulative = (cumulative + "\n" + part).strip()
        scores.append(score_prefix(problem_statement, cumulative))

    diffs = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
    lengths = [len(parts[i]) for i in range(1, len(parts))]
    weighted = [d * l for d, l in zip(diffs, lengths)]
    # a step can lower the score (negative diff), and log needs a positive input,
    # so we log the magnitude of the weighted delta rather than the signed value
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
        print(f"[{i}] {row['instance_id']}")
        r = run_example(row["problem_statement"])
        print(f"    parts={len(r['parts'])} scores={r['scores']} mean_logged={r['mean_logged']:.4f}")
        results.append(r)

    overall_mean = sum(r["mean_logged"] for r in results) / len(results)
    print(f"\noverall mean(log(|delta_score * length|)) over {len(results)} examples: {overall_mean:.4f}")


if __name__ == "__main__":
    main()
