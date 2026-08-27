"""MACI judge: scores a completed run's manifest.jsonl.

Reuses the same robustness patterns as benchmark.py's HLE evaluator
(streaming, smart retry, context-overflow shrinking, truncation-safe score
extraction) but judges MACI agent turns instead: for each turn, a simplified
map-state summary and the acting agent's CoT/reasoning are shown together to
a judge model, which combines state-consistency (no hallucinated walls,
positions, or events) and action quality into a single 0-10 score.

Per-turn scores are averaged into a raw 0-10 index, then linearly scaled to a
0-5000 benchmark score calibrated so a reference model (upstage/solar-pro4,
by default) defines 100 points - run once on a solar-pro-4 manifest with
--set-baseline to establish that reference before scoring other runs.

Usage:
    python maci_judge.py path/to/manifest.jsonl --set-baseline
    python maci_judge.py path/to/other_manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re

from maci_components.agent_support import (
    build_llm_client,
    extract_reasoning_delta,
    is_context_overflow_error,
    with_retry,
)
from openai import APIStatusError

HERE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_FILE = os.path.join(HERE_DIR, "maci_judge_baseline.json")

MAX_SCORE = 5000
BASELINE_TARGET = 100
BASELINE_MODEL = "upstage/solar-pro4"
DEFAULT_JUDGE_MODEL = "nvidia/nemotron-3.5-lightning"

JUDGE_PARAMS = dict(temperature=0.0, top_p=1.0, seed=0)
DEFAULT_TOKEN_BUDGET = 2000

SYSTEM_PROMPT = (
    "You are an expert judge evaluating one turn of a cooperative multi-agent "
    "maze-solving AI. You will be shown a simplified summary of the current map "
    "state and one agent's reasoning, stated reason, and chosen action for its "
    "turn. Judge two things together as a single combined score: "
    "(1) consistency - does the reasoning accurately reflect the actual map "
    "state, with no hallucinated walls, positions, or events; "
    "(2) quality - is the resulting action sound and does it make progress "
    "toward the shared goal. "
    "Reply with only a single number from 0 to 10."
)


def build_map_state_summary(record: dict) -> str:
    """Simplifies a manifest.jsonl turn record down to the entities that
    matter for judging (target, agents, gate/switch state, recent comms) -
    the full wall grid is left out on purpose, matching the instruction to
    simplify the situation before handing it to the judge."""
    lines = [f"Step {record.get('step')} ({record.get('group', '')})"]
    lines.append(f"Target position: {record.get('target')}")
    lines.append(f"Gates open: {record.get('gates_open')}; open gate positions: {record.get('open_gates', [])}")
    for a in record.get("agents", []):
        lines.append(f"Agent {a.get('agent')} ({a.get('model', '')}): position={a.get('position')} done={a.get('done')}")
    interaction_rules = record.get("interaction_rules")
    if interaction_rules:
        lines.append(f"Interaction rules: {json.dumps(interaction_rules, ensure_ascii=False)}")
    comms = (record.get("recent_communication") or [])[-5:]
    if comms:
        lines.append("Recent communication:")
        for c in comms:
            targets = ",".join(c.get("to", []))
            lines.append(f"  {c.get('from', '?')} -> {targets}: {c.get('message', '')} ({c.get('meaning', '')})")
    return "\n".join(lines)


def _build_user_prompt(map_state_text: str, agent_record: dict, reasoning_text: str) -> str:
    decision = agent_record.get("last_decision") or {}
    return (
        f"=== MAP STATE ===\n{map_state_text}\n\n"
        f"=== AGENT UNDER REVIEW ===\n"
        f"Agent: {agent_record.get('agent')}\n"
        f"Position: {agent_record.get('position')}\n"
        f"Chain-of-thought / reasoning: {reasoning_text or '(none captured for this turn)'}\n"
        f"Stated reason: {decision.get('reason', '')}\n"
        f"Action taken: {decision.get('action', '')} x{decision.get('blocks', '')}\n"
        f"Notes: {decision.get('notes', '')}\n"
        f"Broadcast: {decision.get('broadcast_message', '')}\n\n"
        f"Score (0-10):"
    )


def judge_turn(client, judge_model: str, map_state_text: str, agent_record: dict, log=None) -> float:
    """Streams one judge call for a single agent turn. Mirrors benchmark.py's
    score_prefix: grows the token budget (up to 4x) and retries if the model
    is cut off mid-reasoning, returns NaN instead of guessing if it still
    doesn't fit, and reacts to (rather than pre-guesses) context overflow by
    shrinking the reasoning text first."""
    if log is None:
        def log(msg):
            print(msg, flush=True)
    decision = agent_record.get("last_decision") or {}
    full_reasoning = decision.get("reasoning_content") or ""
    raw_response = decision.get("raw_response") or ""

    reasoning_budget = None  # None = send the full CoT; only shrink once the server rejects it
    token_budget = DEFAULT_TOKEN_BUDGET
    max_token_budget = token_budget * 4
    stream_error_count = 0
    max_stream_errors = 3

    while True:
        reasoning_text = full_reasoning
        if reasoning_budget is not None and len(reasoning_text) > reasoning_budget:
            reasoning_text = "...(earlier reasoning trimmed)...\n" + reasoning_text[-reasoning_budget:]
        if not reasoning_text and raw_response:
            reasoning_text = raw_response
        user_prompt = _build_user_prompt(map_state_text, agent_record, reasoning_text)

        try:
            stream = with_retry(
                client.chat.completions.create,
                model=judge_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                extra_body={"reasoning": {"enabled": True}},
                stream=True,
                max_tokens=token_budget,
                **JUDGE_PARAMS,
            )
            content_chunks = []
            reasoning_chunks = []
            finish_reason = None
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                reasoning_piece = extract_reasoning_delta(delta)
                if reasoning_piece:
                    reasoning_chunks.append(reasoning_piece)
                if delta.content:
                    content_chunks.append(delta.content)
            text = "".join(content_chunks).strip()
            judge_reasoning_text = "".join(reasoning_chunks).strip()

            unclosed_inline_think = "<think>" in text and "</think>" not in text
            no_answer_yet = bool(judge_reasoning_text) and not re.sub(r"(?is)<think>.*?</think>", "", text).strip()
            truncated = finish_reason == "length" and (unclosed_inline_think or no_answer_yet)
            if truncated and token_budget < max_token_budget:
                token_budget = min(token_budget * 2, max_token_budget)
                log(f"[maci_judge] judge cut off mid-reasoning; retrying with max_tokens={token_budget}")
                continue
            break
        except APIStatusError as e:
            if not is_context_overflow_error(e):
                raise
            next_budget = (reasoning_budget // 2) if reasoning_budget is not None else (len(full_reasoning) // 2)
            if next_budget < 200:
                log("[maci_judge] WARNING: prompt too long even after trimming reasoning; scoring as NaN.")
                return float("nan")
            reasoning_budget = next_budget
            log(f"[maci_judge] judge prompt too long; trimming reasoning to last {reasoning_budget} chars and retrying...")
            continue
        except Exception as e:
            # with_retry only guards the initial request; a dropped connection
            # mid-stream (e.g. httpx/httpcore RemoteProtocolError) surfaces
            # here instead, while iterating the response - without this it
            # would crash the whole run over one flaky connection.
            stream_error_count += 1
            if stream_error_count > max_stream_errors:
                log(f"[maci_judge] WARNING: judge stream kept failing ({type(e).__name__}: {e}); returning NaN for this turn.")
                return float("nan")
            log(f"[maci_judge] judge stream interrupted ({type(e).__name__}: {e}); retrying ({stream_error_count}/{max_stream_errors})...")
            continue

    answer_only = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    if ("<think>" in text and "</think>" not in text) or (not answer_only and judge_reasoning_text):
        log(f"[maci_judge] WARNING: judge reasoning still truncated after retries (finish_reason={finish_reason!r}); returning NaN.")
        return float("nan")

    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer_only or text)
    if not numbers:
        return float("nan")
    score = float(numbers[-1])
    return max(0.0, min(10.0, score))


def score_run(manifest_path, client, judge_model, agent_filter=None, max_turns=None, log=None):
    if log is None:
        def log(msg):
            print(msg, flush=True)
    scores = []
    per_turn = []
    actor_models = set()

    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    if max_turns:
        lines = lines[:max_turns]

    for line in lines:
        record = json.loads(line)
        map_state_text = build_map_state_summary(record)
        for agent_record in record.get("agents", []):
            label = agent_record.get("agent")
            if agent_filter and label != agent_filter:
                continue
            decision = agent_record.get("last_decision") or {}
            if not decision.get("turn"):
                continue  # no decision made yet at this snapshot (e.g. the initial turn-0 image)
            actor_models.add(agent_record.get("model", ""))
            score = judge_turn(client, judge_model, map_state_text, agent_record, log=log)
            log(f"[maci_judge] step={record.get('step')} agent={label} score={score}")
            per_turn.append({"step": record.get("step"), "agent": label, "score": score})
            if score == score:  # skip NaN
                scores.append(score)

    raw_avg = sum(scores) / len(scores) if scores else float("nan")
    return {
        "raw_avg": raw_avg,
        "n_scored": len(scores),
        "n_total": len(per_turn),
        "per_turn": per_turn,
        "actor_models": sorted(m for m in actor_models if m),
    }


def load_baselines() -> dict:
    if os.path.exists(BASELINE_FILE):
        try:
            with open(BASELINE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_baseline(model_name: str, raw_avg: float) -> None:
    baselines = load_baselines()
    baselines[model_name] = raw_avg
    with open(BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(baselines, f, ensure_ascii=False, indent=2)


def scale_score(raw_avg: float):
    """Returns (scaled_score, baseline_raw) if a solar-pro-4 baseline has
    been recorded, else (None, None). Calibration is (raw_avg / baseline_raw)
    * BASELINE_TARGET, capped at MAX_SCORE - so the baseline run itself
    always scores exactly BASELINE_TARGET, and better/worse runs scale
    proportionally around it."""
    baseline_raw = load_baselines().get(BASELINE_MODEL)
    if not baseline_raw:
        return None, None
    scaled = (raw_avg / baseline_raw) * BASELINE_TARGET
    return min(MAX_SCORE, scaled), baseline_raw


def main():
    parser = argparse.ArgumentParser(description="Score a completed MACI run's manifest.jsonl with a judge model.")
    parser.add_argument("manifest", help="path to manifest.jsonl")
    parser.add_argument("--agent", default=None, help="only judge this agent label (e.g. A); default: all agents")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-turns", type=int, default=None, help="limit how many manifest lines to judge (for a quick check)")
    parser.add_argument("--set-baseline", action="store_true", help=f"register this run's average as the {BASELINE_MODEL} {BASELINE_TARGET}-point baseline")
    parser.add_argument("--force-baseline", action="store_true", help="allow --set-baseline even if the manifest's actor model doesn't look like BASELINE_MODEL")
    args = parser.parse_args()

    client, _base_url, _api_key = build_llm_client(args.provider, args.api_key, args.base_url)
    result = score_run(args.manifest, client, args.judge_model, agent_filter=args.agent, max_turns=args.max_turns)

    print(f"\nActor model(s) in this run: {result['actor_models']}")
    print(f"Raw average judge score: {result['raw_avg']:.3f} / 10 over {result['n_scored']}/{result['n_total']} scored turns")

    if args.set_baseline:
        actor_matches = any(BASELINE_MODEL in m for m in result["actor_models"])
        if not actor_matches and not args.force_baseline:
            print(
                f"[REFUSED] --set-baseline expects a manifest actually run with {BASELINE_MODEL}, "
                f"but this manifest's actor model(s) were {result['actor_models']}. "
                f"Re-run with a {BASELINE_MODEL} manifest, or pass --force-baseline if this is intentional."
            )
        else:
            save_baseline(BASELINE_MODEL, result["raw_avg"])
            print(f"Saved baseline for {BASELINE_MODEL}: raw {result['raw_avg']:.3f} -> now defines {BASELINE_TARGET} points")

    scaled, baseline_raw = scale_score(result["raw_avg"])
    if scaled is None:
        fallback = (result["raw_avg"] / 10) * MAX_SCORE if result["raw_avg"] == result["raw_avg"] else float("nan")
        print(f"[WARNING] No {BASELINE_MODEL} baseline recorded yet - run this judge on a {BASELINE_MODEL} manifest with --set-baseline first.")
        print(f"Uncalibrated fallback score (linear 0-10 -> 0-{MAX_SCORE}): {fallback:.1f} / {MAX_SCORE}")
    else:
        print(f"Benchmark score: {scaled:.1f} / {MAX_SCORE}  (baseline raw={baseline_raw:.3f} => {BASELINE_MODEL} = {BASELINE_TARGET})")


if __name__ == "__main__":
    main()
