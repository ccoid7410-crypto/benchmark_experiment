"""Sandboxed execution of LLM-generated policy code (Code as Policies) for
MACI agents.

Agents no longer return a fixed JSON action - each turn the model writes a
short Python snippet that calls auto_move()/broadcast() against a read-only
`state` object describing what it perceives. auto_move() can carry a stop
condition (a boolean expression re-evaluated every subsequent turn) so the
agent can keep moving the same direction for several turns without an LLM
call, until that condition fires or it gets blocked - see
agent_step_mixin.py for the turn loop that drives this.

There is no explicit memory API; "memory" is just the accumulating
conversation history. Both the policy code and the per-turn condition
re-checks run in a separate subprocess (see _policy_harness.py) with a
restricted builtins set and a hard wall-clock timeout, so a buggy or
runaway policy can't hang or corrupt the simulation process. See
_policy_harness.py's docstring for the real (limited) security model this
provides.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HARNESS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_policy_harness.py")

VALID_DIRECTIONS = {"UP", "DOWN", "LEFT", "RIGHT"}


def _run_harness(payload: dict, timeout: float):
    payload_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(payload, f)
            payload_path = f.name

        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", HARNESS_PATH, payload_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None, f"timed out after {timeout}s (likely an infinite loop)"

        if proc.returncode != 0:
            return None, f"harness crashed (exit {proc.returncode}): {proc.stderr.strip()[-500:]}"

        try:
            last_line = proc.stdout.strip().splitlines()[-1]
            return json.loads(last_line), None
        except Exception as e:
            return None, f"could not parse harness output ({e}): {proc.stdout[:500]!r}"
    finally:
        if payload_path:
            try:
                os.unlink(payload_path)
            except OSError:
                pass


def run_policy_code(code: str, state: dict, timeout: float = 15.0) -> dict:
    """Executes one agent's policy `code` against `state` in a locked-down
    subprocess. Returns a dict with keys: action (str or None), blocks (int),
    broadcast (str or None), until (str or None - a condition expression to
    re-check on later turns instead of calling the LLM again, or None for a
    plain one-off move), error (str or None - set whenever the code raised,
    timed out, or never called auto_move() with a valid direction)."""
    result, err = _run_harness({"mode": "policy", "state": state, "code": code}, timeout)
    if err:
        return _error_result(err)

    action = str(result.get("action") or "").upper()
    if action not in VALID_DIRECTIONS and not result.get("error"):
        result["error"] = f"policy code did not call auto_move() with a valid direction (got {action!r})"
    return result


def check_condition(condition_expr: str, state: dict, timeout: float = 5.0) -> dict:
    """Re-evaluates an active auto_move() plan's stop condition against the
    current `state`, in the same locked-down subprocess sandbox. Returns
    {"result": bool or None, "error": str or None} - a non-None error (or a
    None result) should be treated as "stop the plan and ask the LLM",
    since a condition that can't be trusted isn't safe to keep auto-running on."""
    result, err = _run_harness({"mode": "condition", "state": state, "condition_expr": condition_expr}, timeout)
    if err:
        return {"result": None, "error": err}
    return result


def _error_result(message: str) -> dict:
    return {"action": None, "blocks": 1, "broadcast": None, "until": None, "error": message}
