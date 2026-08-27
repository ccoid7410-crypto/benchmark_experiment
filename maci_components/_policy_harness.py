"""Runs one agent's policy code (or a standalone condition check) in a
restricted namespace.

Invoked as a subprocess by policy_executor.py with the launcher's -I -S
flags (isolated mode, no site module) - not meant to be imported or run
standalone otherwise. Reads a JSON payload from the path in argv[1] and
prints one JSON result line to stdout.

Two payload shapes:
  {"mode": "policy", "state": ..., "code": ...}
      Execs `code` against a locked-down globals dict exposing state/
      auto_move/broadcast, and returns the resulting decision.
  {"mode": "condition", "state": ..., "condition_expr": ...}
      Evaluates a single boolean expression against `state` - this is how
      an active auto_move() plan is re-checked every turn without an LLM
      call (see agent_step_mixin.py).

There is no explicit memory API (no remember()/set_notes()) - the agent's
only "memory" is the accumulating conversation history (see
agent_step_mixin.py), not anything engineered here.

This restricts the *ordinary* way in - no __import__, open, eval-of-code
(exec is used deliberately, but only through this harness), compile,
getattr - but plain Python has no true capability sandbox, so a
sufficiently adversarial snippet can still escape via object introspection
(e.g. chaining through __class__/__subclasses__). The real safety net here
is OS-level: this runs as a separate, disposable process with a hard
wall-clock timeout enforced by the caller, so a bad policy can only hang or
crash *itself*, not the simulation - it is not a hardened boundary against a
truly malicious model.
"""
import json
import sys

_builtins_ns = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
SAFE_BUILTINS = {
    name: _builtins_ns[name]
    for name in [
        # Core language features (no imports/file/network access, but the
        # rest of the language - loops, comprehensions, helper functions,
        # try/except, string formatting, etc. - is meant to be fully usable).
        "len", "range", "min", "max", "abs", "sum", "sorted", "enumerate", "zip",
        "map", "filter", "pow", "divmod", "chr", "ord", "format", "repr",
        "int", "float", "str", "bool", "list", "dict", "set", "tuple", "frozenset", "round",
        "True", "False", "None", "isinstance", "any", "all", "reversed",
        # Exception types, so try/except can name what it's catching.
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "AttributeError", "ZeroDivisionError", "StopIteration", "RuntimeError",
    ]
    if name in _builtins_ns
}


class _State:
    """Read-only view of the agent's perception, passed as `state`. Supports
    both attribute access (state.pos) and dict-style access
    (state['pos'], state.get('pos')) since models write either style."""

    def __init__(self, d):
        self._d = dict(d)
        for k, v in self._d.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return self._d[key]

    def __contains__(self, key):
        return key in self._d

    def get(self, key, default=None):
        return self._d.get(key, default)

    def keys(self):
        return self._d.keys()

    def __repr__(self):
        return f"State({self._d!r})"


def _run_condition(payload):
    state = _State(payload["state"])
    expr = payload.get("condition_expr") or "True"
    try:
        value = eval(compile(expr, "<condition>", "eval"), {"__builtins__": SAFE_BUILTINS, "state": state}, {})
        return {"result": bool(value), "error": None}
    except Exception as e:
        return {"result": None, "error": f"{type(e).__name__}: {e}"}


def _run_policy(payload):
    result = {"action": None, "blocks": 1, "broadcast": None, "until": None, "error": None}
    state = _State(payload["state"])

    def auto_move(direction, blocks=1, until=None):
        result["action"] = str(direction).upper()
        try:
            result["blocks"] = int(blocks)
        except (TypeError, ValueError):
            result["blocks"] = 1
        result["until"] = str(until).strip() if until else None

    def broadcast(text):
        result["broadcast"] = str(text)[:200]

    safe_globals = {
        "__builtins__": SAFE_BUILTINS,
        "state": state,
        "auto_move": auto_move,
        "broadcast": broadcast,
    }

    try:
        # A separate (even empty) locals dict here would make exec() treat
        # top-level statements as being inside a function body: any `def`/
        # assignment would land in that dict and vanish once exec() returns,
        # instead of being visible afterward - so globals doubles as locals.
        exec(compile(payload["code"], "<policy>", "exec"), safe_globals, safe_globals)
        # Models sometimes wrap the logic in `def decide(state): ...` out of
        # habit instead of writing top-level statements, which would
        # otherwise define the function and never call it (auto_move() never
        # fires, silently). Call it for them if that happened.
        if result["action"] is None:
            decide_fn = safe_globals.get("decide")
            if callable(decide_fn):
                decide_fn(state)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def main():
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        payload = json.load(f)

    if payload.get("mode") == "condition":
        result = _run_condition(payload)
    else:
        result = _run_policy(payload)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
