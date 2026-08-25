"""Logging and token summary helpers for MACI models."""

from .agent_support import *


class ModelLoggingMixin:
    def log(self, message):
        """Logs message to terminal and file."""
        print(message)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(message + "\n")

    def log_llm_io(self, record):
        """Appends one full LLM call record (prompts, raw response, reasoning/CoT,
        parsed decision, token usage) as a JSON line, so every model output -
        including inline <think> chain-of-thought - is preserved on disk."""
        path = getattr(self, "llm_io_log_path", None) or "llm_io.jsonl"
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        record = dict(record)
        record.setdefault("timestamp", datetime.datetime.now().isoformat(timespec="seconds"))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def token_summary(self):
        agents = []
        totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0
        }
        for agent in self.agents:
            item = {
                "agent": self._agent_label(agent),
                "model": getattr(agent, "model_name", ""),
                "turns": getattr(agent, "turns", 0),
                "prompt_tokens": getattr(agent, "prompt_tokens", 0),
                "completion_tokens": getattr(agent, "completion_tokens", 0),
                "reasoning_tokens": getattr(agent, "reasoning_tokens", 0),
                "cached_tokens": getattr(agent, "cached_tokens", 0),
                "total_tokens": getattr(agent, "total_tokens", 0)
            }
            agents.append(item)
            for key in totals:
                totals[key] += int(item.get(key, 0))
        return {
            "total": totals,
            "agents": agents
        }
