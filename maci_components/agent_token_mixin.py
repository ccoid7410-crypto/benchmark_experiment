"""Token accounting helpers for MACI agents."""

from .agent_support import *


class AgentTokenMixin:
    def _record_token_usage(self, usage, source="step"):
        record = extract_token_usage(usage)
        record["source"] = source
        record["turn"] = self.turns
        self.prompt_tokens += record["prompt_tokens"]
        self.completion_tokens += record["completion_tokens"]
        self.reasoning_tokens += record["reasoning_tokens"]
        self.cached_tokens += record["cached_tokens"]
        self.total_tokens += record["total_tokens"]
        self.last_token_usage = record
        self.token_history.append(record)
        return record
