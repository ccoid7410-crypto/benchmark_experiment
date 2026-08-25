"""Movement validation helpers for MACI agents."""

from .agent_support import *


class AgentMovementMixin:
    def _movement_block_reason(self, pos):
        x, y = pos
        if x < 0 or x >= self.model.width or y < 0 or y >= self.model.height:
            return "BLOCKED: out of bounds"

        if hasattr(self.model, 'gates') and pos in self.model.gates:
            if pos not in getattr(self.model, "open_gates", set()):
                return "BLOCKED: gate is closed; needs switch support"
            if not self.model.can_agent_use_gate(self, pos):
                return "BLOCKED: this agent is not allowed to use this gate"
            return ""

        if hasattr(self.model, 'fake_symbols') and self.model.fake_symbols.get(pos) == 'D':
            if "Key" not in self.structured_memory["inventory"]:
                return "BLOCKED: locked door requires key"
            return ""

        if pos == getattr(self.model, "target_pos", None):
            return ""
        if hasattr(self.model, 'fake_symbols') and pos in self.model.fake_symbols:
            return ""
        if self.model.map_data[y][x] == 0:
            return ""
        return "BLOCKED: wall"

    def _can_move_to(self, pos):
        return self._movement_block_reason(pos) == ""
