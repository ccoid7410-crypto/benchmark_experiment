"""Broadcast and tile interaction helpers for MACI agents."""

from .agent_support import *


class AgentCommunicationMixin:
    def _normalize_broadcast_message(self, message):
        if not getattr(self.model, "coded_communication", False):
            text = str(message or "").strip()
            return text[:160]
        text = str(message or "").strip().upper()
        if not text:
            return "N" if "N" in getattr(self.model, "allowed_broadcast_codes", DEFAULT_BROADCAST_CODES) else ""
        allowed_codes = "".join(re.escape(code) for code in getattr(self.model, "allowed_broadcast_codes", DEFAULT_BROADCAST_CODES))
        match = re.match(rf"^([{allowed_codes}])\s*(?:10|[0-9])?$", text)
        if not match:
            return "N" if "N" in getattr(self.model, "allowed_broadcast_codes", DEFAULT_BROADCAST_CODES) else ""
        code = match.group(1)
        return code

    def _tile_symbol_at(self, pos):
        if hasattr(self.model, 'target_pos') and pos == self.model.target_pos:
            return 'F'
        if hasattr(self.model, 'switches') and pos in self.model.switches:
            return 'S'
        if hasattr(self.model, 'gates') and pos in self.model.gates:
            return 'G'
        if hasattr(self.model, 'fake_symbols') and pos in self.model.fake_symbols:
            return self.model.fake_symbols[pos]
        return None

    def _apply_tile_interactions(self):
        events = []
        symbol = self._tile_symbol_at(self.pos)

        if symbol == 'K':
            if "Key" not in self.structured_memory["inventory"]:
                self.structured_memory["inventory"].append("Key")
            if hasattr(self.model, 'fake_symbols') and self.pos in self.model.fake_symbols:
                del self.model.fake_symbols[self.pos]
            events.append(f"Picked up Key at {self.pos}")

        elif symbol == 'D' and "Key" in self.structured_memory["inventory"]:
            self.structured_memory["inventory"].remove("Key")
            if hasattr(self.model, 'fake_symbols') and self.pos in self.model.fake_symbols:
                del self.model.fake_symbols[self.pos]
            events.append(f"Unlocked Door at {self.pos}")

        elif symbol == 'S':
            if self.model.can_agent_use_switch(self, self.pos):
                events.append(f"Standing on Switch at {self.pos}")
            else:
                events.append(f"Cannot activate Switch at {self.pos}")

        if hasattr(self.model, "update_gates"):
            self.model.update_gates()

        for event in events:
            self.model.log(f"> [Agent {self.unique_id}] {event}")

        return events
