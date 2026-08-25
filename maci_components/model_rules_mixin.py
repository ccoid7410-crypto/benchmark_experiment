"""Interaction rule helpers for MACI models."""

from .agent_support import *


class ModelRulesMixin:
    def _pos_key(self, pos):
        return f"{pos[0]},{pos[1]}"

    def _agent_label(self, agent):
        try:
            return chr(65 + list(self.agents).index(agent))
        except ValueError:
            return str(agent.unique_id)

    def _allowed_agents_for(self, rule_name, pos):
        rules = getattr(self, "interaction_rules", {}) or {}
        allowed = rules.get(rule_name, {}).get(self._pos_key(pos), ["*"])
        if isinstance(allowed, str):
            allowed = [a.strip().upper() for a in allowed.split(",") if a.strip()]
        return allowed or ["*"]

    def can_agent_use_switch(self, agent, switch_pos):
        allowed = self._allowed_agents_for("switch_agents", switch_pos)
        return "*" in allowed or self._agent_label(agent) in allowed

    def can_agent_use_gate(self, agent, gate_pos):
        allowed = self._allowed_agents_for("gate_agents", gate_pos)
        return "*" in allowed or self._agent_label(agent) in allowed

    def linked_gates_for_switch(self, switch_pos):
        links = (getattr(self, "interaction_rules", {}) or {}).get("switch_links", {})
        linked = links.get(self._pos_key(switch_pos), ["*"])
        if isinstance(linked, str):
            linked = [g.strip() for g in linked.split(";") if g.strip()]
        if not linked or "*" in linked:
            return list(self.gates)
        result = []
        for item in linked:
            try:
                gx, gy = [int(part.strip()) for part in str(item).split(",", 1)]
                gate_pos = (gx, gy)
                if gate_pos in self.gates:
                    result.append(gate_pos)
            except Exception:
                continue
        return result
