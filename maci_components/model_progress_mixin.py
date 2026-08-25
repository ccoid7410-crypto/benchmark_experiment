"""Simulation step and gate update logic for MACI models."""

from .agent_support import *


class ModelProgressMixin:
    def step(self):
        """
        Executes one step for all agents.
        Includes Gate/Switch logic update.
        """
        self.log(f"\n>>> Model Step Starting...")

        self.update_gates()

        # Proceed with agent steps
        self.agents.shuffle_do("step")

        self.update_gates()

    def update_gates(self):
        """Opens linked gates while an allowed agent is standing on a switch."""
        if not hasattr(self, "open_gates"):
            self.open_gates = set()
        if not hasattr(self, "interaction_rules"):
            self.interaction_rules = {
                "switch_agents": {},
                "gate_agents": {},
                "switch_links": {}
            }
        open_gates = set()
        for agent in self.agents:
            if agent.pos in self.switches:
                if self.can_agent_use_switch(agent, agent.pos):
                    open_gates.update(self.linked_gates_for_switch(agent.pos))
        
        if open_gates != self.open_gates:
            self.open_gates = open_gates
            self.gates_open = bool(open_gates)
            for gx, gy in self.gates:
                self.map_data[gy][gx] = 0 if (gx, gy) in self.open_gates else 1
            status = ", ".join([self._pos_key(pos) for pos in sorted(self.open_gates)]) if self.open_gates else "none"
            self.log(f">>> [SYSTEM] Open gates: {status}")
