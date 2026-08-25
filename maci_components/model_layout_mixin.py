"""Static layout capture and restore helpers for MACI models."""

from .agent_support import *


class ModelLayoutMixin:
    def _capture_static_layout(self):
        grid_snapshot = [row[:] for row in self.map_data]
        for gx, gy in getattr(self, "gates", []):
            if 0 <= gy < len(grid_snapshot) and 0 <= gx < len(grid_snapshot[gy]):
                grid_snapshot[gy][gx] = 1
        self.initial_round_layout = {
            "grid": grid_snapshot,
            "target_pos": tuple(getattr(self, "target_pos", (0, 0))),
            "fake_symbols": dict(getattr(self, "fake_symbols", {})),
            "switches": list(getattr(self, "switches", [])),
            "gates": list(getattr(self, "gates", [])),
            "interaction_rules": json.loads(json.dumps(getattr(self, "interaction_rules", {})))
        }
        return self.initial_round_layout

    def _restore_static_layout(self):
        layout = getattr(self, "initial_round_layout", None)
        if not layout:
            return
        self.map_data = [row[:] for row in layout.get("grid", self.map_data)]
        self.map_generator.grid = self.map_data
        self.target_pos = tuple(layout.get("target_pos", getattr(self, "target_pos", (0, 0))))
        self.fake_symbols = dict(layout.get("fake_symbols", {}))
        self.switches = list(layout.get("switches", []))
        self.gates = list(layout.get("gates", []))
        self.interaction_rules = json.loads(json.dumps(layout.get("interaction_rules", getattr(self, "interaction_rules", {}))))
        self.open_gates = set()
        self.gates_open = False
