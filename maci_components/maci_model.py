"""Simulation model and environment orchestration for Project MACI."""

from .agent_support import *
from .maci_agent import MACI_Agent
from .model_layout_mixin import ModelLayoutMixin
from .model_rules_mixin import ModelRulesMixin
from .model_progress_mixin import ModelProgressMixin
from .model_placement_mixin import ModelPlacementMixin
from .model_logging_mixin import ModelLoggingMixin
from .model_success_debrief_mixin import ModelSuccessDebriefMixin
from .model_reflection_mixin import ModelReflectionMixin
from .model_restart_mixin import ModelRestartMixin


class MACI_Model(
    ModelLayoutMixin,
    ModelRulesMixin,
    ModelProgressMixin,
    ModelPlacementMixin,
    ModelLoggingMixin,
    ModelSuccessDebriefMixin,
    ModelReflectionMixin,
    ModelRestartMixin,
    mesa.Model,
):
    """
    The main environment model managing the grid, agents, and target.
    """
    def __init__(self, num_agents, map_generator, agent_configs, thinking_effort="medium", provider="openai", api_key="", base_url=None, optimization_mode=False, log_file=None, llm_io_log_path=None):
        super().__init__()

        self.num_agents = num_agents
        self.map_generator = map_generator
        self.map_data = map_generator.grid
        self.width = map_generator.width
        self.height = map_generator.height
        self.thinking_effort = thinking_effort
        self.optimization_mode = optimization_mode
        self.log_file = log_file
        self.provider = str(provider or "openai").lower()
        self.api_key = api_key
        self.base_url = base_url
        self.json_response_format_supported = self.provider not in RELAXED_API_PROVIDERS
        self.reasoning_effort_supported = self.provider not in RELAXED_API_PROVIDERS
        self.llm_io_log_path = llm_io_log_path or (
            os.path.join(os.path.dirname(log_file), "llm_io.jsonl") if log_file and os.path.dirname(log_file)
            else "llm_io.jsonl"
        )
        self.communication_log = []
        self.communication_improvements = []
        self.current_communication_guideline = ""
        self.strategy_improvements = []
        self.reflection_consultations = []
        self.success_debrief_done = False
        self.success_debrief = {
            "discussion": [],
            "keywords": [],
            "communication_improvements": []
        }

        if log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"SIMULATION START: {datetime.datetime.now()}\n")
                f.write(f"{'='*60}\n")

        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=False)

        self.llm_client, self.base_url, _resolved_api_key = build_llm_client(self.provider, api_key, base_url)

        empty_spaces = map_generator.get_empty_spaces()

        self.fake_symbols = {}
        self.gates = []
        self.switches = []
        self.open_gates = set()
        self.gates_open = False
        self.interaction_rules = {
            "switch_agents": {},
            "gate_agents": {},
            "switch_links": {}
        }
        
        # Determine target 'F' position (empty space far from center)
        if empty_spaces:
            # For Room maps, try to pick a room for F and another for agents
            if hasattr(self.map_generator, 'rooms') and len(self.map_generator.rooms) >= 2:
                rooms = list(self.map_generator.rooms)
                random.shuffle(rooms)
                
                # Target F room
                f_room = rooms.pop()
                rx, ry, rw, rh = f_room
                f_spaces = [(x, y) for y in range(ry, ry + rh) for x in range(rx, rx + rw) if (x, y) in empty_spaces]
                self.target_pos = random.choice(f_spaces) if f_spaces else random.choice(empty_spaces)
                if self.target_pos in empty_spaces: empty_spaces.remove(self.target_pos)
                
                # Cooperation Element: Add a Gate at the entrance of F room and a Switch elsewhere
                # Gate: Just pick a spot near the target or at room exit
                gate_pos = (rx + rw // 2, ry) if 0 <= ry < self.height else None
                if gate_pos and gate_pos in empty_spaces:
                    self.gates.append(gate_pos)
                    # Initially, gates are walls
                    self.map_data[gate_pos[1]][gate_pos[0]] = 1
                
                # Switch room
                s_room = rooms.pop()
                srx, sry, srw, srh = s_room
                s_spaces = [(x, y) for y in range(sry, sry + srh) for x in range(srx, srx + srw) if (x, y) in empty_spaces]
                if s_spaces:
                    self.switches.append(random.choice(s_spaces))
            else:
                self.target_pos = random.choice(empty_spaces)
                empty_spaces.remove(self.target_pos)
            
            fake_chars = ['X', 'Y', 'Z', 'W', 'K', 'P']
            num_fakes = min(len(empty_spaces), 5)
            for i in range(num_fakes):
                fake_pos = random.choice(empty_spaces)
                empty_spaces.remove(fake_pos)
                self.fake_symbols[fake_pos] = fake_chars[i]
        else:
            self.target_pos = (0, 0)

        self._capture_static_layout()

        self.placed_agent_positions = []
        self._place_agents(agent_configs, empty_spaces)


























