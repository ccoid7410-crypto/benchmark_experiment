"""Single-agent behavior for Project MACI."""

from .agent_support import *
from .agent_token_mixin import AgentTokenMixin
from .agent_communication_mixin import AgentCommunicationMixin
from .agent_movement_mixin import AgentMovementMixin
from .agent_prompt_mixin import AgentPromptMixin
from .agent_perception_mixin import AgentPerceptionMixin
from .agent_pathfinding_mixin import AgentPathfindingMixin
from .agent_step_mixin import AgentStepMixin


class MACI_Agent(
    AgentTokenMixin,
    AgentCommunicationMixin,
    AgentMovementMixin,
    AgentPromptMixin,
    AgentPerceptionMixin,
    AgentPathfindingMixin,
    AgentStepMixin,
    mesa.Agent,
):
    """
    An AI agent representing a single entity navigating the maze.
    Each agent can have its own specific LLM model assigned to it.
    """
    def __init__(self, model, config, thinking_effort="medium"):
        super().__init__(model)
        self.llm_client = model.llm_client
        self.model_name = config.get("model_name", "gpt-4o-mini")
        self.thinking_effort = thinking_effort
        self.prompt_profile = config.get("prompt_profile", "gpt")
        self.custom_system_prompt = strip_numeric_suffix_prompt_text(config.get("custom_system_prompt", ""), include_default=False)
        self.symbol_space_prompt = strip_numeric_suffix_prompt_text(config.get("symbol_space_prompt", ""), include_default=True)
        
        # Constraints & Parameters
        self.vision_range = config.get("vision_range", 5)
        self.speed_limit = config.get("speed_limit", 1)
        self.map_share_radius = config.get("map_share_radius", 0)


        self.optimization_mode = config.get("optimization_mode", False)
        self.coded_communication = bool(config.get("coded_communication", getattr(model, "coded_communication", False)))
        self.prompt_addition = config.get("prompt_addition", "")
        self.optimization_base_prompt = config.get("optimization_base_prompt", "")
        self.prompt_addition_source = config.get("prompt_addition_source", "")
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.last_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0
        }
        self.token_history = []
        self.turns = 0
        self.last_decision = {
            "turn": 0,
            "reason": "",
            "action": "",
            "blocks": 0,
            "notes": "",
            "broadcast_message": "",
            "raw_response": "",
            "reasoning_content": "",
            "start_position": None,
            "end_position": None,
            "blocked": False
        }

        self.messages = []
        self.action_history = []
        self.is_done = False
        self.memory = "Just started."
        self.memory_history = []
        self.structured_memory = {
            "path": [],
            "landmarks": {}, # Symbol -> [x, y]
            "frontier_memory": [], # Known open tiles adjacent to unknown space
            "blocked_pos": [],
            "private_codebook": {}, # Your self-defined symbols -> meaning
            "communication_space": {},
            "symbol_space_notes": self.symbol_space_prompt,
            "inventory": [] # Collected items like Keys
        }



        self.known_map = {}
        self.inbox = []
















