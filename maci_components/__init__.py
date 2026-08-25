"""Feature modules for the MACI simulation."""

from .agent_state import AgentState
from .map_generator import MapGenerator
from .mission_state import MissionState


def __getattr__(name):
    if name == "MACI_Agent":
        from .maci_agent import MACI_Agent

        return MACI_Agent
    if name == "MACI_Model":
        from .maci_model import MACI_Model

        return MACI_Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MACI_Agent",
    "MACI_Model",
    "MapGenerator",
    "AgentState",
    "MissionState",
]
