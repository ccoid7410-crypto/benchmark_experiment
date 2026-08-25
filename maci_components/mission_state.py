"""Shared mission state for the Streamlit protocol demo."""

from dataclasses import dataclass, field
from typing import Dict, List

from .agent_state import AgentState

@dataclass
class MissionState:
    turn: int = 0
    phase: int = 0
    mode: str = "6x6 시각 패턴"
    object_locked: bool = False
    joint_lift: bool = False
    safety_locked: bool = False
    device_done: bool = False
    delivered: bool = False
    ambiguity: int = 0
    idle_turns: int = 0
    repair_count: int = 0
    agents: Dict[str, AgentState] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)
