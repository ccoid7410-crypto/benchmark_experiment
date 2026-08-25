"""Per-agent UI state for the Streamlit protocol demo."""

from dataclasses import dataclass

@dataclass
class AgentState:
    name: str
    role: str
    color: str
    station: str
    pattern: str = "dark"
    intention: str = "대기"
    interpretation: str = "관측 전"
    confidence: int = 0
