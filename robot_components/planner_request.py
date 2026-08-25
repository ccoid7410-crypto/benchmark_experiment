"""Planner request payload."""

from dataclasses import dataclass


@dataclass
class PlannerRequest:
    frame_index: int
    frame_timestamp: float
    images_b64: list[str]
    fast_result: dict
    memory: list[dict]
