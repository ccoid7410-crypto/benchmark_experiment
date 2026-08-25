"""Thread-safe bounded observation memory."""

import threading
from collections import deque
from dataclasses import dataclass, field


DEFAULT_PLANNER_MEMORY_SIZE = 8


@dataclass
class ObservationMemory:
    observations: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_PLANNER_MEMORY_SIZE))
    lock: threading.Lock = field(default_factory=threading.Lock)
