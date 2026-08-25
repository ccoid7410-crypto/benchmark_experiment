"""Thread-safe camera frame container."""

import threading
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SharedFrame:
    frame: Optional[Any] = None
    timestamp: float = 0.0
    index: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
