"""Thread-safe shared movement policy."""

import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SharedPolicy:
    command: Optional[dict] = None
    expires_at: float = 0.0
    updated_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
