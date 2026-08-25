"""Thread-safe command container."""

import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SharedCommand:
    command: Optional[dict] = None
    updated_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
