import threading
from collections import deque
from enum import Enum


class QueuePolicy(Enum):
    """Policy when queue is full."""

    STRICT = "strict"
    OPPORTUNISTIC = "opportunistic"


class FifoFullError(Exception):
    """Raised by STRICT policy when the queue is at capacity."""


class EventQueue:
    """Thread-safe FIFO queue with configurable full-queue policy."""

    def __init__(
        self, max_size: int = 200, policy: QueuePolicy = QueuePolicy.OPPORTUNISTIC
    ):
        self.max_size = max_size
        self.policy = policy
        self.events: deque[dict] = deque()
        self.lock = threading.Lock()

    def add(self, event: dict) -> None:
        with self.lock:
            if len(self.events) >= self.max_size:
                if self.policy is QueuePolicy.STRICT:
                    raise FifoFullError(
                        f"Event queue is full ({self.max_size} items). "
                        "Use QueuePolicy.OPPORTUNISTIC to silently drop oldest."
                    )
                # OPPORTUNISTIC: drop oldest to make room
                self.events.popleft()
            self.events.append(event)

    def peek(self) -> dict | None:
        with self.lock:
            return self.events[0] if self.events else None

    def peek_many(self, n: int) -> list[dict]:
        with self.lock:
            return list(self.events)[:n]

    def remove_first(self) -> None:
        with self.lock:
            if self.events:
                self.events.popleft()

    def remove_first_n(self, n: int) -> None:
        with self.lock:
            for _ in range(min(n, len(self.events))):
                self.events.popleft()

    def length(self) -> int:
        with self.lock:
            return len(self.events)
