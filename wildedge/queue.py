import json
import threading
import time
import uuid
from collections import deque
from enum import Enum
from pathlib import Path


class QueuePolicy(Enum):
    """Policy when queue is full."""

    STRICT = "strict"
    OPPORTUNISTIC = "opportunistic"


class FifoFullError(Exception):
    """Raised by STRICT policy when the queue is at capacity."""


class EventQueue:
    """Thread-safe FIFO queue with configurable full-queue policy."""

    def __init__(
        self,
        max_size: int = 200,
        policy: QueuePolicy = QueuePolicy.OPPORTUNISTIC,
        persist_to_disk: bool = False,
        disk_dir: str | None = None,
    ):
        self.max_size = max_size
        self.policy = policy
        self.events: deque[dict] = deque()
        self.persist_to_disk = persist_to_disk
        self.disk_dir = Path(disk_dir).expanduser() if disk_dir else None
        self._event_paths: deque[Path] = deque()
        self.lock = threading.Lock()
        self._persist_seq = 0
        if self.persist_to_disk:
            if self.disk_dir is None:
                raise ValueError("disk_dir is required when persist_to_disk=True")
            self.disk_dir.mkdir(parents=True, exist_ok=True)
            self.load_from_disk()

    def load_from_disk(self) -> None:
        assert self.disk_dir is not None
        files = sorted(self.disk_dir.glob("*.json"))
        for path in files:
            try:
                payload = json.loads(path.read_text())
                if isinstance(payload, dict):
                    self.events.append(payload)
                    self._event_paths.append(path)
                else:
                    path.unlink(missing_ok=True)
            except Exception:
                path.unlink(missing_ok=True)
        while len(self.events) > self.max_size:
            self.events.popleft()
            dropped = self._event_paths.popleft()
            dropped.unlink(missing_ok=True)

    def persist_event(self, event: dict) -> None:
        if not self.persist_to_disk:
            return
        assert self.disk_dir is not None
        filename = f"{time.time_ns():020d}-{self._persist_seq:010d}-{uuid.uuid4()}.json"
        self._persist_seq += 1
        path = self.disk_dir / filename
        path.write_text(json.dumps(event, separators=(",", ":"), ensure_ascii=True))
        self._event_paths.append(path)

    def drop_oldest_for_capacity(self) -> None:
        self.events.popleft()
        if self.persist_to_disk and self._event_paths:
            path = self._event_paths.popleft()
            path.unlink(missing_ok=True)

    def add(self, event: dict) -> None:
        with self.lock:
            if len(self.events) >= self.max_size:
                if self.policy is QueuePolicy.STRICT:
                    raise FifoFullError(
                        f"Event queue is full ({self.max_size} items). "
                        "Use QueuePolicy.OPPORTUNISTIC to silently drop oldest."
                    )
                self.drop_oldest_for_capacity()
            self.persist_event(event)
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
                if self.persist_to_disk and self._event_paths:
                    path = self._event_paths.popleft()
                    path.unlink(missing_ok=True)

    def remove_first_n(self, n: int) -> None:
        with self.lock:
            for _ in range(min(n, len(self.events))):
                self.events.popleft()
                if self.persist_to_disk and self._event_paths:
                    path = self._event_paths.popleft()
                    path.unlink(missing_ok=True)

    def length(self) -> int:
        with self.lock:
            return len(self.events)
