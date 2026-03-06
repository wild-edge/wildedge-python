from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from wildedge.logging import logger


class DeadLetterStore:
    """File-backed dead-letter store with capped batch count."""

    def __init__(
        self,
        *,
        enabled: bool,
        directory: str,
        max_batches: int,
    ) -> None:
        self.enabled = enabled
        self.directory = Path(directory).expanduser()
        self.max_batches = max_batches
        self._lock = threading.Lock()
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        reason: str,
        events: list[dict[str, Any]],
        batch_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        payload = {
            "id": str(uuid.uuid4()),
            "at_unix": time.time(),
            "reason": reason,
            "batch_id": batch_id,
            "event_count": len(events),
            "details": details or {},
            "events": events,
        }
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()
        filename = f"{int(payload['at_unix'] * 1000)}-{payload['id']}.json"
        path = self.directory / filename
        with self._lock:
            path.write_bytes(data)
            self.enforce_limit()

    def enforce_limit(self) -> None:
        if self.max_batches <= 0:
            for path in sorted(self.directory.glob("*.json")):
                path.unlink(missing_ok=True)
            return
        files = sorted(self.directory.glob("*.json"))
        overflow = len(files) - self.max_batches
        if overflow <= 0:
            return
        for path in files[:overflow]:
            path.unlink(missing_ok=True)
        logger.debug(
            "wildedge: dead-letter cap enforced, removed %d old batch files",
            overflow,
        )
