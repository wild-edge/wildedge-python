from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ModelUnloadEvent:
    model_id: str
    duration_ms: int
    reason: str
    memory_freed_bytes: int | None = None
    peak_memory_bytes: int | None = None
    uptime_ms: int | None = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        unload_data: dict[str, Any] = {
            "duration_ms": self.duration_ms,
            "reason": self.reason,
        }
        for k, v in [
            ("memory_freed_bytes", self.memory_freed_bytes),
            ("peak_memory_bytes", self.peak_memory_bytes),
            ("uptime_ms", self.uptime_ms),
        ]:
            if v is not None:
                unload_data[k] = v

        return {
            "event_id": self.event_id,
            "event_type": "model_unload",
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "unload": unload_data,
        }
