from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    OOM = "OOM"
    CORRUPTED_MODEL = "CORRUPTED_MODEL"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    UNSUPPORTED_OP = "UNSUPPORTED_OP"
    THERMAL_SHUTDOWN = "THERMAL_SHUTDOWN"
    UNKNOWN = "UNKNOWN"


@dataclass
class ErrorEvent:
    model_id: str
    error_code: str | ErrorCode
    error_message: str | None = None
    stack_trace_hash: str | None = None
    related_event_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    run_id: str | None = None
    agent_id: str | None = None
    step_index: int | None = None
    conversation_id: str | None = None
    attributes: dict[str, Any] | None = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        code = (
            self.error_code.value
            if isinstance(self.error_code, ErrorCode)
            else self.error_code
        )
        error_data: dict[str, Any] = {"error_code": code}
        if self.error_message is not None:
            error_data["error_message"] = self.error_message
        if self.stack_trace_hash is not None:
            error_data["stack_trace_hash"] = self.stack_trace_hash
        if self.related_event_id is not None:
            error_data["related_event_id"] = self.related_event_id

        event = {
            "event_id": self.event_id,
            "event_type": "error",
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "error": error_data,
        }
        from wildedge.events.common import add_optional_fields

        add_optional_fields(
            event,
            {
                "trace_id": self.trace_id,
                "span_id": self.span_id,
                "parent_span_id": self.parent_span_id,
                "run_id": self.run_id,
                "agent_id": self.agent_id,
                "step_index": self.step_index,
                "conversation_id": self.conversation_id,
                "attributes": self.attributes,
            },
        )
        return event
