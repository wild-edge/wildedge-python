from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from wildedge.events.common import add_optional_fields


class ErrorCode(str, Enum):
    OOM = "OOM"
    CORRUPTED_MODEL = "CORRUPTED_MODEL"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    UNSUPPORTED_OP = "UNSUPPORTED_OP"
    THERMAL_SHUTDOWN = "THERMAL_SHUTDOWN"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    UNKNOWN = "UNKNOWN"


@dataclass
class ErrorEvent:
    model_id: str
    error_code: str | ErrorCode
    error_message: str | None = None
    stack_trace_hash: str | None = None
    related_event_id: str | None = None
    http_status: int | None = None
    provider_error_code: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    run_id: str | None = None
    agent_id: str | None = None
    step_index: int | None = None
    conversation_id: str | None = None
    context: dict[str, Any] | None = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        code = (
            self.error_code.value
            if isinstance(self.error_code, ErrorCode)
            else self.error_code
        )
        error_data = add_optional_fields(
            {"error_code": code},
            {
                "error_message": self.error_message,
                "stack_trace_hash": self.stack_trace_hash,
                "related_event_id": self.related_event_id,
                "http_status": self.http_status,
                "provider_error_code": self.provider_error_code,
            },
        )
        event = {
            "event_id": self.event_id,
            "event_type": "error",
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "error": error_data,
        }
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
                "attributes": self.context,
            },
        )
        return event
