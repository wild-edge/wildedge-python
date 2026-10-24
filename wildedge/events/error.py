from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from wildedge import constants
from wildedge.events.common import add_optional_fields


class ErrorCode(str, Enum):
    OOM = "OOM"
    CORRUPTED_MODEL = "CORRUPTED_MODEL"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    UNSUPPORTED_OP = "UNSUPPORTED_OP"
    THERMAL_SHUTDOWN = "THERMAL_SHUTDOWN"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    CONTEXT_LENGTH_EXCEEDED = "CONTEXT_LENGTH_EXCEEDED"
    CONTENT_FILTERED = "CONTENT_FILTERED"
    RATE_LIMITED = "RATE_LIMITED"
    BAD_REQUEST = "BAD_REQUEST"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    TOOL_CALL_FAILED = "TOOL_CALL_FAILED"
    UNKNOWN = "UNKNOWN"


ELLIPSIS = " … "


def truncate(value: str | None, limit: int) -> str | None:
    return None if value is None else str(value)[:limit]


def truncate_message(value: str | None, limit: int) -> str | None:
    """Keep the start and the end of a long message.

    Errors put their detail at either end. mlx-whisper prints the ffmpeg banner
    first and the failure last, and Hugging Face Hub starts with a request id.
    """
    if value is None:
        return None
    value = str(value)
    if len(value) <= limit:
        return value
    keep = limit - len(ELLIPSIS)
    head = keep - keep // 2
    return value[:head] + ELLIPSIS + value[len(value) - keep // 2 :]


@dataclass
class ErrorEvent:
    model_id: str
    error_code: str | ErrorCode
    error_message: str | None = None
    stack_trace_hash: str | None = None
    related_event_id: str | None = None
    http_status: int | None = None
    provider_error_code: str | None = None
    duration_ms: int | None = None
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
                # Capped here, so every error event gets the same limit,
                # from an integration or a manual track_error() call.
                "error_message": truncate_message(
                    self.error_message, constants.ERROR_MSG_MAX_LEN
                ),
                "stack_trace_hash": self.stack_trace_hash,
                "related_event_id": self.related_event_id,
                "http_status": self.http_status,
                "provider_error_code": truncate(
                    self.provider_error_code, constants.PROVIDER_ERROR_CODE_MAX_LEN
                ),
                "duration_ms": self.duration_ms,
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
