from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from wildedge.events.common import add_optional_fields

SpanKind = Literal[
    "agent_step",
    "tool",
    "retrieval",
    "memory",
    "router",
    "guardrail",
    "cache",
    "eval",
    "custom",
]
SpanStatus = Literal["ok", "error"]


@dataclass
class SpanEvent:
    kind: SpanKind
    name: str
    duration_ms: int
    status: SpanStatus
    model_id: str | None = None
    input_summary: str | None = None
    output_summary: str | None = None
    attributes: dict[str, Any] | None = None
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
        span_data: dict[str, Any] = {
            "kind": self.kind,
            "name": self.name,
            "duration_ms": self.duration_ms,
            "status": self.status,
        }
        if self.input_summary is not None:
            span_data["input_summary"] = self.input_summary
        if self.output_summary is not None:
            span_data["output_summary"] = self.output_summary
        if self.attributes is not None:
            span_data["attributes"] = self.attributes

        event = {
            "event_id": self.event_id,
            "event_type": "span",
            "timestamp": self.timestamp.isoformat(),
            "span": span_data,
        }
        add_optional_fields(
            event,
            {
                "model_id": self.model_id,
                "trace_id": self.trace_id,
                "span_id": self.span_id,
                "parent_span_id": self.parent_span_id,
                "run_id": self.run_id,
                "agent_id": self.agent_id,
                "step_index": self.step_index,
                "conversation_id": self.conversation_id,
                "context": self.context,
            },
        )
        return event
