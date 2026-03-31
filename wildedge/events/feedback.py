from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class FeedbackType(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    UNDO = "undo"
    EDIT = "edit"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    REPORT = "report"


@dataclass
class FeedbackEvent:
    model_id: str
    related_inference_id: str
    feedback_type: str | FeedbackType
    delay_ms: int | None = None
    edit_distance: int | None = None
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
        feedback_type = (
            self.feedback_type.value
            if isinstance(self.feedback_type, FeedbackType)
            else self.feedback_type
        )
        feedback_data: dict[str, Any] = {
            "related_inference_id": self.related_inference_id,
            "feedback_type": feedback_type,
        }
        if self.delay_ms is not None:
            feedback_data["delay_ms"] = self.delay_ms
        if self.edit_distance is not None:
            feedback_data["edit_distance"] = self.edit_distance

        event = {
            "event_id": self.event_id,
            "event_type": "feedback",
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "feedback": feedback_data,
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
                "attributes": self.context,
            },
        )
        return event
