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

        return {
            "event_id": self.event_id,
            "event_type": "feedback",
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "feedback": feedback_data,
        }
