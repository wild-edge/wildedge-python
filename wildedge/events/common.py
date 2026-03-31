from __future__ import annotations

from typing import Any


def add_optional_fields(event: dict, fields: dict[str, Any]) -> dict:
    """Add non-None fields to an event payload."""
    for key, value in fields.items():
        if value is not None:
            event[key] = value
    return event
