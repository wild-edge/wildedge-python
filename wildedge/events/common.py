from __future__ import annotations

import json
import threading
from typing import Any

from wildedge.logging import logger

_SCALAR_TYPES = (str, int, float, bool, type(None))
_warned_attribute_keys: set[str] = set()
_warned_lock = threading.Lock()


def add_optional_fields(event: dict, fields: dict[str, Any]) -> dict:
    """Add non-None fields to an event payload."""
    for key, value in fields.items():
        if value is not None:
            event[key] = value
    return event


def flatten_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Return attributes with every value a scalar the ingest API accepts.

    The server rejects the whole batch when one attribute value is a list,
    dict or other non-scalar. Such values are sent as JSON strings instead
    (``["a", "b"]`` becomes ``'["a", "b"]'``), with one warning per key.
    """
    flat: dict[str, Any] = {}
    for key, value in attributes.items():
        key = str(key)
        if isinstance(value, _SCALAR_TYPES):
            flat[key] = value
            continue
        try:
            flat[key] = json.dumps(value, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            flat[key] = str(value)
        with _warned_lock:
            first = key not in _warned_attribute_keys
            _warned_attribute_keys.add(key)
        if first:
            logger.warning(
                "wildedge: span attribute %r is a %s, not a str, int, float or "
                "bool; sending it as a JSON string",
                key,
                type(value).__name__,
            )
    return flat
