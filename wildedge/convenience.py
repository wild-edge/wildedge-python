from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from wildedge.client import WildEdge
from wildedge.logging import logger


def _normalize_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [item for item in value if item]


def init(
    *,
    integrations: str | Iterable[str] | None = None,
    hubs: str | Iterable[str] | None = None,
    **kwargs: Any,
) -> WildEdge:
    """
    Convenience initializer: construct a WildEdge client and instrument integrations.

    Additional keyword arguments are forwarded to WildEdge(...).
    """
    client = WildEdge(**kwargs)
    normalized_integrations = _normalize_list(integrations)
    normalized_hubs = _normalize_list(hubs)

    if normalized_integrations:
        for integration in normalized_integrations:
            client.instrument(integration, hubs=normalized_hubs or None)
    elif normalized_hubs:
        client.instrument(None, hubs=normalized_hubs)
    elif getattr(client, "debug", False):
        logger.debug("wildedge: init called without integrations or hubs")

    return client
