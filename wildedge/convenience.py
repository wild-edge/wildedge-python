from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from wildedge.client import WildEdge
from wildedge.defaults import peek_default_client, set_default_client
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
    Initialize the process default client and instrument integrations.

    When a default client already exists (typically installed by ``wildedge
    run`` before user code loaded) and no ``dsn`` is passed, that client is
    reused: the requested integrations are applied to it (instrumenting is
    idempotent per process) and it is returned. Passing ``dsn`` always
    constructs a fresh client and makes it the new default.

    Additional keyword arguments are forwarded to WildEdge(...) when a new
    client is constructed.
    """
    client = None if "dsn" in kwargs else peek_default_client()
    if client is not None and kwargs:
        logger.warning(
            "wildedge: init() reusing the existing default client; ignoring %s",
            ", ".join(sorted(kwargs)),
        )
    if client is None:
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

    set_default_client(client)
    return client
