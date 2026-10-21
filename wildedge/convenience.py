from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from wildedge.client import WildEdge, parse_dsn
from wildedge.defaults import peek_default_client, set_default_client
from wildedge.logging import logger
from wildedge.settings import read_client_env


def _normalize_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [item for item in value if item]


def _same_destination(client: WildEdge, dsn: str | None) -> bool:
    """True when ``dsn`` (or WILDEDGE_DSN) points where ``client`` sends."""
    destination = getattr(client, "destination", None)
    if destination is None:
        return False
    resolved = read_client_env(dsn=dsn).dsn
    if not resolved:
        return False
    try:
        return parse_dsn(resolved) == destination
    except ValueError:
        return False


def init(
    *,
    integrations: str | Iterable[str] | None = None,
    hubs: str | Iterable[str] | None = None,
    **kwargs: Any,
) -> WildEdge:
    """
    Initialize the process default client and instrument integrations.

    When a live default client already exists (typically installed by ``wildedge
    run`` before user code loaded) and no ``dsn`` is passed, or ``dsn`` points
    to the same project, that client is reused: the requested integrations are
    applied to it (instrumenting is idempotent per process) and it is returned.
    A ``dsn`` for another project constructs a fresh client and makes it the new
    default. A closed default is replaced. While another live client owns
    process-wide instrumentation, the new client logs a warning and leaves
    automatic tracking with that owner. With ``wildedge run
    --strict-integrations`` it raises ``RuntimeError`` instead.

    Additional keyword arguments are forwarded to WildEdge(...) when a new
    client is constructed.
    """
    client = peek_default_client()
    if client is not None and client.closed:
        client = None
    if (
        client is not None
        and "dsn" in kwargs
        and not _same_destination(client, kwargs["dsn"])
    ):
        client = None
    ignored = sorted(key for key in kwargs if key != "dsn")
    if client is not None and ignored:
        logger.warning(
            "wildedge: init() reusing the existing default client; ignoring %s",
            ", ".join(ignored),
        )
    created = client is None
    if created:
        client = WildEdge(**kwargs)

    normalized_integrations = _normalize_list(integrations)
    normalized_hubs = _normalize_list(hubs)

    try:
        if normalized_integrations:
            for integration in normalized_integrations:
                client.instrument(integration, hubs=normalized_hubs or None)
        elif normalized_hubs:
            client.instrument(None, hubs=normalized_hubs)
        elif getattr(client, "debug", False):
            logger.debug("wildedge: init called without integrations or hubs")
    except Exception:
        if created:
            client.close()
        raise

    set_default_client(client)
    return client
