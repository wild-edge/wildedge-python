"""Ownership of process-wide instrumentation, independent of the default client."""

from __future__ import annotations

import os
import threading
import weakref
from typing import TYPE_CHECKING

from wildedge.logging import logger

if TYPE_CHECKING:
    from wildedge.client import WildEdge

_lock = threading.Lock()
_owner: weakref.ReferenceType[WildEdge] | None = None
_enabled: set[str] = set()
_strict_ownership = False


class InstrumentationOwnedError(RuntimeError):
    """Another live client already owns process-wide instrumentation."""


def set_strict_ownership(strict: bool) -> None:
    """Raise on ownership conflicts instead of warning (`--strict-integrations`)."""
    global _strict_ownership
    _strict_ownership = strict


def strict_ownership() -> bool:
    return _strict_ownership


class ClientRoute:
    """Stable callable captured by a factory patch, resolving its current owner."""

    def __init__(self, integration: str) -> None:
        self.integration = integration

    def __call__(self) -> WildEdge | None:
        with _lock:
            client = _owner() if _owner is not None else None
            if client is None or client.closed or self.integration not in _enabled:
                return None
            return client


class BoundClientReference:
    """Existing provider clients stay with their original telemetry destination."""

    def __init__(self, client: WildEdge) -> None:
        self._ref = weakref.ref(client)
        self._warned = False

    def __call__(self) -> WildEdge | None:
        client = self._ref()
        if client is None or client.closed:
            if not self._warned:
                logger.warning(
                    "wildedge: provider client belongs to an unavailable SDK client; "
                    "recreate it after initializing the replacement SDK client"
                )
                self._warned = True
            return None
        return client


def bind_client_reference(reference: object, client: WildEdge) -> object:
    """Snapshot a shared factory route when creating an individual provider client."""
    return (
        BoundClientReference(client)
        if isinstance(reference, ClientRoute)
        else reference
    )


def claim_instrumentation(client: WildEdge, integration: str) -> ClientRoute:
    global _owner
    with _lock:
        if client.closed:
            raise RuntimeError("Cannot instrument with a closed WildEdge client")
        owner = _owner() if _owner is not None else None
        if owner is not None and not owner.closed and owner is not client:
            raise InstrumentationOwnedError(
                "Auto-instrumentation is owned by another live WildEdge client; "
                "close that client before enabling instrumentation on its replacement"
            )
        if owner is not client:
            _enabled.clear()
            _owner = weakref.ref(client)
        _enabled.add(integration)
        return ClientRoute(integration)


def has_other_live_owner(client: WildEdge) -> bool:
    with _lock:
        owner = _owner() if _owner is not None else None
        return owner is not None and not owner.closed and owner is not client


def _reset_lock_after_fork() -> None:
    global _lock
    _lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_lock_after_fork)
