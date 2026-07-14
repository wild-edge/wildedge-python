"""Process-wide default client and the module-level tracking API.

One client per process is the common case: `wildedge run` installs one before
user code loads, `wildedge.init()` creates or reuses one, and the module-level
functions below (`wildedge.trace`, `wildedge.span`, ...) delegate to whichever
client is current. Application code never has to thread a client instance
through call sites.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from wildedge import constants
from wildedge.client import SpanContextManager, WildEdge
from wildedge.events.span import SpanKind, SpanStatus
from wildedge.hubs.registry import supported_hubs
from wildedge.integrations.registry import supported_integrations
from wildedge.logging import logger
from wildedge.model import ModelHandle
from wildedge.settings import parse_hub_list, parse_integration_list
from wildedge.tracing import trace_context

_lock = threading.Lock()
_default_client: WildEdge | None = None


def set_default_client(client: WildEdge | None) -> None:
    """Register ``client`` as the process default; ``None`` clears it."""
    global _default_client
    with _lock:
        _default_client = client


def peek_default_client() -> WildEdge | None:
    """Return the process default client without creating one."""
    return _default_client


def get_client() -> WildEdge:
    """
    Return the process default client, creating one from the environment on
    first use.

    The lazily created client reads ``WILDEDGE_DSN`` exactly like a directly
    constructed ``WildEdge`` (without a DSN it is a no-op) and enables only
    the integrations and hubs named in ``WILDEDGE_INTEGRATIONS`` and
    ``WILDEDGE_HUBS``. When those variables are unset nothing is patched:
    auto-instrumentation stays opt-in.
    """
    global _default_client
    with _lock:
        if _default_client is None:
            _default_client = _client_from_env()
        return _default_client


def _client_from_env() -> WildEdge:
    client = WildEdge()
    raw_integrations = os.environ.get(constants.ENV_INTEGRATIONS)
    raw_hubs = os.environ.get(constants.ENV_HUBS)
    integrations = (
        parse_integration_list(raw_integrations, sorted(supported_integrations()))
        if raw_integrations
        else []
    )
    hubs = parse_hub_list(raw_hubs, sorted(supported_hubs())) if raw_hubs else []

    targets: list[tuple[str | None, list[str] | None]] = []
    if integrations:
        targets = [(integration, hubs or None) for integration in integrations]
    elif hubs:
        targets = [(None, hubs)]
    # A bad environment value must not break the caller that triggered lazy
    # creation, so failures downgrade to a warning here; explicit init() with
    # the same names would raise.
    for integration, target_hubs in targets:
        try:
            client.instrument(integration, hubs=target_hubs)
        except Exception as exc:
            logger.warning(
                "wildedge: instrument(%r) from environment failed: %s",
                integration,
                exc,
            )
    return client


# A trace sets correlation contextvars and emits nothing, so it needs no
# client; the module-level name is the context manager itself.
trace = trace_context


def span(
    *,
    kind: SpanKind,
    name: str,
    status: SpanStatus = "ok",
    model_id: str | None = None,
    input_summary: str | None = None,
    output_summary: str | None = None,
    attributes: dict[str, Any] | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    run_id: str | None = None,
    agent_id: str | None = None,
    step_index: int | None = None,
    conversation_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> SpanContextManager:
    """``WildEdge.span`` on the default client."""
    return get_client().span(
        kind=kind,
        name=name,
        status=status,
        model_id=model_id,
        input_summary=input_summary,
        output_summary=output_summary,
        attributes=attributes,
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        run_id=run_id,
        agent_id=agent_id,
        step_index=step_index,
        conversation_id=conversation_id,
        context=context,
    )


def track_span(
    *,
    kind: SpanKind,
    name: str,
    duration_ms: int,
    status: SpanStatus = "ok",
    model_id: str | None = None,
    input_summary: str | None = None,
    output_summary: str | None = None,
    attributes: dict[str, Any] | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    run_id: str | None = None,
    agent_id: str | None = None,
    step_index: int | None = None,
    conversation_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    """``WildEdge.track_span`` on the default client."""
    return get_client().track_span(
        kind=kind,
        name=name,
        duration_ms=duration_ms,
        status=status,
        model_id=model_id,
        input_summary=input_summary,
        output_summary=output_summary,
        attributes=attributes,
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        run_id=run_id,
        agent_id=agent_id,
        step_index=step_index,
        conversation_id=conversation_id,
        context=context,
    )


def register_model(
    model_obj: object,
    *,
    model_id: str | None = None,
    source: str | None = None,
    family: str | None = None,
    version: str | None = None,
    quantization: str | None = None,
    auto_instrument: bool = True,
) -> ModelHandle:
    """``WildEdge.register_model`` on the default client."""
    return get_client().register_model(
        model_obj,
        model_id=model_id,
        source=source,
        family=family,
        version=version,
        quantization=quantization,
        auto_instrument=auto_instrument,
    )


def flush(timeout: float = 5.0) -> None:
    """Flush the default client's queued events."""
    get_client().flush(timeout=timeout)
