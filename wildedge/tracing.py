from __future__ import annotations

import contextlib
import contextvars
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    run_id: str | None = None
    agent_id: str | None = None
    conversation_id: str | None = None
    attributes: dict[str, Any] | None = None


@dataclass(frozen=True)
class SpanContext:
    span_id: str
    parent_span_id: str | None = None
    step_index: int | None = None
    attributes: dict[str, Any] | None = None


_TRACE_CTX: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "wildedge_trace_ctx", default=None
)
_SPAN_CTX: contextvars.ContextVar[SpanContext | None] = contextvars.ContextVar(
    "wildedge_span_ctx", default=None
)


def get_trace_context() -> TraceContext | None:
    return _TRACE_CTX.get()


def get_span_context() -> SpanContext | None:
    return _SPAN_CTX.get()


def set_trace_context(ctx: TraceContext) -> contextvars.Token:
    return _TRACE_CTX.set(ctx)


def reset_trace_context(token: contextvars.Token) -> None:
    _TRACE_CTX.reset(token)


def set_span_context(ctx: SpanContext) -> contextvars.Token:
    return _SPAN_CTX.set(ctx)


def reset_span_context(token: contextvars.Token) -> None:
    _SPAN_CTX.reset(token)


@contextlib.contextmanager
def trace_context(
    *,
    trace_id: str | None = None,
    run_id: str | None = None,
    agent_id: str | None = None,
    conversation_id: str | None = None,
    attributes: dict[str, Any] | None = None,
):
    if trace_id is None:
        trace_id = str(uuid.uuid4())
    token = set_trace_context(
        TraceContext(
            trace_id=trace_id,
            run_id=run_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            attributes=attributes,
        )
    )
    try:
        yield get_trace_context()
    finally:
        reset_trace_context(token)


@contextlib.contextmanager
def span_context(
    *,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    step_index: int | None = None,
    attributes: dict[str, Any] | None = None,
):
    if span_id is None:
        span_id = str(uuid.uuid4())
    if parent_span_id is None:
        current = get_span_context()
        parent_span_id = current.span_id if current else None
    token = set_span_context(
        SpanContext(
            span_id=span_id,
            parent_span_id=parent_span_id,
            step_index=step_index,
            attributes=attributes,
        )
    )
    try:
        yield get_span_context()
    finally:
        reset_span_context(token)


def _merge_attributes(*candidates: dict[str, Any] | None) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for attrs in candidates:
        if not attrs:
            continue
        merged.update(attrs)
    return merged or None


def merge_correlation_fields(
    *,
    trace_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    run_id: str | None = None,
    agent_id: str | None = None,
    step_index: int | None = None,
    conversation_id: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trace = get_trace_context()
    span = get_span_context()

    resolved_trace_id = trace_id or (trace.trace_id if trace else None)
    resolved_span_id = span_id
    resolved_parent_span_id = parent_span_id or (span.span_id if span else None)
    resolved_run_id = run_id or (trace.run_id if trace else None)
    resolved_agent_id = agent_id or (trace.agent_id if trace else None)
    resolved_step_index = (
        step_index if step_index is not None else (span.step_index if span else None)
    )
    resolved_conversation_id = conversation_id or (
        trace.conversation_id if trace else None
    )
    resolved_attributes = _merge_attributes(
        trace.attributes if trace else None,
        span.attributes if span else None,
        attributes,
    )

    return {
        "trace_id": resolved_trace_id,
        "span_id": resolved_span_id,
        "parent_span_id": resolved_parent_span_id,
        "run_id": resolved_run_id,
        "agent_id": resolved_agent_id,
        "step_index": resolved_step_index,
        "conversation_id": resolved_conversation_id,
        "attributes": resolved_attributes,
    }
