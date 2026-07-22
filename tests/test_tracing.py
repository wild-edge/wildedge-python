from __future__ import annotations

from wildedge.client import SpanContextManager
from wildedge.model import ModelHandle, ModelInfo
from wildedge.tracing import get_span_context, span_context, trace_context


def test_track_inference_uses_trace_context():
    events: list[dict] = []

    def publish(event: dict) -> None:
        events.append(event)

    handle = ModelHandle(
        "model-1",
        ModelInfo(
            model_name="test",
            model_version="1.0",
            model_source="local",
            model_format="onnx",
        ),
        publish,
    )

    with trace_context(
        trace_id="trace-123",
        run_id="run-1",
        agent_id="agent-1",
        attributes={"trace_key": "trace_val"},
    ):
        with span_context(span_id="span-abc", step_index=2, attributes={"span_key": 2}):
            handle.track_inference(duration_ms=5)

    assert events[0]["trace_id"] == "trace-123"
    assert events[0]["parent_span_id"] == "span-abc"
    assert events[0]["run_id"] == "run-1"
    assert events[0]["agent_id"] == "agent-1"
    assert events[0]["step_index"] == 2
    assert events[0]["attributes"] == {"trace_key": "trace_val", "span_key": 2}


class _FakeClient:
    def __init__(self, events: list[dict]) -> None:
        self._events = events

    def track_span(self, **kwargs) -> str:
        self._events.append(kwargs)
        return kwargs.get("span_id", "")


def test_span_root_has_no_parent():
    """A root span must not reference itself as its own parent."""
    events: list[dict] = []
    client = _FakeClient(events)

    with SpanContextManager(client, kind="agent_step", name="root"):
        pass

    assert len(events) == 1
    assert events[0]["parent_span_id"] is None


def test_span_context_restored_after_exit():
    """The active span context must revert to the parent after a span exits."""
    events: list[dict] = []
    client = _FakeClient(events)

    with span_context(span_id="parent-span"):
        with SpanContextManager(client, kind="agent_step", name="child"):
            inner_id = get_span_context().span_id

        assert get_span_context().span_id == "parent-span"

    assert inner_id != "parent-span"
    assert events[0]["parent_span_id"] == "parent-span"
    assert events[0]["span_id"] != "parent-span"


def test_nested_spans_correct_parent_chain():
    """Nested spans must each point to their direct parent, not themselves."""
    events: list[dict] = []
    client = _FakeClient(events)

    with SpanContextManager(client, kind="agent_step", name="outer") as outer:
        with SpanContextManager(client, kind="tool", name="inner") as inner:
            pass

    assert len(events) == 2
    inner_ev, outer_ev = events[0], events[1]
    assert inner_ev["span_id"] == inner.span_id
    assert inner_ev["parent_span_id"] == outer.span_id
    assert outer_ev["span_id"] == outer.span_id
    assert outer_ev["parent_span_id"] is None


def test_span_set_attributes_merges():
    events: list[dict] = []
    client = _FakeClient(events)

    with SpanContextManager(
        client, kind="eval", name="lint", attributes={"a": 1}
    ) as span:
        span.set_attributes(b=2)
        span.set_attributes(a=3)

    assert events[0]["attributes"] == {"a": 3, "b": 2}


def test_span_set_attributes_from_none():
    events: list[dict] = []
    client = _FakeClient(events)

    with SpanContextManager(client, kind="eval", name="lint") as span:
        span.set_attributes(lint_errors=4)

    assert events[0]["attributes"] == {"lint_errors": 4}


def test_span_fail_sets_status_and_summary():
    events: list[dict] = []
    client = _FakeClient(events)

    with SpanContextManager(client, kind="eval", name="lint") as span:
        span.fail("4 lint errors")

    assert events[0]["status"] == "error"
    assert events[0]["output_summary"] == "4 lint errors"


def test_span_fail_without_detail_keeps_summary():
    events: list[dict] = []
    client = _FakeClient(events)

    with SpanContextManager(
        client, kind="eval", name="lint", output_summary="kept"
    ) as span:
        span.fail()

    assert events[0]["status"] == "error"
    assert events[0]["output_summary"] == "kept"
