from __future__ import annotations

from wildedge.model import ModelHandle, ModelInfo
from wildedge.tracing import span_context, trace_context


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
