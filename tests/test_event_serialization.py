from __future__ import annotations

from wildedge.events import common
from wildedge.events.error import ErrorCode, ErrorEvent
from wildedge.events.feedback import FeedbackEvent, FeedbackType
from wildedge.events.inference import InferenceEvent, TextInputMeta
from wildedge.events.model_download import AdapterDownload, ModelDownloadEvent
from wildedge.events.model_load import AdapterLoad, ModelLoadEvent
from wildedge.events.span import SpanEvent


def test_inference_event_to_dict_omits_none_fields():
    event = InferenceEvent(
        model_id="m1",
        duration_ms=12,
        input_modality="text",
        output_modality="text",
        input_meta=TextInputMeta(char_count=4),
    )
    data = event.to_dict()
    inference = data["inference"]
    assert "batch_size" not in inference
    assert "output_meta" not in inference
    assert inference["input_meta"] == {"char_count": 4}


def test_model_load_event_to_dict_optional_fields():
    event = ModelLoadEvent(
        model_id="m1",
        duration_ms=20,
        memory_bytes=1024,
        adapter=AdapterLoad(adapter_id="a1", adapter_type="lora", rank=16),
    )
    data = event.to_dict()
    load = data["load"]
    assert load["memory_bytes"] == 1024
    assert load["adapter"]["adapter_id"] == "a1"
    assert "threads" not in load


def test_model_download_event_to_dict_cache_hit_and_bandwidth():
    event = ModelDownloadEvent(
        model_id="m1",
        source_url="hf://org/model",
        source_type="huggingface",
        file_size_bytes=100,
        downloaded_bytes=0,
        duration_ms=5,
        network_type="wifi",
        resumed=False,
        cache_hit=True,
        success=True,
        bandwidth_bps=1234,
        adapter=AdapterDownload(
            adapter_id="a1", adapter_type="lora", for_base_model="m1"
        ),
    )
    data = event.to_dict()
    download = data["download"]
    assert download["cache_hit"] is True
    assert download["bandwidth_bps"] == 1234
    assert download["adapter"]["for_base_model"] == "m1"


def test_feedback_event_enum_and_string_forms():
    enum_event = FeedbackEvent(
        model_id="m1",
        related_inference_id="i1",
        feedback_type=FeedbackType.ACCEPT,
    )
    string_event = FeedbackEvent(
        model_id="m1",
        related_inference_id="i1",
        feedback_type="reject",
    )
    assert enum_event.to_dict()["feedback"]["feedback_type"] == "accept"
    assert string_event.to_dict()["feedback"]["feedback_type"] == "reject"


def test_span_event_to_dict_includes_required_fields():
    event = SpanEvent(
        kind="tool",
        name="search",
        duration_ms=250,
        status="ok",
        attributes={"provider": "custom"},
    )
    data = event.to_dict()
    assert data["event_type"] == "span"
    assert data["span"]["kind"] == "tool"
    assert data["span"]["attributes"]["provider"] == "custom"


def test_span_event_context_serializes_under_context_key():
    event = SpanEvent(
        kind="agent_step",
        name="plan",
        duration_ms=10,
        status="ok",
        context={"user_id": "u1"},
    )
    data = event.to_dict()
    assert data["context"] == {"user_id": "u1"}
    assert "attributes" not in data


def test_span_event_attributes_and_context_are_independent():
    event = SpanEvent(
        kind="tool",
        name="search",
        duration_ms=50,
        status="ok",
        attributes={"provider": "custom"},
        context={"user_id": "u1"},
    )
    data = event.to_dict()
    assert data["span"]["attributes"] == {"provider": "custom"}
    assert data["context"] == {"user_id": "u1"}


def test_error_event_to_dict_http_status():
    event = ErrorEvent(
        model_id="m1",
        error_code=ErrorCode.UNKNOWN,
        http_status=429,
        provider_error_code="insufficient_quota",
        duration_ms=30_000,
    )
    assert event.to_dict()["error"] == {
        "error_code": "UNKNOWN",
        "http_status": 429,
        "provider_error_code": "insufficient_quota",
        "duration_ms": 30_000,
    }
    assert "http_status" not in ErrorEvent("m1", "OOM").to_dict()["error"]


def test_span_event_sends_non_scalar_attributes_as_json_strings():
    event = SpanEvent(
        kind="custom",
        name="s",
        duration_ms=1,
        status="ok",
        attributes={
            "rejections": ["English", "too long"],
            "scores": {"a": 1},
            "count": 3,
            "ratio": 0.5,
            "ok": True,
            "note": None,
            "label": "x",
        },
    )
    attrs = event.to_dict()["span"]["attributes"]
    assert attrs == {
        "rejections": '["English", "too long"]',
        "scores": '{"a": 1}',
        "count": 3,
        "ratio": 0.5,
        "ok": True,
        "note": None,
        "label": "x",
    }
    # The caller's dict is left as it was.
    assert event.attributes["rejections"] == ["English", "too long"]


def test_span_attribute_flattening_handles_unserializable_values():
    class Thing:
        def __str__(self):
            return "thing"

    event = SpanEvent(
        kind="custom",
        name="s",
        duration_ms=1,
        status="ok",
        attributes={"obj": Thing(), "tags": ("a", Thing())},
    )
    attrs = event.to_dict()["span"]["attributes"]
    assert attrs == {"obj": '"thing"', "tags": '["a", "thing"]'}


def test_span_attribute_warning_is_logged_once_per_key(caplog):
    common._warned_attribute_keys.discard("warn_once_tags")
    with caplog.at_level("WARNING", logger="wildedge"):
        for _ in range(3):
            SpanEvent(
                kind="custom",
                name="s",
                duration_ms=1,
                status="ok",
                attributes={"warn_once_tags": ["a"]},
            ).to_dict()
    warnings = [r for r in caplog.records if "warn_once_tags" in r.getMessage()]
    assert len(warnings) == 1


def test_span_attribute_falls_back_to_str_when_json_fails():
    circular: list = []
    circular.append(circular)
    event = SpanEvent(
        kind="custom",
        name="s",
        duration_ms=1,
        status="ok",
        attributes={"loop": circular, 7: "int key"},
    )
    attrs = event.to_dict()["span"]["attributes"]
    assert attrs == {"loop": "[[...]]", "7": "int key"}
