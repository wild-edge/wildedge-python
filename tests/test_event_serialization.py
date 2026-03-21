from __future__ import annotations

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
        span_attributes={"provider": "custom"},
    )
    data = event.to_dict()
    assert data["event_type"] == "span"
    assert data["span"]["kind"] == "tool"
    assert data["span"]["attributes"]["provider"] == "custom"
