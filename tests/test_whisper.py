"""Whisper adapters preserve results, lazy lifecycles, and capture privacy."""

import gc
import io
import sys
import wave
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from wildedge.attachments import AttachmentManager, AttachmentStore
from wildedge.client import WildEdge
from wildedge.instrumentation import claim_instrumentation
from wildedge.integrations import faster_whisper as faster
from wildedge.integrations.faster_whisper import (
    install_auto_load_patch,
    wrap_transcribe,
)
from wildedge.integrations.mlx_whisper import install_auto_load_patch as install_mlx
from wildedge.integrations.whisper import (
    CLIENT_ATTR,
    DESCRIPTOR_ATTR,
    HANDLE_ATTR,
    WhisperDescriptor,
    WhisperExtractor,
    model_name_and_source,
    track_transcription,
)
from wildedge.platforms.device_info import DeviceInfo


@pytest.fixture
def handle():
    return SimpleNamespace(
        capture_attachments=None, track_inference=MagicMock(), track_error=MagicMock()
    )


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "input.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 8000)
    return path


def tracked_model(handle):
    """A faster-whisper model object as the constructor patch leaves it."""
    model = SimpleNamespace()
    setattr(model, HANDLE_ATTR, handle)
    setattr(model, CLIENT_ATTR, lambda: object())
    return model


def stubbed_client():
    """A second client, set up like the client_with_stubbed_runtime fixture."""
    with (
        patch("wildedge.client.detect_device", return_value=DeviceInfo("id", "linux")),
        patch("wildedge.client.Transmitter"),
        patch("wildedge.client.Consumer"),
    ):
        return WildEdge(
            dsn="https://secret@ingest.wildedge.dev/key", sampling_interval_s=None
        )


def events_of(events, event_type):
    return [e for e in events if e["event_type"] == event_type]


# Shared metadata ------------------------------------------------------------


def test_transcription_metadata(wav, handle):
    track_transcription(
        handle,
        wav,
        "hello world",
        duration_ms=123,
        language="en",
        language_probability=0.9,
    )
    event = handle.track_inference.call_args.kwargs
    assert event["duration_ms"] == 123
    assert event["input_meta"].to_dict() == {
        "duration_ms": 500,
        "sample_rate": 16000,
        "channels": 1,
        "bit_depth": 16,
        "format": "wav",
    }
    assert event["context"] == {
        "transcript_word_count": 2,
        "language": "en",
        "language_confidence": 0.9,
    }


@pytest.mark.parametrize("uploads_enabled", [False, True])
def test_transcription_never_attaches_payloads(wav, handle, uploads_enabled):
    if uploads_enabled:
        handle.capture_attachments = MagicMock()
    track_transcription(handle, wav, "hello world", duration_ms=1)
    assert "attachments" not in handle.track_inference.call_args.kwargs


def test_file_like_audio_is_not_read(handle):
    source = io.BytesIO(b"RIFF....")
    handle.capture_attachments = MagicMock()
    track_transcription(handle, source, "hello", duration_ms=1)
    event = handle.track_inference.call_args.kwargs
    assert event["input_meta"].to_dict() == {}
    assert "attachments" not in event
    assert source.tell() == 0


def test_word_count_is_left_out_for_unspaced_languages(handle):
    track_transcription(handle, None, "你好世界", duration_ms=1, language="zh")
    context = handle.track_inference.call_args.kwargs["context"]
    assert "transcript_word_count" not in context


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "/cache/hub/models--Systran--faster-whisper-tiny/snapshots/d90ca5fe",
            ("Systran/faster-whisper-tiny", "huggingface"),
        ),
        ("mlx-community/whisper-tiny", ("mlx-community/whisper-tiny", "huggingface")),
        ("small", ("small", "huggingface")),
    ],
)
def test_model_name_and_source(path, expected):
    assert model_name_and_source(path) == expected


def test_local_model_is_named_after_its_folder(tmp_path, monkeypatch):
    (tmp_path / "whisper-small").mkdir()
    monkeypatch.chdir(tmp_path / "whisper-small")
    assert model_name_and_source(".") == ("whisper-small", "local")


def test_extractor():
    model = SimpleNamespace()
    setattr(
        model,
        DESCRIPTOR_ATTR,
        WhisperDescriptor("whisper-small", "local", "faster_whisper", "cpu", "int8"),
    )
    ident, info = WhisperExtractor().extract_info(model, {})
    assert ident == "faster_whisper:whisper-small"
    assert info.model_source == "local"
    assert info.model_format == "ctranslate2"
    assert info.model_family == "whisper"
    assert info.quantization == "int8"


# faster-whisper segment stream ---------------------------------------------


@pytest.mark.parametrize("outcome", ["completed", "closed", "error"])
def test_lazy_segments(handle, outcome):
    segment = SimpleNamespace(text="hello world")
    info = SimpleNamespace(language="en", language_probability=0.8, duration=2)

    def segments():
        yield segment
        if outcome == "error":
            raise ValueError("decoder failure")

    call = wrap_transcribe(lambda *_: (segments(), info))
    stream, returned_info = call(tracked_model(handle), "audio.wav")
    assert returned_info is info
    handle.track_inference.assert_not_called()
    assert next(stream) is segment
    if outcome == "closed":
        stream.close()
        stream.close()
    elif outcome == "error":
        with pytest.raises(ValueError, match="decoder failure"):
            next(stream)
        handle.track_error.assert_called_once()
        handle.track_inference.assert_not_called()
        return
    else:
        assert list(stream) == []
    handle.track_inference.assert_called_once()
    event = handle.track_inference.call_args.kwargs
    assert event["success"] == (outcome == "completed")
    assert event["context"]["transcript_word_count"] == 2
    assert "time_to_first_segment_ms" in event["context"]
    assert event["input_meta"].duration_ms == 2000


def test_loop_that_stops_early_is_reported_as_abandoned(handle):
    segments = [SimpleNamespace(text="one"), SimpleNamespace(text="two")]
    stream, _ = wrap_transcribe(lambda *_: (iter(segments), SimpleNamespace()))(
        tracked_model(handle), "audio.wav"
    )
    for _segment in stream:
        break
    handle.track_inference.assert_not_called()
    del stream
    gc.collect()
    event = handle.track_inference.call_args.kwargs
    assert event["error_code"] == "STREAM_ABANDONED"
    assert event["context"]["transcript_word_count"] == 1


def test_stream_that_was_never_iterated_sends_nothing(handle):
    stream, _ = wrap_transcribe(lambda *_: (iter([]), SimpleNamespace()))(
        tracked_model(handle), "audio.wav"
    )
    del stream
    gc.collect()
    handle.track_inference.assert_not_called()


def test_stream_counts_words_without_attaching_transcript(handle):
    handle.capture_attachments = MagicMock()
    segments = [SimpleNamespace(text="hello "), SimpleNamespace(text="world")]
    stream, _ = wrap_transcribe(lambda *_: (iter(segments), SimpleNamespace()))(
        tracked_model(handle), object()
    )
    assert list(stream) == segments
    event = handle.track_inference.call_args.kwargs
    assert "attachments" not in event
    assert event["context"]["transcript_word_count"] == 2


def test_transcribe_errors_preserve_exception(handle):
    failure = ValueError("invalid audio")

    def original(*_args):
        raise failure

    handle.track_error.side_effect = RuntimeError("telemetry failed")
    with pytest.raises(ValueError, match="invalid audio") as raised:
        wrap_transcribe(original)(tracked_model(handle), "missing.wav")
    assert raised.value is failure
    handle.track_inference.assert_not_called()


def test_lazy_telemetry_failure_does_not_break_iterator(handle):
    handle.track_inference.side_effect = RuntimeError("telemetry failed")
    segment = SimpleNamespace(text="hello")
    stream, _ = wrap_transcribe(lambda *_: (iter([segment]), SimpleNamespace()))(
        tracked_model(handle), "audio.wav"
    )
    assert list(stream) == [segment]
    assert list(stream) == []
    handle.track_inference.assert_called_once()


# faster-whisper model patch -------------------------------------------------


@pytest.fixture
def fake_faster_whisper(monkeypatch):
    module = ModuleType("faster_whisper")

    class WhisperModel:
        def __init__(self, model_size_or_path, device="cpu", compute_type="int8"):
            self.model = SimpleNamespace(device=device, compute_type=compute_type)

        def transcribe(self, audio):
            return iter([SimpleNamespace(text="hello")]), SimpleNamespace()

    class BatchedInferencePipeline:
        def __init__(self, model):
            self.model = model

        def transcribe(self, audio):
            return iter([SimpleNamespace(text="hello")]), SimpleNamespace()

    module.WhisperModel = WhisperModel
    module.BatchedInferencePipeline = BatchedInferencePipeline
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return module


@pytest.fixture
def instrumented_client(client_with_stubbed_runtime):
    client = client_with_stubbed_runtime
    client.events = []
    client.publish = client.events.append
    return client


def test_faster_constructor_describes_the_runtime(
    fake_faster_whisper, instrumented_client
):
    route = claim_instrumentation(instrumented_client, "faster_whisper")
    install_auto_load_patch(route)
    install_auto_load_patch(route)
    model = fake_faster_whisper.WhisperModel(
        "small", device="cuda", compute_type="float16"
    )
    info = getattr(model, DESCRIPTOR_ATTR)
    assert info.model_id == "faster_whisper:small"
    assert info.accelerator == "cuda"
    assert info.quantization == "f16"
    assert len(events_of(instrumented_client.events, "model_load")) == 1


def test_each_faster_model_instance_reports_its_load(
    fake_faster_whisper, instrumented_client
):
    install_auto_load_patch(
        claim_instrumentation(instrumented_client, "faster_whisper")
    )
    fake_faster_whisper.WhisperModel("small")
    fake_faster_whisper.WhisperModel("small")
    assert len(events_of(instrumented_client.events, "model_load")) == 2


def test_batched_pipeline_is_tracked(fake_faster_whisper, instrumented_client):
    install_auto_load_patch(
        claim_instrumentation(instrumented_client, "faster_whisper")
    )
    model = fake_faster_whisper.WhisperModel("small")
    pipeline = fake_faster_whisper.BatchedInferencePipeline(model)
    segments, _ = pipeline.transcribe("audio.wav")
    assert [s.text for s in segments] == ["hello"]
    inference = events_of(instrumented_client.events, "inference")
    assert [e["model_id"] for e in inference] == ["faster_whisper:small"]


def test_faster_model_stays_with_its_original_client(
    fake_faster_whisper, instrumented_client
):
    install_auto_load_patch(
        claim_instrumentation(instrumented_client, "faster_whisper")
    )
    model = fake_faster_whisper.WhisperModel("small")
    instrumented_client.close()
    segments, _ = model.transcribe("audio.wav")
    assert not isinstance(segments, faster.SegmentStream)
    assert [s.text for s in segments] == ["hello"]
    assert not events_of(instrumented_client.events, "inference")


# mlx-whisper ----------------------------------------------------------------


class FakeMlxModel:
    """Stands in for an MLX Whisper model; weak-referenceable like the real one."""

    def __init__(self, bits=None):
        self.layers = [("", SimpleNamespace(bits=bits))] if bits else []

    def named_modules(self):
        return self.layers


@pytest.fixture
def fake_mlx_whisper(monkeypatch):
    """mlx_whisper with ModelHolder's caching: one model, reloaded on change."""
    root, module, loaders = (
        ModuleType(n)
        for n in ("mlx_whisper", "mlx_whisper.transcribe", "mlx_whisper.load_models")
    )

    def load_model(path_or_hf_repo, dtype="float32"):
        return FakeMlxModel(bits=4)

    class ModelHolder:
        model = None
        model_path = None

    def transcribe(_audio, *, path_or_hf_repo="mlx-community/whisper-tiny", **_kw):
        if path_or_hf_repo == "broken":
            raise ValueError("cannot load")
        if ModelHolder.model is None or ModelHolder.model_path != path_or_hf_repo:
            ModelHolder.model = module.load_model(path_or_hf_repo)
            ModelHolder.model_path = path_or_hf_repo
        return {"text": "hello world", "language": "en"}

    module.ModelHolder = ModelHolder
    module.transcribe = transcribe
    module.load_model = loaders.load_model = load_model
    for mod in (root, module, loaders):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    root.holder = ModelHolder
    return root


def test_mlx_returns_the_original_result(fake_mlx_whisper, instrumented_client):
    install_mlx(claim_instrumentation(instrumented_client, "mlx_whisper"))
    result = fake_mlx_whisper.transcribe("audio.wav", path_or_hf_repo="org/model")
    assert result == {"text": "hello world", "language": "en"}
    assert len(events_of(instrumented_client.events, "inference")) == 1


def test_mlx_model_cached_before_init_is_reported_as_a_hit(
    fake_mlx_whisper, instrumented_client
):
    fake_mlx_whisper.holder.model = FakeMlxModel()
    fake_mlx_whisper.holder.model_path = "org/model"
    install_mlx(claim_instrumentation(instrumented_client, "mlx_whisper"))
    fake_mlx_whisper.transcribe("audio.wav", path_or_hf_repo="org/model")
    (load,) = events_of(instrumented_client.events, "model_load")
    assert load["model_id"] == "mlx_whisper:org/model"
    assert load["attributes"]["model_cache_hit"] is True
    registered = instrumented_client.registry.models["mlx_whisper:org/model"]
    assert registered.quantization == "f16"


def test_mlx_load_reads_quantization(fake_mlx_whisper, instrumented_client):
    install_mlx(claim_instrumentation(instrumented_client, "mlx_whisper"))
    fake_mlx_whisper.transcribe("audio.wav", path_or_hf_repo="org/model-q4")
    fake_mlx_whisper.transcribe("audio.wav", path_or_hf_repo="org/model-q4")
    (load,) = events_of(instrumented_client.events, "model_load")
    assert load["load"]["accelerator"] == "mps"
    assert len(events_of(instrumented_client.events, "inference")) == 2
    registered = instrumented_client.registry.models["mlx_whisper:org/model-q4"]
    assert registered.quantization == "int4"


def test_mlx_reload_after_switching_models_is_reported(
    fake_mlx_whisper, instrumented_client
):
    install_mlx(claim_instrumentation(instrumented_client, "mlx_whisper"))
    for repo in ("org/a", "org/b", "org/a"):
        fake_mlx_whisper.transcribe("audio.wav", path_or_hf_repo=repo)
    loads = events_of(instrumented_client.events, "model_load")
    assert [e["model_id"] for e in loads] == [
        "mlx_whisper:org/a",
        "mlx_whisper:org/b",
        "mlx_whisper:org/a",
    ]


def test_mlx_error_is_tracked_and_reraised(fake_mlx_whisper, instrumented_client):
    install_mlx(claim_instrumentation(instrumented_client, "mlx_whisper"))
    with pytest.raises(ValueError, match="cannot load"):
        fake_mlx_whisper.transcribe("audio.wav", path_or_hf_repo="broken")
    errors = events_of(instrumented_client.events, "error")
    assert [e["model_id"] for e in errors] == ["mlx_whisper:broken"]


def test_mlx_sends_nothing_after_client_closes(fake_mlx_whisper, instrumented_client):
    install_mlx(claim_instrumentation(instrumented_client, "mlx_whisper"))
    fake_mlx_whisper.transcribe("audio.wav", path_or_hf_repo="org/model")
    instrumented_client.close()
    assert fake_mlx_whisper.transcribe("audio.wav", path_or_hf_repo="org/model")
    assert len(events_of(instrumented_client.events, "inference")) == 1


def test_mlx_replacement_client_reports_the_cached_model(
    fake_mlx_whisper, instrumented_client
):
    install_mlx(claim_instrumentation(instrumented_client, "mlx_whisper"))
    fake_mlx_whisper.transcribe("audio.wav", path_or_hf_repo="org/model")
    instrumented_client.close()
    replacement = stubbed_client()
    replacement_events = []
    replacement.publish = replacement_events.append
    claim_instrumentation(replacement, "mlx_whisper")
    fake_mlx_whisper.transcribe("audio.wav", path_or_hf_repo="org/model")
    (load,) = events_of(replacement_events, "model_load")
    assert load["attributes"]["model_cache_hit"] is True
    assert len(events_of(replacement_events, "inference")) == 1
    assert len(events_of(instrumented_client.events, "inference")) == 1


@pytest.mark.parametrize("backend", ["faster_whisper", "batched", "mlx_whisper"])
def test_instrumentation_buffers_no_payloads_with_uploads_enabled(
    backend, fake_faster_whisper, fake_mlx_whisper, instrumented_client, tmp_path, wav
):
    store = AttachmentStore(str(tmp_path / "attachments"))
    instrumented_client.attachment_manager = AttachmentManager(store)
    if backend == "mlx_whisper":
        install_mlx(claim_instrumentation(instrumented_client, "mlx_whisper"))
        result = fake_mlx_whisper.transcribe(str(wav), path_or_hf_repo="org/model")
        assert result["text"]
    else:
        install_auto_load_patch(
            claim_instrumentation(instrumented_client, "faster_whisper")
        )
        model = fake_faster_whisper.WhisperModel("small")
        if backend == "batched":
            model = fake_faster_whisper.BatchedInferencePipeline(model)
        segments, _ = model.transcribe(str(wav))
        assert list(segments)
    (event,) = events_of(instrumented_client.events, "inference")
    assert "attachments" not in event
    assert store.length() == 0
