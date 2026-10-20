"""Whisper adapters preserve results, lazy lifecycles, and capture privacy."""

import sys
import wave
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from wildedge.instrumentation import claim_instrumentation
from wildedge.integrations.faster_whisper import (
    install_auto_load_patch,
    wrap_transcribe,
)
from wildedge.integrations.mlx_whisper import install_auto_load_patch as install_mlx
from wildedge.integrations.whisper import (
    DESCRIPTOR_ATTR,
    HANDLE_ATTR,
    WhisperDescriptor,
    WhisperExtractor,
    audio_meta,
    track_transcription,
)


@pytest.fixture
def handle():
    return SimpleNamespace(
        capture_attachments=None, track_inference=MagicMock(), track_error=MagicMock()
    )


def test_metadata_and_no_default_capture(tmp_path, handle):
    path = tmp_path / "input.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 8000)
    track_transcription(handle, path, "hello world", "en", 123, probability=0.9)
    event = handle.track_inference.call_args.kwargs
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
    assert event["attachments"] is None
    handle.capture_attachments = MagicMock()
    track_transcription(handle, path, "hello world", "en", 123)
    attachments = handle.track_inference.call_args.kwargs["attachments"]
    assert [a.role for a in attachments] == ["input", "output"]
    assert attachments[1].data == b"hello world"


def test_unknown_audio_is_not_consumed():
    source = MagicMock()
    assert audio_meta(source).to_dict() == {}
    source.read.assert_not_called()


def test_extractor(tmp_path):
    model = SimpleNamespace()
    setattr(
        model,
        DESCRIPTOR_ATTR,
        WhisperDescriptor(str(tmp_path), "faster_whisper", "cpu", "int8"),
    )
    extractor = WhisperExtractor()
    ident, info = extractor.extract_info(model, {})
    assert ident == f"faster_whisper:{tmp_path.name}"
    assert info.model_format == "ctranslate2"
    assert info.model_family == "whisper"
    assert info.quantization == "int8"


@pytest.mark.parametrize("outcome", ["completed", "closed", "error"])
def test_lazy_segments(handle, outcome):
    segment = SimpleNamespace(text="hello world")
    info = SimpleNamespace(language="en", language_probability=0.8, duration=2)

    def segments():
        yield segment
        if outcome == "error":
            raise ValueError("decoder failure")

    model = SimpleNamespace()
    setattr(model, HANDLE_ATTR, handle)
    call = wrap_transcribe(lambda *_: (segments(), info))
    stream, returned_info = call(model, "audio.wav")
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
    assert event["input_meta"].duration_ms == 2000


def test_faster_constructor_routes_once(monkeypatch):
    module = ModuleType("faster_whisper")

    class Model:
        def __init__(self, model_size_or_path, device="cpu", compute_type="int8"):
            self.model = SimpleNamespace(device=device, compute_type=compute_type)

        def transcribe(self, audio):
            return iter([]), SimpleNamespace()

    module.WhisperModel = Model
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    client = SimpleNamespace(
        closed=False, _on_model_auto_loaded=MagicMock(), register_model=MagicMock()
    )
    install_auto_load_patch(lambda: client)
    install_auto_load_patch(lambda: client)
    model = Model("small", device="cuda", compute_type="float16")
    client._on_model_auto_loaded.assert_called_once()
    assert getattr(model, DESCRIPTOR_ATTR).accelerator == "gpu"
    assert getattr(model, DESCRIPTOR_ATTR).quantization == "float16"


def test_mlx_result_identity_and_cached_load(monkeypatch, handle):
    root, module, loaders = (
        ModuleType(n)
        for n in ("mlx_whisper", "mlx_whisper.transcribe", "mlx_whisper.load_models")
    )
    result = {"text": "hello world", "language": "en"}
    module.ModelHolder = SimpleNamespace(model=SimpleNamespace())
    module.transcribe = lambda *a, **kw: result
    module.load_model = loaders.load_model = lambda *a, **kw: SimpleNamespace()
    for mod in (root, module, loaders):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    client = SimpleNamespace(
        closed=False,
        auto_loaded=set(),
        register_model=MagicMock(return_value=handle),
        _on_model_auto_loaded=MagicMock(),
    )
    handle.model_id = "mlx_whisper:test"
    install_mlx(lambda: client)
    assert root.transcribe("audio.wav", path_or_hf_repo="test") is result
    client._on_model_auto_loaded.assert_called_once()
    assert client._on_model_auto_loaded.call_args.kwargs["load_kwargs"]["context"][
        "model_cache_hit"
    ]
    handle.track_inference.assert_called_once()


def test_transcribe_errors_preserve_exception(handle):
    failure = ValueError("invalid audio")

    def original(*_args):
        raise failure

    model = SimpleNamespace()
    setattr(model, HANDLE_ATTR, handle)
    handle.track_error.side_effect = RuntimeError("telemetry failed")
    with pytest.raises(ValueError, match="invalid audio") as raised:
        wrap_transcribe(original)(model, "missing.wav")
    assert raised.value is failure
    handle.track_inference.assert_not_called()


def test_lazy_telemetry_failure_does_not_break_iterator(handle):
    handle.track_inference.side_effect = RuntimeError("telemetry failed")
    model = SimpleNamespace()
    setattr(model, HANDLE_ATTR, handle)
    segment = SimpleNamespace(text="hello")
    stream, _ = wrap_transcribe(lambda *_: (iter([segment]), SimpleNamespace()))(
        model, "audio.wav"
    )
    assert list(stream) == [segment]
    assert list(stream) == []
    handle.track_inference.assert_called_once()


def test_mlx_load_quantization_cache_and_closed_owner(
    monkeypatch, client_with_stubbed_runtime, handle
):
    root, module, loaders = (
        ModuleType(n)
        for n in ("mlx_whisper", "mlx_whisper.transcribe", "mlx_whisper.load_models")
    )
    model = SimpleNamespace(named_modules=lambda: [("", SimpleNamespace(bits=4))])
    module.ModelHolder = SimpleNamespace(model=None)
    module.load_model = loaders.load_model = lambda *_a, **_kw: model

    def original(_audio, **kwargs):
        if module.ModelHolder.model is None:
            module.ModelHolder.model = module.load_model(kwargs["path_or_hf_repo"])
        return {"text": "hello world", "language": "en"}

    module.transcribe = original
    for mod in (root, module, loaders):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    client = client_with_stubbed_runtime
    events = []
    client.publish = events.append
    install_mlx(claim_instrumentation(client, "mlx_whisper"))
    root.transcribe("audio.wav", path_or_hf_repo="org/model-q4")
    root.transcribe("audio.wav", path_or_hf_repo="org/model-q4")
    assert len([e for e in events if e["event_type"] == "model_load"]) == 1
    assert len([e for e in events if e["event_type"] == "inference"]) == 2
    assert client.registry.models["mlx_whisper:org/model-q4"].quantization == "q4"
    client.close()
    root.transcribe("audio.wav", path_or_hf_repo="org/model-q4")
    assert len([e for e in events if e["event_type"] == "inference"]) == 2
    replacement = MagicMock(closed=False, auto_loaded=set())
    handle.model_id = "mlx_whisper:org/model-q4"
    replacement.register_model.return_value = handle
    claim_instrumentation(replacement, "mlx_whisper")
    root.transcribe("audio.wav", path_or_hf_repo="org/model-q4")
    replacement._on_model_auto_loaded.assert_called_once()
    assert replacement._on_model_auto_loaded.call_args.kwargs["load_kwargs"]["context"][
        "model_cache_hit"
    ]
    handle.track_inference.assert_called_once()
    assert len([e for e in events if e["event_type"] == "inference"]) == 2
