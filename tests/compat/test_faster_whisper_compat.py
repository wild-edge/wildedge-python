"""Real CTranslate2 model load and lazy transcription using Whisper tiny."""

import os
from pathlib import Path

import faster_whisper
import numpy as np
import pytest
from huggingface_hub import snapshot_download


@pytest.fixture(scope="module")
def model_path():
    # A local model can be supplied for offline development. CI caches HF downloads.
    path = os.environ.get("WILDEDGE_FASTER_WHISPER_MODEL") or snapshot_download(
        "Systran/faster-whisper-tiny",
        revision="d90ca5fe260221311c53c58e660288d3deb8d356",
        allow_patterns=["config.json", "model.bin", "tokenizer.json", "vocabulary.*"],
    )
    assert Path(path).is_dir()
    return path


def test_load_and_lazy_transcribe(compat_client, model_path):
    compat_client.instrument("faster_whisper")
    model = faster_whisper.WhisperModel(model_path, device="cpu", compute_type="int8")
    load = compat_client.one("model_load")
    if "WILDEDGE_FASTER_WHISPER_MODEL" not in os.environ:
        # Named after the hub repo, not the snapshot hash in the cache path.
        assert load["model_id"] == "faster_whisper:Systran/faster-whisper-tiny"
    segments, info = model.transcribe(
        np.zeros(16000, dtype=np.float32), language="en", beam_size=1
    )
    assert not compat_client.of_type("inference")
    text = " ".join(segment.text for segment in segments)
    event = compat_client.one("inference")
    assert event["model_id"] == load["model_id"]
    assert event["attributes"]["transcript_word_count"] == len(text.split())
    assert event["attributes"]["language_confidence"] == info.language_probability
    assert event["inference"]["input_meta"]["duration_ms"] == 1000


def test_batched_pipeline(compat_client, model_path):
    compat_client.instrument("faster_whisper")
    model = faster_whisper.WhisperModel(model_path, device="cpu", compute_type="int8")
    load = compat_client.one("model_load")
    pipeline = faster_whisper.BatchedInferencePipeline(model)
    segments, _info = pipeline.transcribe(
        np.zeros(16000, dtype=np.float32), language="en", batch_size=2
    )
    list(segments)
    event = compat_client.one("inference")
    assert event["model_id"] == load["model_id"]
