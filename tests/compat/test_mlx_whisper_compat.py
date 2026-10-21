"""Real MLX Whisper loader and decoding, with locally built random weights."""

import importlib
import json
from dataclasses import asdict

import mlx.core as mx
import mlx_whisper
import numpy as np
from mlx.utils import tree_flatten
from mlx_whisper.whisper import ModelDimensions, Whisper


def test_load_and_transcribe(compat_client, tmp_path):
    dimensions = ModelDimensions(
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=8,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=51865,
        n_text_ctx=448,
        n_text_state=8,
        n_text_head=2,
        n_text_layer=1,
    )
    mx.random.seed(0)
    model = Whisper(dimensions, mx.float32)
    (tmp_path / "config.json").write_text(json.dumps(asdict(dimensions)))
    mx.save_safetensors(
        str(tmp_path / "weights.safetensors"), dict(tree_flatten(model.parameters()))
    )
    importlib.import_module("mlx_whisper.transcribe").ModelHolder.model = None
    compat_client.instrument("mlx_whisper")
    result = mlx_whisper.transcribe(
        np.zeros(16000, dtype=np.float32),
        path_or_hf_repo=str(tmp_path),
        language="en",
        fp16=False,
        sample_len=2,
        temperature=0,
        condition_on_previous_text=False,
    )
    load = compat_client.one("model_load")
    event = compat_client.one("inference")
    assert event["model_id"] == load["model_id"]
    assert event["attributes"]["transcript_word_count"] == len(result["text"].split())
    assert event["inference"]["input_meta"]["duration_ms"] == 1000
    assert event["inference"]["input_meta"]["sample_rate"] == 16000
