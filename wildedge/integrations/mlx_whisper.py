"""Opt-in instrumentation for mlx-whisper's function API and model cache."""

from __future__ import annotations

import contextvars
import functools
import importlib
import threading
import time

from wildedge.integrations.common import debug_failure, track_api_error
from wildedge.integrations.whisper import (
    DESCRIPTOR_ATTR,
    descriptor,
    mlx_descriptor,
    track_transcription,
)
from wildedge.timing import elapsed_ms

_PATCH_LOCK = threading.Lock()


def install_auto_load_patch(client_ref):
    with _PATCH_LOCK:
        _install(client_ref)


def _install(client_ref):
    root = importlib.import_module("mlx_whisper")
    module = importlib.import_module("mlx_whisper.transcribe")
    loaders = importlib.import_module("mlx_whisper.load_models")
    original = module.transcribe
    if getattr(original, "__wildedge_patch_name__", None) == "mlx_whisper":
        return
    original_load = module.load_model
    active_client = contextvars.ContextVar("wildedge_mlx_whisper_client", default=None)

    @functools.wraps(original_load)
    def load(path_or_hf_repo, *args, **kwargs):
        client = active_client.get() or client_ref()
        t0 = time.perf_counter()
        model = original_load(path_or_hf_repo, *args, **kwargs)
        try:
            dtype = kwargs.get("dtype", args[0] if args else "float32")
            info = mlx_descriptor(path_or_hf_repo, dtype)
            for _, layer in model.named_modules():
                bits = getattr(layer, "bits", None)
                if isinstance(bits, int):
                    info.quantization = f"q{bits}"
                    break
            setattr(model, DESCRIPTOR_ATTR, info)
            if client is not None and not client.closed:
                client._on_model_auto_loaded(model, load_ms=elapsed_ms(t0))
        except Exception as exc:
            debug_failure("mlx_whisper", "model registration", exc)
        return model

    @functools.wraps(original)
    def transcribe(audio, *args, **kwargs):
        client = client_ref()
        if client is None:
            return original(audio, *args, **kwargs)
        t0 = time.perf_counter()
        token = active_client.set(client)
        try:
            result = original(audio, *args, **kwargs)
        except Exception as exc:
            try:
                info = mlx_descriptor(
                    kwargs.get("path_or_hf_repo", "mlx-community/whisper-tiny"),
                    "float16" if kwargs.get("fp16", True) else "float32",
                )
                track_api_error(client.register_model(info), exc, elapsed_ms(t0))
            except Exception as telemetry_exc:
                debug_failure("mlx_whisper", "error tracking", telemetry_exc)
            raise
        finally:
            active_client.reset(token)
        if not client.closed:
            try:
                model = module.ModelHolder.model
                info = descriptor(model) or mlx_descriptor(
                    kwargs.get("path_or_hf_repo", "mlx-community/whisper-tiny"),
                    "float16" if kwargs.get("fp16", True) else "float32",
                )
                if model is not None:
                    setattr(model, DESCRIPTOR_ATTR, info)
                    if info.model_id not in client.auto_loaded:
                        client._on_model_auto_loaded(
                            model,
                            load_ms=0,
                            load_kwargs={"context": {"model_cache_hit": True}},
                        )
                handle = client.register_model(model if model is not None else info)
                track_transcription(
                    handle,
                    audio,
                    result.get("text", ""),
                    result.get("language"),
                    elapsed_ms(t0),
                )
            except Exception as exc:
                debug_failure("mlx_whisper", "transcription tracking", exc)
        return result

    transcribe.__wildedge_patch_name__ = "mlx_whisper"
    module.load_model = load
    loaders.load_model = load
    module.transcribe = transcribe
    root.transcribe = transcribe
