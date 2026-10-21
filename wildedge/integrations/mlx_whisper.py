"""Opt-in instrumentation for mlx-whisper's function API and model cache."""

from __future__ import annotations

import contextvars
import functools
import importlib
import threading
import time

from wildedge.integrations.common import (
    debug_failure,
    dtype_to_quantization,
    track_api_error,
)
from wildedge.integrations.whisper import (
    DESCRIPTOR_ATTR,
    WhisperDescriptor,
    descriptor,
    report_load,
    track_transcription,
)
from wildedge.timing import elapsed_ms

_PATCH_LOCK = threading.Lock()

# mlx_whisper.transcribe's default for path_or_hf_repo.
DEFAULT_REPO = "mlx-community/whisper-tiny"


def mlx_accelerator() -> str:
    try:
        import mlx.core as mx  # noqa: PLC0415 -- optional runtime

        return "cpu" if "cpu" in str(mx.default_device()) else "gpu"
    except Exception as exc:
        debug_failure("mlx_whisper", "device detection", exc)
        return "gpu"


def mlx_quantization(model: object, dtype: object) -> str:
    """Bit width of quantized layers, else the weight dtype."""
    try:
        for _, layer in model.named_modules():  # type: ignore[union-attr]
            bits = getattr(layer, "bits", None)
            if isinstance(bits, int):
                return f"q{bits}"
    except Exception as exc:
        debug_failure("mlx_whisper", "quantization detection", exc)
    return dtype_to_quantization(str(dtype)) or "f32"


def describe(model: object, path: object, dtype: object) -> WhisperDescriptor:
    return WhisperDescriptor.from_path(
        path, "mlx_whisper", mlx_accelerator(), mlx_quantization(model, dtype)
    )


def requested_model(kwargs: dict) -> tuple[str, str]:
    """Model path and dtype that ``mlx_whisper.transcribe`` uses for these arguments."""
    dtype = "float16" if kwargs.get("fp16", True) else "float32"
    return kwargs.get("path_or_hf_repo", DEFAULT_REPO), dtype


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
    # The client that owns the transcription in progress, so a load inside it
    # reports to the same client even if the default changes meanwhile.
    active_client = contextvars.ContextVar("wildedge_mlx_whisper_client", default=None)

    @functools.wraps(original_load)
    def load(path_or_hf_repo, *args, **kwargs):
        client = active_client.get() or client_ref()
        t0 = time.perf_counter()
        model = original_load(path_or_hf_repo, *args, **kwargs)
        try:
            dtype = kwargs.get("dtype", args[0] if args else "float32")
            setattr(model, DESCRIPTOR_ATTR, describe(model, path_or_hf_repo, dtype))
            if client is not None and not client.closed:
                report_load(client, model, elapsed_ms(t0))
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
                if not client.closed:
                    path, dtype = requested_model(kwargs)
                    handle = client.register_model(describe(None, path, dtype))
                    track_api_error(handle, exc, elapsed_ms(t0))
            except Exception as telemetry_exc:
                debug_failure("mlx_whisper", "error tracking", telemetry_exc)
            raise
        finally:
            active_client.reset(token)
        if client.closed:
            return result
        try:
            # mlx_whisper keeps the last model in ModelHolder. It was either
            # loaded by this call (and reported by load above) or reused.
            model = module.ModelHolder.model
            info = descriptor(model)
            if info is None:
                # Loaded before instrumentation started.
                info = describe(model, *requested_model(kwargs))
                setattr(model, DESCRIPTOR_ATTR, info)
            if info.model_id not in client.auto_loaded:
                # Cached before this client existed: report it once, with no
                # load time of its own.
                client._on_model_auto_loaded(
                    model,
                    load_ms=0,
                    load_kwargs={"context": {"model_cache_hit": True}},
                )
            track_transcription(
                client.register_model(model),
                audio,
                result.get("text", ""),
                duration_ms=elapsed_ms(t0),
                language=result.get("language"),
            )
        except Exception as exc:
            debug_failure("mlx_whisper", "transcription tracking", exc)
        return result

    transcribe.__wildedge_patch_name__ = "mlx_whisper"
    module.load_model = load
    loaders.load_model = load
    module.transcribe = transcribe
    root.transcribe = transcribe
