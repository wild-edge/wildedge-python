"""Opt-in instrumentation for mlx-whisper's function API and model cache."""

from __future__ import annotations

import contextvars
import functools
import importlib
import inspect
import threading
import time

from wildedge import accelerators
from wildedge.integrations.common import (
    debug_failure,
    dtype_to_quantization,
    track_api_error,
)
from wildedge.integrations.whisper import (
    DESCRIPTOR_ATTR,
    MLX_WHISPER,
    WhisperDescriptor,
    descriptor,
    report_cached_load,
    report_load,
    track_transcription,
)
from wildedge.timing import elapsed_ms

_PATCH_LOCK = threading.Lock()

# Defaults of mlx_whisper.transcribe, which picks the model and its dtype from
# path_or_hf_repo and decode_options["fp16"].
DEFAULT_REPO = "mlx-community/whisper-tiny"
DEFAULT_FP16 = True


def mlx_accelerator() -> str:
    """MLX runs on the Apple GPU unless the app switched its default to CPU."""
    try:
        import mlx.core as mx  # noqa: PLC0415 -- optional runtime

        return accelerators.CPU if mx.default_device() == mx.cpu else accelerators.MPS
    except Exception as exc:
        debug_failure(MLX_WHISPER, "device detection", exc)
        return accelerators.MPS


def mlx_quantization(model: object, dtype: object) -> str | None:
    """Bit width of quantized layers, else the weight dtype, if recognised."""
    if model is not None:
        try:
            for _, layer in model.named_modules():  # type: ignore[attr-defined]
                bits = getattr(layer, "bits", None)
                if isinstance(bits, int):
                    return f"q{bits}"
        except Exception as exc:
            debug_failure(MLX_WHISPER, "quantization detection", exc)
    return dtype_to_quantization(str(dtype))


def describe(path: object, dtype: object, model: object = None) -> WhisperDescriptor:
    """Describe a model; without ``model`` (when loading failed) only the dtype
    is known."""
    return WhisperDescriptor.from_path(
        path, MLX_WHISPER, mlx_accelerator(), mlx_quantization(model, dtype)
    )


def requested_model(kwargs: dict) -> tuple[str, str]:
    """Model path and dtype that ``mlx_whisper.transcribe`` uses for these arguments."""
    dtype = "float16" if kwargs.get("fp16", DEFAULT_FP16) else "float32"
    return kwargs.get("path_or_hf_repo", DEFAULT_REPO), dtype


def install_auto_load_patch(client_ref):
    with _PATCH_LOCK:
        _install(client_ref)


def _install(client_ref):
    root = importlib.import_module("mlx_whisper")
    module = importlib.import_module("mlx_whisper.transcribe")
    loaders = importlib.import_module("mlx_whisper.load_models")
    original = module.transcribe
    if getattr(original, "__wildedge_patch_name__", None) == MLX_WHISPER:
        return
    original_load = module.load_model
    load_signature = inspect.signature(original_load)
    # The client that owns the transcription in progress, so a load inside it
    # reports to the same client even if the default changes meanwhile.
    active_client = contextvars.ContextVar("wildedge_mlx_whisper_client", default=None)

    @functools.wraps(original_load)
    def load(*args, **kwargs):
        client = active_client.get() or client_ref()
        t0 = time.perf_counter()
        model = original_load(*args, **kwargs)
        try:
            bound = load_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            info = describe(
                bound.arguments["path_or_hf_repo"],
                bound.arguments.get("dtype"),
                model,
            )
            setattr(model, DESCRIPTOR_ATTR, info)
            if client is not None and not client.closed:
                report_load(client, model, elapsed_ms(t0))
        except Exception as exc:
            debug_failure(MLX_WHISPER, "model registration", exc)
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
                    handle = client.register_model(describe(*requested_model(kwargs)))
                    track_api_error(handle, exc, elapsed_ms(t0))
            except Exception as telemetry_exc:
                debug_failure(MLX_WHISPER, "error tracking", telemetry_exc)
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
                info = describe(*requested_model(kwargs), model)
                setattr(model, DESCRIPTOR_ATTR, info)
            if info.model_id not in client.auto_loaded:
                report_cached_load(client, model)
            handle = client.registry.handles.get(
                info.model_id
            ) or client.register_model(model)
            track_transcription(
                handle,
                audio,
                result.get("text", ""),
                duration_ms=elapsed_ms(t0),
                language=result.get("language"),
            )
        except Exception as exc:
            debug_failure(MLX_WHISPER, "transcription tracking", exc)
        return result

    transcribe.__wildedge_patch_name__ = MLX_WHISPER
    module.load_model = load
    loaders.load_model = load
    module.transcribe = transcribe
    root.transcribe = transcribe
