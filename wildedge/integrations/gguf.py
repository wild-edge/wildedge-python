"""GGUF / llama.cpp integration."""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from wildedge import config
from wildedge.device import CURRENT_PLATFORM
from wildedge.events.inference import GenerationOutputMeta, TextInputMeta
from wildedge.integrations.base import BaseExtractor, patch_instance_call_once
from wildedge.logging import logger
from wildedge.model import ModelInfo
from wildedge.timing import elapsed_ms

try:
    import llama_cpp as _llama_cpp
except ImportError:
    _llama_cpp = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from wildedge.model import ModelHandle

# Matches quantization suffixes in filenames: Q4_K_M, Q8_0, Q4_0, F16, etc.
_QUANT_RE = re.compile(
    r"[._-](Q\d+_[KF\d]+(?:_[MSL])?|[Ff]16|[Ff]32|[Ii]8)\b",
    re.IGNORECASE,
)

_llama_patched = False
_LLAMA_PATCH_LOCK = threading.Lock()
GGUF_CALL_PATCH_NAME = "gguf_call"
GGUF_HANDLE_ATTR = "__wildedge_gguf_handle__"
LLAMA_AUTO_LOAD_PATCH_NAME = "gguf_auto_load"


def _debug_gguf_failure(context: str, exc: BaseException) -> None:
    logger.debug("wildedge: gguf %s failed: %s", context, exc)


def _n_gpu_layers(obj: object) -> int:
    """Return the configured gpu_layers value, normalising 0x7FFFFFFF back to -1."""
    n = getattr(obj, "n_gpu_layers", None)
    if n is None:
        n = getattr(getattr(obj, "model_params", None), "n_gpu_layers", None)
    if n is None:
        return 0
    n = int(n)
    return -1 if n == 0x7FFFFFFF else n


def _detect_accelerator(obj: object) -> str:
    try:
        if _n_gpu_layers(obj) != 0:
            return CURRENT_PLATFORM.gpu_accelerator_for_offload()
    except Exception as exc:
        _debug_gguf_failure("accelerator detection", exc)
    return "cpu"


def _parse_quantization(filename: str) -> str | None:
    match = _QUANT_RE.search(filename)
    if match:
        return match.group(1).lower()
    return None


def _build_patched_call(original_call):
    def patched_call(self_inner, *args, **kwargs):
        handle = getattr(self_inner, GGUF_HANDLE_ATTR, None)
        if handle is None:
            return original_call(self_inner, *args, **kwargs)

        prompt = args[0] if args else kwargs.get("prompt", "")
        t0 = time.perf_counter()
        try:
            result = original_call(self_inner, *args, **kwargs)
            duration_ms = elapsed_ms(t0)
            tokens_in = None
            tokens_out = None
            try:
                if isinstance(result, dict):
                    usage = result.get("usage", {})
                    tokens_in = usage.get("prompt_tokens")
                    tokens_out = usage.get("completion_tokens")
            except Exception as exc:
                _debug_gguf_failure("usage extraction", exc)

            input_meta = None
            if isinstance(prompt, str) and prompt:
                input_meta = TextInputMeta(
                    char_count=len(prompt),
                    word_count=len(prompt.split()),
                    token_count=tokens_in,
                )

            output_meta = None
            if tokens_out is not None:
                tps = (
                    round(tokens_out / duration_ms * 1000, 1)
                    if duration_ms > 0
                    else None
                )
                output_meta = GenerationOutputMeta(
                    task="generation",
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    tokens_per_second=tps,
                )

            handle.track_inference(
                duration_ms=duration_ms,
                input_modality="text",
                output_modality="text",
                input_meta=input_meta,
                success=True,
                output_meta=output_meta,
            )
            return result
        except Exception as exc:
            handle.track_error(
                error_code="UNKNOWN",
                error_message=str(exc)[: config.ERROR_MSG_MAX_LEN],
            )
            raise

    return patched_call


class GgufExtractor(BaseExtractor):
    # llama-cpp-python's Llama class is the universal GGUF loader and covers Mistral,
    # Phi, Gemma, etc. Model family is read from general.architecture in GGUF metadata.
    def can_handle(self, obj: object) -> bool:
        # String check avoids importing llama_cpp when it's not installed
        return type(obj).__name__ == "Llama"

    def extract_info(
        self, obj: object, overrides: dict
    ) -> tuple[str | None, ModelInfo]:
        model_path = getattr(obj, "model_path", None) or ""
        stem = Path(model_path).stem if model_path else None

        model_id = overrides.pop("id", None) or stem

        quantization = overrides.pop("quantization", None)
        if quantization is None and stem:
            quantization = _parse_quantization(stem)
        if quantization is None:
            try:
                metadata = getattr(obj, "metadata", {}) or {}
                file_type = metadata.get("general.file_type")
                if file_type:
                    quantization = str(file_type).lower()
            except Exception as exc:
                _debug_gguf_failure("quantization metadata read", exc)
        if quantization is None:
            logger.warning(
                "wildedge: GGUF quantization could not be detected - sending as null"
            )

        family = overrides.pop("family", None)
        if family is None:
            try:
                metadata = getattr(obj, "metadata", {}) or {}
                family = metadata.get("general.architecture")
            except Exception as exc:
                _debug_gguf_failure("family metadata read", exc)
        if family is None:
            logger.warning(
                "wildedge: GGUF model family could not be detected - sending as null"
            )

        version = overrides.pop("version", None)
        if version is None:
            try:
                metadata = getattr(obj, "metadata", {}) or {}
                version = metadata.get("general.version")
            except Exception as exc:
                _debug_gguf_failure("version metadata read", exc)
        if version is None:
            logger.warning(
                "wildedge: GGUF model version could not be detected - sending as null"
            )
            version = "unknown"

        source = overrides.pop("source", "local")
        model_name = stem or "gguf-model"

        info = ModelInfo(
            model_name=model_name,
            model_version=version,
            model_source=source,
            model_format="gguf",
            model_family=family,
            quantization=quantization,
        )
        for k, v in overrides.items():
            if hasattr(info, k):
                setattr(info, k, v)

        return model_id, info

    def memory_bytes(self, obj: object) -> int | None:
        try:
            return Path(getattr(obj, "model_path", None) or "").stat().st_size
        except Exception as exc:
            _debug_gguf_failure("model size detection", exc)
            return None

    def install_hooks(self, obj: object, handle: ModelHandle) -> None:
        handle.detected_accelerator = _detect_accelerator(obj)
        setattr(obj, GGUF_HANDLE_ATTR, handle)
        patch_instance_call_once(
            obj,
            patch_name=GGUF_CALL_PATCH_NAME,
            make_patched_call=_build_patched_call,
        )

    @classmethod
    def install_auto_load_patch(cls, client_ref: object) -> None:
        """Patch llama_cpp.Llama.__init__ for automatic load/unload tracking.

        Called once at WildEdge client initialisation. Any subsequent
        ``Llama(...)`` construction is timed and registered automatically;
        no ``client.load()`` call is needed.
        """
        global _llama_patched
        if _llama_patched or _llama_cpp is None:
            return

        with _LLAMA_PATCH_LOCK:
            if _llama_patched:
                return

            original_init = _llama_cpp.Llama.__init__
            if (
                getattr(original_init, "__wildedge_patch_name__", None)
                == LLAMA_AUTO_LOAD_PATCH_NAME
            ):
                _llama_patched = True
                return

            def patched_init(self_inner, *args, **kwargs):  # type: ignore[no-untyped-def]
                t0 = time.perf_counter()
                original_init(self_inner, *args, **kwargs)
                load_ms = elapsed_ms(t0)
                c = client_ref()  # type: ignore[call-arg]
                if c is not None and not c.closed:
                    load_kwargs: dict = {}
                    try:
                        load_kwargs["context_length"] = self_inner.n_ctx()
                    except Exception as exc:
                        _debug_gguf_failure("context length extraction", exc)
                    try:
                        n_gpu = _n_gpu_layers(self_inner)
                        if n_gpu != 0:
                            load_kwargs["gpu_layers"] = n_gpu
                    except Exception as exc:
                        _debug_gguf_failure("gpu layers extraction", exc)
                    c._on_model_auto_loaded(
                        self_inner, load_ms=load_ms, load_kwargs=load_kwargs
                    )

            patched_init.__wildedge_patch_name__ = LLAMA_AUTO_LOAD_PATCH_NAME  # type: ignore[attr-defined]
            patched_init.__wildedge_original_call__ = original_init  # type: ignore[attr-defined]
            _llama_cpp.Llama.__init__ = patched_init
            _llama_patched = True
