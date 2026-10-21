"""GGUF / llama.cpp integration."""

from __future__ import annotations

import contextvars
import functools
import inspect
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from wildedge import accelerators, constants, quantizations
from wildedge.events.inference import (
    GenerationConfig,
    GenerationOutputMeta,
    TextInputMeta,
)
from wildedge.integrations.base import BaseExtractor, patch_instance_call_once
from wildedge.integrations.common import (
    SyncStreamWrapper,
    build_input_meta,
    debug_failure,
    stream_error_code,
)
from wildedge.logging import logger
from wildedge.model import ModelInfo
from wildedge.platforms import CURRENT_PLATFORM
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
# llama-cpp-python's __call__ and create_chat_completion (via the chat handler)
# both run create_completion, so patching it tracks every entry point once.
GGUF_COMPLETION_PATCH_NAME = "gguf_create_completion"
GGUF_PATCHED_METHOD = "create_completion"
GGUF_CHAT_METHOD = "create_chat_completion"
GGUF_HANDLE_ATTR = "__wildedge_gguf_handle__"
LLAMA_AUTO_LOAD_PATCH_NAME = "gguf_auto_load"


debug_gguf_failure = functools.partial(debug_failure, "llama_cpp")


def n_gpu_layers(obj: object) -> int:
    """Return the configured gpu_layers value, normalising 0x7FFFFFFF back to -1."""
    n = getattr(obj, "n_gpu_layers", None)
    if n is None:
        n = getattr(getattr(obj, "model_params", None), "n_gpu_layers", None)
    if n is None:
        return 0
    n = int(n)
    return -1 if n == 0x7FFFFFFF else n


def detect_accelerator(obj: object) -> str:
    try:
        if n_gpu_layers(obj) != 0:
            return CURRENT_PLATFORM.gpu_accelerator_for_offload()
    except Exception as exc:
        debug_gguf_failure("accelerator detection", exc)
    return accelerators.CPU


# Filename suffixes that have a shared label under another name.
_SUFFIX_LABELS = {"i8": quantizations.INT8}

# llama.cpp file types (gguf-py's LlamaFileType) as stored in the model's
# general.file_type metadata. Removed and unknown types are left out.
GGUF_FILE_TYPES = {
    0: quantizations.F32,
    1: quantizations.F16,
    2: "q4_0",
    3: "q4_1",
    7: "q8_0",
    8: "q5_0",
    9: "q5_1",
    10: "q2_k",
    11: "q3_k_s",
    12: "q3_k_m",
    13: "q3_k_l",
    14: "q4_k_s",
    15: "q4_k_m",
    16: "q5_k_s",
    17: "q5_k_m",
    18: "q6_k",
    19: "iq2_xxs",
    20: "iq2_xs",
    21: "q2_k_s",
    22: "iq3_xs",
    23: "iq3_xxs",
    24: "iq1_s",
    25: "iq4_nl",
    26: "iq3_s",
    27: "iq3_m",
    28: "iq2_s",
    29: "iq2_m",
    30: "iq4_xs",
    31: "iq1_m",
    32: quantizations.BF16,
    36: "tq1_0",
    37: "tq2_0",
    38: "mxfp4_moe",
    39: "nvfp4",
    40: "q1_0",
}


def parse_quantization(filename: str) -> str | None:
    match = _QUANT_RE.search(filename)
    if match:
        suffix = match.group(1).lower()
        return _SUFFIX_LABELS.get(suffix, suffix)
    return None


def file_type_quantization(file_type: object) -> str | None:
    """Label for a general.file_type metadata value, such as "15" (q4_k_m)."""
    try:
        return GGUF_FILE_TYPES.get(int(str(file_type)))
    except ValueError:
        return None


# Messages of the create_chat_completion call in progress, so the
# create_completion it runs can record the event as a chat turn.
_chat_messages: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "wildedge_gguf_chat_messages", default=None
)


def make_gguf_input_meta(
    prompt: object, tokens_in: int | None, messages: list | None = None
) -> TextInputMeta | None:
    if messages is not None:
        meta = build_input_meta(messages, tokens_in)
        if meta is not None:
            user_turns = sum(
                1 for m in messages if isinstance(m, dict) and m.get("role") == "user"
            )
            meta.turn_index = max(user_turns - 1, 0)
        return meta
    if not isinstance(prompt, str) or not prompt:
        return None
    return TextInputMeta(
        char_count=len(prompt),
        word_count=len(prompt.split()),
        token_count=tokens_in,
    )


def make_gguf_output_meta(
    tokens_in: int | None,
    tokens_out: int | None,
    stop_reason: str | None,
    ttft_ms: int | None,
    duration_ms: int,
) -> GenerationOutputMeta | None:
    if tokens_out is None and ttft_ms is None and stop_reason is None:
        return None
    tps = (
        round(tokens_out / duration_ms * 1000, 1)
        if duration_ms > 0 and tokens_out
        else None
    )
    return GenerationOutputMeta(
        task="generation",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        time_to_first_token_ms=ttft_ms,
        tokens_per_second=tps,
        stop_reason=stop_reason,
    )


def make_gguf_generation_config(kwargs: dict) -> GenerationConfig | None:
    stop = kwargs.get("stop")
    if isinstance(stop, str):
        stop_count: int | None = 1
    elif isinstance(stop, (list, tuple)):
        stop_count = len(stop) or None
    else:
        stop_count = None
    config = GenerationConfig(
        temperature=kwargs.get("temperature"),
        top_p=kwargs.get("top_p"),
        top_k=kwargs.get("top_k"),
        max_tokens=kwargs.get("max_tokens"),
        repetition_penalty=kwargs.get("repeat_penalty"),
        frequency_penalty=kwargs.get("frequency_penalty"),
        presence_penalty=kwargs.get("presence_penalty"),
        seed=kwargs.get("seed"),
        stop_sequences_count=stop_count,
    )
    return config if config.to_dict() else None


def make_gguf_stream_callbacks(
    handle: object,
    prompt: object,
    messages: list | None = None,
    generation_config: GenerationConfig | None = None,
) -> tuple:
    """Return (on_chunk, on_done) callbacks for a llama-cpp-python streaming response.

    Chunks are dicts; usage appears in the final chunk when available.
    """
    tokens_in: list[int | None] = [None]
    tokens_out: list[int | None] = [None]
    stop_reason: list[str | None] = [None]

    def on_chunk(chunk: object) -> None:
        if not isinstance(chunk, dict):
            return
        usage = chunk.get("usage")
        if usage:
            tokens_in[0] = usage.get("prompt_tokens")
            tokens_out[0] = usage.get("completion_tokens")
        choices = chunk.get("choices") or []
        if choices and isinstance(choices[0], dict):
            reason = choices[0].get("finish_reason")
            if reason:
                stop_reason[0] = reason

    def on_done(
        duration_ms: int, ttft_ms: int | None, *, outcome: str = "completed"
    ) -> None:
        ti, to, sr = tokens_in[0], tokens_out[0], stop_reason[0]
        handle.track_inference(  # type: ignore[union-attr]
            duration_ms=duration_ms,
            input_modality="text",
            output_modality="generation",
            input_meta=make_gguf_input_meta(prompt, ti, messages),
            success=outcome == "completed",
            error_code=stream_error_code(outcome),
            output_meta=make_gguf_output_meta(ti, to, sr, ttft_ms, duration_ms),
            generation_config=generation_config,
        )

    return on_chunk, on_done


def call_arguments(signature, self_inner, args, kwargs) -> dict:
    """Arguments of the call with the method's defaults filled in.

    The chat handler and __call__ pass every sampling argument explicitly; this
    gives a direct create_completion call the same generation config.
    """
    if signature is not None:
        try:
            bound = signature.bind(self_inner, *args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            arguments.update(arguments.pop("kwargs", None) or {})
            return arguments
        except Exception as exc:
            debug_gguf_failure("argument binding", exc)
    arguments = dict(kwargs)
    if args:
        arguments.setdefault("prompt", args[0])
    return arguments


def build_patched_call(original_call):
    try:
        signature = inspect.signature(original_call)
    except (TypeError, ValueError):
        signature = None

    def patched_call(self_inner, *args, **kwargs):
        handle = getattr(self_inner, GGUF_HANDLE_ATTR, None)
        if handle is None:
            return original_call(self_inner, *args, **kwargs)

        arguments = call_arguments(signature, self_inner, args, kwargs)
        prompt = arguments.get("prompt", "")
        messages = _chat_messages.get()
        is_streaming: bool = bool(arguments.get("stream", False))
        generation_config = None
        try:
            generation_config = make_gguf_generation_config(arguments)
        except Exception as exc:
            debug_gguf_failure("generation config extraction", exc)
        t0 = time.perf_counter()
        try:
            result = original_call(self_inner, *args, **kwargs)
            if is_streaming:
                on_chunk, on_done = make_gguf_stream_callbacks(
                    handle, prompt, messages, generation_config
                )
                return SyncStreamWrapper(result, handle, t0, on_chunk, on_done)
            duration_ms = elapsed_ms(t0)
            tokens_in = None
            tokens_out = None
            stop_reason = None
            try:
                if isinstance(result, dict):
                    usage = result.get("usage") or {}
                    tokens_in = usage.get("prompt_tokens")
                    tokens_out = usage.get("completion_tokens")
                    choices = result.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        stop_reason = choices[0].get("finish_reason")
            except Exception as exc:
                debug_gguf_failure("usage extraction", exc)
            handle.track_inference(
                duration_ms=duration_ms,
                input_modality="text",
                output_modality="generation",
                input_meta=make_gguf_input_meta(prompt, tokens_in, messages),
                success=True,
                output_meta=make_gguf_output_meta(
                    tokens_in, tokens_out, stop_reason, None, duration_ms
                ),
                generation_config=generation_config,
            )
            return result
        except Exception as exc:
            handle.track_error(
                error_code="UNKNOWN",
                error_message=str(exc)[: constants.ERROR_MSG_MAX_LEN],
            )
            raise

    return patched_call


def build_patched_chat(original_chat):
    """Expose the chat messages to the create_completion call the handler makes.

    Tracking itself stays in create_completion, so each call is counted once.
    """

    def patched_chat(self_inner, *args, **kwargs):
        if getattr(self_inner, GGUF_HANDLE_ATTR, None) is None:
            return original_chat(self_inner, *args, **kwargs)
        messages = args[0] if args else kwargs.get("messages")
        token = _chat_messages.set(list(messages) if messages else None)
        try:
            return original_chat(self_inner, *args, **kwargs)
        finally:
            _chat_messages.reset(token)

    return patched_chat


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
            quantization = parse_quantization(stem)
        if quantization is None:
            try:
                metadata = getattr(obj, "metadata", {}) or {}
                file_type = metadata.get("general.file_type")
                if file_type is not None:
                    quantization = file_type_quantization(file_type)
            except Exception as exc:
                debug_gguf_failure("quantization metadata read", exc)
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
                debug_gguf_failure("family metadata read", exc)
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
                debug_gguf_failure("version metadata read", exc)
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
            debug_gguf_failure("model size detection", exc)
            return None

    def install_hooks(self, obj: object, handle: ModelHandle) -> None:
        handle.detected_accelerator = detect_accelerator(obj)
        setattr(obj, GGUF_HANDLE_ATTR, handle)
        patch_instance_call_once(
            obj,
            patch_name=GGUF_COMPLETION_PATCH_NAME,
            make_patched_call=build_patched_call,
            method=GGUF_PATCHED_METHOD,
            extra_methods={GGUF_CHAT_METHOD: build_patched_chat},
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
                        debug_gguf_failure("context length extraction", exc)
                    try:
                        n_gpu = n_gpu_layers(self_inner)
                        if n_gpu != 0:
                            load_kwargs["gpu_layers"] = n_gpu
                    except Exception as exc:
                        debug_gguf_failure("gpu layers extraction", exc)
                    c._on_model_auto_loaded(
                        self_inner, load_ms=load_ms, load_kwargs=load_kwargs
                    )

            patched_init.__wildedge_patch_name__ = LLAMA_AUTO_LOAD_PATCH_NAME  # type: ignore[attr-defined]
            patched_init.__wildedge_original_call__ = original_init  # type: ignore[attr-defined]
            _llama_cpp.Llama.__init__ = patched_init
            _llama_patched = True


# Entry point named by the integration registry.
install_auto_load_patch = GgufExtractor.install_auto_load_patch
