"""Apple Foundation Models integration (``apple-fm-sdk``).

Tracks the on-device model behind Apple Intelligence through Apple's Python
SDK. Sessions are tied to a model handle when they are created, and
``respond()`` and ``stream_response()`` record one inference event per call.

Token counts come from ``SystemLanguageModel.token_count()``. That takes two
extra calls into the model service per request, so it is opt-in: set
``WILDEDGE_APPLE_FM_COUNT_TOKENS=1``.
"""

from __future__ import annotations

import asyncio
import functools
import os
import platform
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

from wildedge import accelerators, constants
from wildedge.events.error import ErrorCode
from wildedge.events.inference import (
    GenerationConfig,
    GenerationOutputMeta,
    TextInputMeta,
)
from wildedge.integrations.base import BaseExtractor
from wildedge.integrations.common import (
    AsyncStreamWrapper,
    debug_failure,
    stream_error_code,
)
from wildedge.loading import optional_module
from wildedge.model import ModelInfo
from wildedge.settings import parse_bool
from wildedge.timing import elapsed_ms

# Imported when the integration is activated: importing the SDK loads its
# native library.
_fm = sys.modules.get("apple_fm_sdk")

if TYPE_CHECKING:
    from wildedge.model import ModelHandle

_apple_fm_patched = False
_APPLE_FM_PATCH_LOCK = threading.Lock()
APPLE_FM_PATCH_NAME = "apple_fm_auto_load"
APPLE_FM_HANDLE_ATTR = "__wildedge_apple_fm_handle__"
# SystemLanguageModel passes its use case to native code without keeping it.
APPLE_FM_USE_CASE_ATTR = "__wildedge_apple_fm_use_case__"
# The model a session was created with, which counts its tokens.
APPLE_FM_MODEL_ATTR = "__wildedge_apple_fm_model__"

# Read from WILDEDGE_APPLE_FM_COUNT_TOKENS when the integration is activated.
_count_tokens = False
# A default model counts tokens for sessions created without one.
_default_model: Any = None
# Strong references, so pending token counts aren't garbage collected.
_counting_tasks: set[asyncio.Task] = set()

MODEL_ID = "apple-foundation-model"
MODEL_SOURCE = "apple"
MODEL_FORMAT = "apple_fm"
APPLE_FM_PACKAGE = "apple_fm_sdk"

# The SDK's error classes, named as GenerationErrorCode names them.
PROVIDER_ERROR_CODES = {
    "ExceededContextWindowSizeError": "exceeded_context_window_size",
    "AssetsUnavailableError": "assets_unavailable",
    "GuardrailViolationError": "guardrail_violation",
    "UnsupportedGuideError": "unsupported_guide",
    "UnsupportedLanguageOrLocaleError": "unsupported_language_or_locale",
    "DecodingFailureError": "decoding_failure",
    "RateLimitedError": "rate_limited",
    "ConcurrentRequestsError": "concurrent_requests",
    "RefusalError": "refusal",
    "InvalidGenerationSchemaError": "invalid_schema",
    "ToolCallError": "tool_call",
}

# The error code sent for each SDK error. The server keeps a specific code
# as sent, and none of these failures has an HTTP status to derive one from.
ERROR_CODES_BY_PROVIDER_CODE = {
    "assets_unavailable": ErrorCode.MODEL_UNAVAILABLE,
    "exceeded_context_window_size": ErrorCode.CONTEXT_LENGTH_EXCEEDED,
    "guardrail_violation": ErrorCode.CONTENT_FILTERED,
    "refusal": ErrorCode.CONTENT_FILTERED,
    "rate_limited": ErrorCode.RATE_LIMITED,
    "concurrent_requests": ErrorCode.RATE_LIMITED,
    "unsupported_guide": ErrorCode.BAD_REQUEST,
    "unsupported_language_or_locale": ErrorCode.BAD_REQUEST,
    "invalid_schema": ErrorCode.BAD_REQUEST,
    "decoding_failure": ErrorCode.INVALID_OUTPUT,
    "tool_call": ErrorCode.TOOL_CALL_FAILED,
}

debug_apple_fm_failure = functools.partial(debug_failure, "apple_fm")

# The last is_available() answer reported per model id. A load event is sent
# only when the answer changes, since apps poll while the model downloads.
_reported_availability: dict[str, str | None] = {}
_AVAILABILITY_LOCK = threading.Lock()


def model_id_for(use_case: object) -> str:
    """``apple-foundation-model``, plus the use case when it is not GENERAL."""
    name = getattr(use_case, "name", None)
    if not name or name == "GENERAL":
        return MODEL_ID
    return f"{MODEL_ID}-{name.lower().replace('_', '-')}"


def os_version() -> str:
    """The system model ships with macOS, so its version is the OS version."""
    return platform.mac_ver()[0] or "unknown"


def provider_error_code_of(exc: BaseException) -> str | None:
    for cls in type(exc).__mro__:
        code = PROVIDER_ERROR_CODES.get(cls.__name__)
        if code is not None:
            return code
    return None


def track_apple_fm_error(
    handle: ModelHandle, exc: BaseException, duration_ms: int
) -> None:
    provider_error_code = provider_error_code_of(exc)
    handle.track_error(
        error_code=ERROR_CODES_BY_PROVIDER_CODE.get(
            provider_error_code, ErrorCode.UNKNOWN
        ),
        error_message=str(exc),
        provider_error_code=provider_error_code,
        duration_ms=duration_ms,
    )


def unavailable_error_code(reason: object) -> str:
    """``MODEL_NOT_READY`` and the like, from the SDK's unavailable reason."""
    name = getattr(reason, "name", None)
    return name if isinstance(name, str) and name else ErrorCode.UNKNOWN.value


def track_availability(
    handle: ModelHandle, available: bool, reason: object, duration_ms: int
) -> None:
    """Send a load event for an is_available() answer that changed."""
    error_code = None if available else unavailable_error_code(reason)
    with _AVAILABILITY_LOCK:
        model_id = handle.model_id
        if (
            model_id in _reported_availability
            and _reported_availability[model_id] == error_code
        ):
            return
        _reported_availability[model_id] = error_code
    handle.track_load(duration_ms=duration_ms, success=available, error_code=error_code)


def build_input_meta(prompt: object) -> tuple[str, TextInputMeta | None]:
    """Input modality and text metadata for a prompt.

    A prompt is a string, an attachment, or a list of both.
    """
    parts = prompt if isinstance(prompt, list) else [prompt]
    text = " ".join(part for part in parts if isinstance(part, str))
    if not text:
        return ("image" if parts else "text"), None
    return "text", TextInputMeta(
        char_count=len(text),
        word_count=len(text.split()),
        prompt_type="chat",
    )


def build_generation_config(options: object) -> GenerationConfig | None:
    if options is None:
        return None
    sampling = getattr(options, "sampling", None)
    return GenerationConfig(
        temperature=getattr(options, "temperature", None),
        max_tokens=getattr(options, "maximum_response_tokens", None),
        top_k=getattr(sampling, "top", None),
        top_p=getattr(sampling, "probability_threshold", None),
        seed=getattr(sampling, "seed", None),
    )


def describe_call(
    args: tuple, kwargs: dict, *, options_position: int | None
) -> tuple[str, TextInputMeta | None, GenerationConfig | None]:
    """Input modality, input metadata and generation config. Never raises."""
    try:
        prompt = kwargs.get("prompt", args[0] if args else None)
        options = kwargs.get("options")
        if (
            options is None
            and options_position is not None
            and len(args) > options_position
        ):
            options = args[options_position]
        modality, input_meta = build_input_meta(prompt)
        return modality, input_meta, build_generation_config(options)
    except Exception as exc:
        debug_apple_fm_failure("call metadata", exc)
        return "text", None, None


class AppleFmStream(AsyncStreamWrapper):
    """Stream wrapper that reports the SDK's own error names."""

    def _error(self, exc: Exception) -> None:
        if not self._finished:
            self._finished = True
            self._safe(track_apple_fm_error, self._handle, exc, elapsed_ms(self._t0))


def prompt_of(args: tuple, kwargs: dict) -> object:
    return kwargs.get("prompt", args[0] if args else None)


def counting_model(session: object) -> Any:
    """The model that counts a session's tokens."""
    global _default_model
    model = getattr(session, APPLE_FM_MODEL_ATTR, None)
    if model is not None:
        return model
    if _default_model is None and _fm is not None:
        _default_model = _fm.SystemLanguageModel()
    return _default_model


async def count_tokens(
    model: Any, prompt: object, reply: object
) -> tuple[int | None, int | None]:
    tokens_in = await model.token_count(prompt) if prompt is not None else None
    tokens_out = await model.token_count(reply) if isinstance(reply, str) else None
    return tokens_in, tokens_out


def add_token_counts(
    kwargs: dict, tokens_in: int | None, tokens_out: int | None
) -> None:
    output_meta: GenerationOutputMeta = kwargs["output_meta"]
    output_meta.tokens_in = tokens_in
    output_meta.tokens_out = tokens_out
    duration_ms = kwargs["duration_ms"]
    if tokens_out and duration_ms > 0:
        output_meta.tokens_per_second = round(tokens_out / duration_ms * 1000, 1)
    input_meta: TextInputMeta | None = kwargs.get("input_meta")
    if input_meta is not None:
        input_meta.token_count = tokens_in


async def count_into(kwargs: dict, model: Any, prompt: object, reply: object) -> None:
    try:
        add_token_counts(kwargs, *await count_tokens(model, prompt, reply))
    except Exception as exc:
        debug_apple_fm_failure("token count", exc)


def send_counted_inference(
    handle: ModelHandle, kwargs: dict, task: asyncio.Task
) -> None:
    """Send the inference once counting ends, however it ends.

    A failed count, or one cancelled before it ran (as when asyncio.run()
    shuts down), sends the event without token counts.
    """
    _counting_tasks.discard(task)
    try:
        handle.track_inference(**kwargs)
    except Exception as exc:
        debug_apple_fm_failure("inference tracking", exc)


def track_generation(
    handle: ModelHandle,
    session: object,
    prompt: object,
    reply: object,
    **kwargs: Any,
) -> None:
    """Record an inference, counting its tokens first when that is enabled.

    Counting runs in a background task, so the call returns as soon as the
    model has answered.
    """
    if _count_tokens:
        try:
            loop = asyncio.get_running_loop()
            model = counting_model(session)
            if model is not None:
                task = loop.create_task(count_into(kwargs, model, prompt, reply))
                _counting_tasks.add(task)
                task.add_done_callback(
                    functools.partial(send_counted_inference, handle, kwargs)
                )
                return
        except Exception as exc:
            # No running loop, as when an abandoned stream is collected.
            debug_apple_fm_failure("token count scheduling", exc)
    handle.track_inference(**kwargs)


# ---------------------------------------------------------------------------
# Method patches
# ---------------------------------------------------------------------------


def build_model_init_patch(original_init):  # type: ignore[no-untyped-def]
    @functools.wraps(original_init)
    def patched_init(self_inner, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_init(self_inner, *args, **kwargs)
        use_case = kwargs.get("use_case", args[0] if args else None)
        if use_case is not None:
            try:
                setattr(self_inner, APPLE_FM_USE_CASE_ATTR, use_case)
            except Exception as exc:
                debug_apple_fm_failure("use case capture", exc)

    return patched_init


def build_availability_patch(original_is_available, client_ref):  # type: ignore[no-untyped-def]
    @functools.wraps(original_is_available)
    def patched_is_available(self_inner, *args, **kwargs):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        result = original_is_available(self_inner, *args, **kwargs)
        duration_ms = elapsed_ms(t0)
        c = client_ref()  # type: ignore[call-arg]
        if c is None or c.closed:
            return result
        try:
            available, reason = result
            model_id = model_id_for(getattr(self_inner, APPLE_FM_USE_CASE_ATTR, None))
            handle = c.register_model(self_inner, model_id=model_id)
            track_availability(handle, bool(available), reason, duration_ms)
        except Exception as exc:
            debug_apple_fm_failure("availability tracking", exc)
        return result

    return patched_is_available


def build_session_init_patch(original_init, client_ref):  # type: ignore[no-untyped-def]
    @functools.wraps(original_init)
    def patched_init(self_inner, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_init(self_inner, *args, **kwargs)
        c = client_ref()  # type: ignore[call-arg]
        if c is None or c.closed:
            return
        try:
            model = kwargs.get("model", args[1] if len(args) > 1 else None)
            model_id = model_id_for(getattr(model, APPLE_FM_USE_CASE_ATTR, None))
            handle = c.register_model(self_inner, model_id=model_id)
            setattr(self_inner, APPLE_FM_HANDLE_ATTR, handle)
            setattr(self_inner, APPLE_FM_MODEL_ATTR, model)
        except Exception as exc:
            debug_apple_fm_failure("session registration", exc)

    return patched_init


def build_respond_patch(original_respond):  # type: ignore[no-untyped-def]
    @functools.wraps(original_respond)
    async def patched_respond(self_inner, *args, **kwargs):  # type: ignore[no-untyped-def]
        handle: ModelHandle | None = getattr(self_inner, APPLE_FM_HANDLE_ATTR, None)
        if handle is None:
            return await original_respond(self_inner, *args, **kwargs)
        # respond(prompt, generating=None, *, schema, json_schema, options)
        modality, input_meta, config = describe_call(
            args, kwargs, options_position=None
        )
        t0 = time.perf_counter()
        try:
            result = await original_respond(self_inner, *args, **kwargs)
        except Exception as exc:
            track_apple_fm_error(handle, exc, elapsed_ms(t0))
            raise
        track_generation(
            handle,
            self_inner,
            prompt_of(args, kwargs),
            result,
            duration_ms=elapsed_ms(t0),
            input_modality=modality,
            output_modality="generation",
            success=True,
            input_meta=input_meta,
            output_meta=GenerationOutputMeta(task="generation"),
            generation_config=config,
        )
        return result

    return patched_respond


def build_stream_patch(original_stream):  # type: ignore[no-untyped-def]
    @functools.wraps(original_stream)
    def patched_stream(self_inner, *args, **kwargs):  # type: ignore[no-untyped-def]
        stream = original_stream(self_inner, *args, **kwargs)
        handle: ModelHandle | None = getattr(self_inner, APPLE_FM_HANDLE_ATTR, None)
        if handle is None:
            return stream
        # stream_response(prompt, options=None)
        modality, input_meta, config = describe_call(args, kwargs, options_position=1)

        # Each chunk is the full text so far, so the first one marks the
        # first output, the last one is the reply, and nothing is joined.
        last_snapshot: list[object] = [None]

        def on_chunk(chunk: object) -> None:
            last_snapshot[0] = chunk

        def on_done(
            duration_ms: int, ttft_ms: int | None, *, outcome: str = "completed"
        ) -> None:
            track_generation(
                handle,
                self_inner,
                prompt_of(args, kwargs),
                last_snapshot[0] if outcome == "completed" else None,
                duration_ms=duration_ms,
                input_modality=modality,
                output_modality="generation",
                success=outcome == "completed",
                error_code=stream_error_code(outcome),
                input_meta=input_meta,
                output_meta=GenerationOutputMeta(
                    task="generation", time_to_first_token_ms=ttft_ms
                ),
                generation_config=config,
            )

        return AppleFmStream(stream, handle, time.perf_counter(), on_chunk, on_done)

    return patched_stream


def patch_method(cls: type, name: str, make_patch, *extra) -> None:  # type: ignore[no-untyped-def]
    original = getattr(cls, name)
    if getattr(original, "__wildedge_patch_name__", None) == APPLE_FM_PATCH_NAME:
        return
    patched = make_patch(original, *extra)
    patched.__wildedge_patch_name__ = APPLE_FM_PATCH_NAME
    patched.__wildedge_original_call__ = original
    setattr(cls, name, patched)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


def is_apple_fm_object(obj: object) -> bool:
    return any(
        cls.__module__.partition(".")[0] == APPLE_FM_PACKAGE
        and cls.__name__ in ("SystemLanguageModel", "LanguageModelSession")
        for cls in type(obj).__mro__
    )


class AppleFmExtractor(BaseExtractor):
    def can_handle(self, obj: object) -> bool:
        return is_apple_fm_object(obj)

    def extract_info(
        self, obj: object, overrides: dict
    ) -> tuple[str | None, ModelInfo]:
        model_id = overrides.pop("id", None) or model_id_for(
            getattr(obj, APPLE_FM_USE_CASE_ATTR, None)
        )
        info = ModelInfo(
            model_name=model_id,
            model_version=overrides.pop("version", None) or os_version(),
            model_source=overrides.pop("source", MODEL_SOURCE),
            model_format=overrides.pop("format", MODEL_FORMAT),
            model_family=overrides.pop("family", MODEL_ID),
            quantization=overrides.pop("quantization", None),
        )
        return model_id, info

    def install_hooks(self, obj: object, handle: ModelHandle) -> None:
        # The system model runs on the Neural Engine. While it generates,
        # Neural Engine power rises from 0 to about 5 W on an M2, and GPU
        # power stays near idle.
        handle.detected_accelerator = accelerators.NPU
        if type(obj).__name__ == "LanguageModelSession":
            setattr(obj, APPLE_FM_HANDLE_ATTR, handle)

    @classmethod
    def install_auto_load_patch(cls, client_ref: Any) -> None:
        """Patch apple_fm_sdk so every new session is tracked.

        - ``SystemLanguageModel.__init__`` keeps the use case, which names the
          model.
        - ``SystemLanguageModel.is_available()`` records a load event when
          its answer changes, failed with the reason the model is unavailable.
        - ``LanguageModelSession.__init__`` registers the session's model.
        - ``respond()`` and ``stream_response()`` record inference events.
        """
        global _apple_fm_patched, _fm, _count_tokens
        if _apple_fm_patched:
            return
        _fm = _fm or optional_module(APPLE_FM_PACKAGE)
        if _fm is None:
            return

        with _APPLE_FM_PATCH_LOCK:
            if _apple_fm_patched:
                return
            model_cls = _fm.SystemLanguageModel
            session_cls = _fm.LanguageModelSession
            patch_method(model_cls, "__init__", build_model_init_patch)
            patch_method(
                model_cls, "is_available", build_availability_patch, client_ref
            )
            patch_method(session_cls, "__init__", build_session_init_patch, client_ref)
            patch_method(session_cls, "respond", build_respond_patch)
            patch_method(session_cls, "stream_response", build_stream_patch)
            _count_tokens = parse_bool(
                os.environ.get(constants.ENV_APPLE_FM_COUNT_TOKENS)
            )
            _apple_fm_patched = True


# Entry point named by the integration registry.
install_auto_load_patch = AppleFmExtractor.install_auto_load_patch
