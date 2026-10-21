"""OpenAI / OpenRouter integration."""

from __future__ import annotations

import base64
import functools
import threading
import time
from typing import TYPE_CHECKING

from wildedge.events.inference import (
    ApiMeta,
    EmbeddingOutputMeta,
    GenerationOutputMeta,
    TextInputMeta,
)
from wildedge.instrumentation import bind_client_reference
from wildedge.integrations.base import BaseExtractor
from wildedge.integrations.common import (
    SOURCE_BY_HOSTNAME,  # noqa: F401  (re-exported; moved to common)
    AsyncStreamWrapper,
    SyncStreamWrapper,
    build_input_meta,
    debug_failure,
    source_from_base_url,
    stream_error_code,
    track_api_error,
)
from wildedge.model import ModelInfo
from wildedge.timing import elapsed_ms

try:
    import openai as _openai
except ImportError:
    _openai = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from wildedge.model import ModelHandle

_openai_patched = False
_OPENAI_PATCH_LOCK = threading.Lock()
OPENAI_INIT_PATCH_NAME = "openai_auto_load"

debug_openai_failure = functools.partial(debug_failure, "openai")


def build_streaming_output_meta(
    ttft_ms: int | None,
    tokens_in: int | None,
    tokens_out: int | None,
    stop_reason: str | None,
    duration_ms: int,
) -> GenerationOutputMeta:
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


def build_output_meta(
    response: object, duration_ms: int
) -> GenerationOutputMeta | None:
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        tokens_in = getattr(usage, "prompt_tokens", None)
        tokens_out = getattr(usage, "completion_tokens", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached_input_tokens = getattr(prompt_details, "cached_tokens", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens_out = getattr(completion_details, "reasoning_tokens", None)
        choices = getattr(response, "choices", None) or []
        stop_reason = getattr(choices[0], "finish_reason", None) if choices else None
        tps = (
            round(tokens_out / duration_ms * 1000, 1)
            if duration_ms > 0 and tokens_out
            else None
        )
        return GenerationOutputMeta(
            task="generation",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cached_input_tokens=cached_input_tokens,
            reasoning_tokens_out=reasoning_tokens_out,
            tokens_per_second=tps,
            stop_reason=stop_reason,
        )
    except Exception as exc:
        debug_openai_failure("output meta extraction", exc)
        return None


def build_api_meta(response: object) -> ApiMeta | None:
    try:
        resolved_model_id = getattr(response, "model", None)
        system_fingerprint = getattr(response, "system_fingerprint", None)
        service_tier = getattr(response, "service_tier", None)
        if not any([resolved_model_id, system_fingerprint, service_tier]):
            return None
        return ApiMeta(
            resolved_model_id=resolved_model_id,
            system_fingerprint=system_fingerprint,
            service_tier=service_tier,
        )
    except Exception as exc:
        debug_openai_failure("api meta extraction", exc)
        return None


def resolve_handle(
    model_id: str,
    completions: object,
    model_handles: dict[str, ModelHandle],
    client: object,
    source: str,
) -> ModelHandle | None:
    if model_id not in model_handles:
        try:
            model_handles[model_id] = client.register_model(  # type: ignore[attr-defined]
                completions, model_id=model_id, source=source
            )
        except Exception as exc:
            debug_openai_failure("model registration", exc)
    return model_handles.get(model_id)


def record_inference(
    handle: ModelHandle,
    result: object,
    messages: list,
    duration: int,
) -> None:
    usage = getattr(result, "usage", None)
    tokens_in = getattr(usage, "prompt_tokens", None) if usage else None
    handle.track_inference(
        duration_ms=duration,
        input_modality="text",
        output_modality="generation",
        success=True,
        input_meta=build_input_meta(messages, tokens_in),
        output_meta=build_output_meta(result, duration),
        api_meta=build_api_meta(result),
    )


def make_openai_stream_callbacks(
    handle: ModelHandle,
    messages: list,
) -> tuple:
    """Return (on_chunk, on_done) callbacks for an OpenAI streaming response.

    on_chunk updates mutable state from each ChatCompletionChunk.
    on_done is called with (duration_ms, ttft_ms) when the stream is exhausted.
    """
    tokens_in: list[int | None] = [None]
    tokens_out: list[int | None] = [None]
    stop_reason: list[str | None] = [None]
    last_chunk: list[object] = [None]

    def on_chunk(chunk: object) -> None:
        last_chunk[0] = chunk
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            tokens_in[0] = getattr(chunk_usage, "prompt_tokens", None)
            tokens_out[0] = getattr(chunk_usage, "completion_tokens", None)
        choices = getattr(chunk, "choices", None) or []
        if choices:
            reason = getattr(choices[0], "finish_reason", None)
            if reason:
                stop_reason[0] = reason

    def on_done(
        duration_ms: int, ttft_ms: int | None, *, outcome: str = "completed"
    ) -> None:
        ti, to, sr = tokens_in[0], tokens_out[0], stop_reason[0]
        if outcome == "completed" and sr is None:
            outcome = "incomplete"
        output_meta = build_streaming_output_meta(ttft_ms, ti, to, sr, duration_ms)
        handle.track_inference(
            duration_ms=duration_ms,
            input_modality="text",
            output_modality="generation",
            success=outcome == "completed",
            error_code=stream_error_code(outcome),
            input_meta=build_input_meta(messages, ti),
            output_meta=output_meta,
            api_meta=build_api_meta(last_chunk[0])
            if last_chunk[0] is not None
            else None,
        )

    return on_chunk, on_done


def wrap_sync_completions(completions: object, source: str, client_ref: object) -> None:
    original_create = completions.create  # type: ignore[attr-defined]
    model_handles: dict[str, ModelHandle] = {}

    def patched_create(*args, **kwargs):
        model_id: str | None = kwargs.get("model") or (args[0] if args else None)
        messages: list = kwargs.get("messages", [])
        is_streaming: bool = bool(kwargs.get("stream", False))
        c = client_ref()  # type: ignore[call-arg]
        if c is None or c.closed or not model_id:
            return original_create(*args, **kwargs)
        handle = resolve_handle(model_id, completions, model_handles, c, source)
        t0 = time.perf_counter()
        try:
            result = original_create(*args, **kwargs)
            if handle is not None:
                if is_streaming:
                    on_chunk, on_done = make_openai_stream_callbacks(handle, messages)
                    return SyncStreamWrapper(result, handle, t0, on_chunk, on_done)
                record_inference(handle, result, messages, elapsed_ms(t0))
            return result
        except Exception as exc:
            if handle is not None:
                track_api_error(handle, exc, elapsed_ms(t0))
            raise

    completions.create = patched_create  # type: ignore[attr-defined]


def wrap_async_completions(
    completions: object, source: str, client_ref: object
) -> None:
    original_create = completions.create  # type: ignore[attr-defined]
    model_handles: dict[str, ModelHandle] = {}

    async def patched_create(*args, **kwargs):
        model_id: str | None = kwargs.get("model") or (args[0] if args else None)
        messages: list = kwargs.get("messages", [])
        is_streaming: bool = bool(kwargs.get("stream", False))
        c = client_ref()  # type: ignore[call-arg]
        if c is None or c.closed or not model_id:
            return await original_create(*args, **kwargs)
        handle = resolve_handle(model_id, completions, model_handles, c, source)
        t0 = time.perf_counter()
        try:
            result = await original_create(*args, **kwargs)
            if handle is not None:
                if is_streaming:
                    on_chunk, on_done = make_openai_stream_callbacks(handle, messages)
                    return AsyncStreamWrapper(result, handle, t0, on_chunk, on_done)
                record_inference(handle, result, messages, elapsed_ms(t0))
            return result
        except Exception as exc:
            if handle is not None:
                track_api_error(handle, exc, elapsed_ms(t0))
            raise

    completions.create = patched_create  # type: ignore[attr-defined]


def responses_messages(value: object) -> list:
    """Normalize text inputs without consuming arbitrary input iterables."""
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    return value if isinstance(value, list) else []


def build_responses_output_meta(
    response: object, duration_ms: int
) -> GenerationOutputMeta:
    usage = getattr(response, "usage", None)
    tokens_out = getattr(usage, "output_tokens", None)
    status = getattr(response, "status", None)
    reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
    return GenerationOutputMeta(
        tokens_in=getattr(usage, "input_tokens", None),
        tokens_out=tokens_out,
        cached_input_tokens=getattr(
            getattr(usage, "input_tokens_details", None), "cached_tokens", None
        ),
        reasoning_tokens_out=getattr(
            getattr(usage, "output_tokens_details", None), "reasoning_tokens", None
        ),
        tokens_per_second=round(tokens_out / duration_ms * 1000, 1)
        if duration_ms > 0 and tokens_out
        else None,
        stop_reason=reason or status,
    )


def record_response(
    handle: ModelHandle,
    response: object,
    messages: list,
    duration: int,
    ttft_ms: int | None = None,
) -> None:
    if getattr(response, "status", None) in ("queued", "in_progress"):
        return
    output = build_responses_output_meta(response, duration)
    output.time_to_first_token_ms = ttft_ms
    error = getattr(response, "error", None)
    handle.track_inference(
        duration_ms=duration,
        input_modality="text",
        output_modality="generation",
        success=getattr(response, "status", None) != "failed" and error is None,
        error_code=getattr(error, "code", None),
        input_meta=build_input_meta(messages, output.tokens_in),
        output_meta=output,
        api_meta=build_api_meta(response),
    )


def make_responses_stream_callbacks(
    handle: ModelHandle, messages: list, t0: float
) -> tuple:
    response = None
    ttft = None
    recorded = False

    def on_chunk(chunk: object) -> None:
        nonlocal response, ttft, recorded
        event_type = getattr(chunk, "type", "")
        if (
            event_type
            in (
                "response.output_text.delta",
                "response.refusal.delta",
                "response.function_call_arguments.delta",
                "response.output_audio.delta",
            )
            and getattr(chunk, "delta", None)
            and ttft is None
        ):
            ttft = elapsed_ms(t0)
        if event_type in (
            "response.completed",
            "response.incomplete",
            "response.failed",
        ):
            response = getattr(chunk, "response", None)
        elif event_type == "error" and not recorded:
            handle.track_error(
                error_code=getattr(chunk, "code", None) or "PROVIDER_ERROR",
                duration_ms=elapsed_ms(t0),
            )
            recorded = True

    def on_done(
        duration_ms: int, _first_event_ms: int | None, *, outcome: str = "completed"
    ) -> None:
        nonlocal recorded
        if response is None and not recorded:
            recorded = True
            handle.track_inference(
                duration_ms=duration_ms,
                input_modality="text",
                output_modality="generation",
                input_meta=build_input_meta(messages, None),
                output_meta=GenerationOutputMeta(
                    time_to_first_token_ms=ttft,
                    stop_reason=outcome if outcome != "completed" else "incomplete",
                ),
                success=False,
                error_code="STREAM_INCOMPLETE"
                if outcome == "completed"
                else stream_error_code(outcome),
            )
        if response is not None and not recorded:
            recorded = True
            record_response(handle, response, messages, duration_ms, ttft)

    return on_chunk, on_done


def wrap_sync_responses(responses: object, source: str, client_ref: object) -> None:
    original_create = responses.create
    model_handles: dict[str, ModelHandle] = {}

    @functools.wraps(original_create)
    def patched_create(*args, **kwargs):
        client = client_ref()
        model_id = kwargs.get("model")
        if client is None or client.closed or not model_id:
            return original_create(*args, **kwargs)
        handle = resolve_handle(model_id, responses, model_handles, client, source)
        t0 = time.perf_counter()
        try:
            result = original_create(*args, **kwargs)
        except Exception as exc:
            if handle is not None:
                track_api_error(handle, exc, elapsed_ms(t0))
            raise
        if handle is not None:
            try:
                messages = responses_messages(kwargs.get("input"))
                if kwargs.get("stream"):
                    callbacks = make_responses_stream_callbacks(handle, messages, t0)
                    return SyncStreamWrapper(result, handle, t0, *callbacks)
                record_response(handle, result, messages, elapsed_ms(t0))
            except Exception as exc:
                debug_openai_failure("responses tracking", exc)
        return result

    responses.create = patched_create


def wrap_async_responses(responses: object, source: str, client_ref: object) -> None:
    original_create = responses.create
    model_handles: dict[str, ModelHandle] = {}

    @functools.wraps(original_create)
    async def patched_create(*args, **kwargs):
        client = client_ref()
        model_id = kwargs.get("model")
        if client is None or client.closed or not model_id:
            return await original_create(*args, **kwargs)
        handle = resolve_handle(model_id, responses, model_handles, client, source)
        t0 = time.perf_counter()
        try:
            result = await original_create(*args, **kwargs)
        except Exception as exc:
            if handle is not None:
                track_api_error(handle, exc, elapsed_ms(t0))
            raise
        if handle is not None:
            try:
                messages = responses_messages(kwargs.get("input"))
                if kwargs.get("stream"):
                    callbacks = make_responses_stream_callbacks(handle, messages, t0)
                    return AsyncStreamWrapper(result, handle, t0, *callbacks)
                record_response(handle, result, messages, elapsed_ms(t0))
            except Exception as exc:
                debug_openai_failure("responses tracking", exc)
        return result

    responses.create = patched_create


def record_embedding(
    handle: ModelHandle, result: object, value: object, duration: int
) -> None:
    """Record metadata only; never retain input text or embedding vectors."""
    usage = getattr(result, "usage", None)
    tokens = getattr(usage, "prompt_tokens", None)
    batch_size = None
    chars = words = None
    if isinstance(value, str):
        batch_size = 1
        chars, words = len(value), len(value.split())
    elif isinstance(value, (list, tuple)):
        if value and isinstance(value[0], int):
            batch_size = 1
        else:
            batch_size = len(value)
            if value and all(isinstance(item, str) for item in value):
                chars = sum(len(item) for item in value)
                words = sum(len(item.split()) for item in value)
    data = getattr(result, "data", None)
    dimensions = None
    if data:
        vector = getattr(data[0], "embedding", None)
        if isinstance(vector, (list, tuple)):
            dimensions = len(vector)
        elif isinstance(vector, str):
            byte_count = len(base64.b64decode(vector, validate=True))
            if byte_count % 4 == 0:
                dimensions = byte_count // 4
    handle.track_inference(
        duration_ms=duration,
        input_modality="text",
        output_modality="embedding",
        success=True,
        batch_size=batch_size,
        input_meta=TextInputMeta(
            char_count=chars, word_count=words, token_count=tokens
        ),
        output_meta=EmbeddingOutputMeta(dimensions=dimensions, tokens_in=tokens),
        api_meta=build_api_meta(result),
    )


def wrap_sync_embeddings(embeddings: object, source: str, client_ref: object) -> None:
    original_create = embeddings.create
    model_handles: dict[str, ModelHandle] = {}

    @functools.wraps(original_create)
    def patched_create(*args, **kwargs):
        client = client_ref()
        model_id = kwargs.get("model")
        if client is None or client.closed or not model_id:
            return original_create(*args, **kwargs)
        handle = resolve_handle(model_id, embeddings, model_handles, client, source)
        t0 = time.perf_counter()
        try:
            result = original_create(*args, **kwargs)
        except Exception as exc:
            if handle is not None:
                track_api_error(handle, exc, elapsed_ms(t0))
            raise
        if handle is not None:
            try:
                record_embedding(handle, result, kwargs.get("input"), elapsed_ms(t0))
            except Exception as exc:
                debug_openai_failure("embedding tracking", exc)
        return result

    embeddings.create = patched_create


def wrap_async_embeddings(embeddings: object, source: str, client_ref: object) -> None:
    original_create = embeddings.create
    model_handles: dict[str, ModelHandle] = {}

    @functools.wraps(original_create)
    async def patched_create(*args, **kwargs):
        client = client_ref()
        model_id = kwargs.get("model")
        if client is None or client.closed or not model_id:
            return await original_create(*args, **kwargs)
        handle = resolve_handle(model_id, embeddings, model_handles, client, source)
        t0 = time.perf_counter()
        try:
            result = await original_create(*args, **kwargs)
        except Exception as exc:
            if handle is not None:
                track_api_error(handle, exc, elapsed_ms(t0))
            raise
        if handle is not None:
            try:
                record_embedding(handle, result, kwargs.get("input"), elapsed_ms(t0))
            except Exception as exc:
                debug_openai_failure("embedding tracking", exc)
        return result

    embeddings.create = patched_create


class OpenAIExtractor(BaseExtractor):
    def can_handle(self, obj: object) -> bool:
        return type(obj).__name__ in (
            "OpenAI",
            "AsyncOpenAI",
            "Completions",
            "AsyncCompletions",
            "Responses",
            "AsyncResponses",
            "Embeddings",
            "AsyncEmbeddings",
        )

    def extract_info(
        self, obj: object, overrides: dict
    ) -> tuple[str | None, ModelInfo]:
        model_id = overrides.pop("id", None)
        source = overrides.pop("source", None) or source_from_base_url(
            str(getattr(obj, "base_url", None) or "")
        )
        info = ModelInfo(
            model_name=model_id or "openai-model",
            model_version=overrides.pop("version", "unknown"),
            model_source=source,
            model_format="api",
            model_family=overrides.pop("family", None),
            quantization=None,
        )
        return model_id, info

    def install_hooks(self, obj: object, handle: ModelHandle) -> None:
        pass

    @classmethod
    def install_auto_load_patch(cls, client_ref: object) -> None:
        """Patch new OpenAI clients to track Chat Completions, Responses, and embeddings."""
        global _openai_patched
        if _openai_patched or _openai is None:
            return

        with _OPENAI_PATCH_LOCK:
            if _openai_patched:
                return

            original_sync_init = _openai.OpenAI.__init__
            original_async_init = _openai.AsyncOpenAI.__init__

            if (
                getattr(original_sync_init, "__wildedge_patch_name__", None)
                == OPENAI_INIT_PATCH_NAME
            ):
                _openai_patched = True
                return

            def patched_sync_init(self_inner, *args, **kwargs):  # type: ignore[no-untyped-def]
                original_sync_init(self_inner, *args, **kwargs)
                c = client_ref()  # type: ignore[call-arg]
                if c is not None and not c.closed:
                    bound_ref = bind_client_reference(client_ref, c)
                    source = source_from_base_url(
                        str(getattr(self_inner, "base_url", None) or "")
                    )
                    try:
                        wrap_sync_completions(
                            self_inner.chat.completions, source, bound_ref
                        )
                    except Exception as exc:
                        debug_openai_failure("sync client wrap", exc)
                    if hasattr(self_inner, "responses"):
                        try:
                            wrap_sync_responses(self_inner.responses, source, bound_ref)
                        except Exception as exc:
                            debug_openai_failure("sync responses wrap", exc)
                    if hasattr(self_inner, "embeddings"):
                        try:
                            wrap_sync_embeddings(
                                self_inner.embeddings, source, bound_ref
                            )
                        except Exception as exc:
                            debug_openai_failure("sync embeddings wrap", exc)

            def patched_async_init(self_inner, *args, **kwargs):  # type: ignore[no-untyped-def]
                original_async_init(self_inner, *args, **kwargs)
                c = client_ref()  # type: ignore[call-arg]
                if c is not None and not c.closed:
                    bound_ref = bind_client_reference(client_ref, c)
                    source = source_from_base_url(
                        str(getattr(self_inner, "base_url", None) or "")
                    )
                    try:
                        wrap_async_completions(
                            self_inner.chat.completions, source, bound_ref
                        )
                    except Exception as exc:
                        debug_openai_failure("async client wrap", exc)
                    if hasattr(self_inner, "responses"):
                        try:
                            wrap_async_responses(
                                self_inner.responses, source, bound_ref
                            )
                        except Exception as exc:
                            debug_openai_failure("async responses wrap", exc)
                    if hasattr(self_inner, "embeddings"):
                        try:
                            wrap_async_embeddings(
                                self_inner.embeddings, source, bound_ref
                            )
                        except Exception as exc:
                            debug_openai_failure("async embeddings wrap", exc)

            patched_sync_init.__wildedge_patch_name__ = OPENAI_INIT_PATCH_NAME  # type: ignore[attr-defined]
            patched_sync_init.__wildedge_original_call__ = original_sync_init  # type: ignore[attr-defined]
            patched_async_init.__wildedge_patch_name__ = OPENAI_INIT_PATCH_NAME  # type: ignore[attr-defined]
            patched_async_init.__wildedge_original_call__ = original_async_init  # type: ignore[attr-defined]

            _openai.OpenAI.__init__ = patched_sync_init
            _openai.AsyncOpenAI.__init__ = patched_async_init
            _openai_patched = True
