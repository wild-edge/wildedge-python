"""Anthropic SDK integration."""

from __future__ import annotations

import functools
import threading
import time
from typing import TYPE_CHECKING

from wildedge import constants
from wildedge.events.inference import ApiMeta, GenerationOutputMeta, TextInputMeta
from wildedge.integrations.base import BaseExtractor
from wildedge.integrations.common import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
    debug_failure,
)
from wildedge.model import ModelInfo
from wildedge.timing import elapsed_ms

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from wildedge.model import ModelHandle

_anthropic_patched = False
_ANTHROPIC_PATCH_LOCK = threading.Lock()
ANTHROPIC_INIT_PATCH_NAME = "anthropic_auto_load"

debug_anthropic_failure = functools.partial(debug_failure, "anthropic")


def _extract_text(content: object) -> str:
    """Return text from a string or Anthropic content-block list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            text = (
                block.get("text", "")
                if isinstance(block, dict)
                else getattr(block, "text", "")
            )
            if text:
                parts.append(text)
        return " ".join(parts)
    return ""


def build_input_meta(
    messages: list,
    tokens_in: int | None,
) -> TextInputMeta | None:
    last_user = next(
        (
            m
            for m in reversed(messages)
            if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None))
            == "user"
        ),
        None,
    )
    if not last_user:
        return None
    raw = (
        last_user.get("content", "")
        if isinstance(last_user, dict)
        else getattr(last_user, "content", "")
    )
    content = _extract_text(raw)
    if not content:
        return None
    return TextInputMeta(
        char_count=len(content),
        word_count=len(content.split()),
        token_count=tokens_in,
        prompt_type="chat",
    )


def build_output_meta(
    response: object, duration_ms: int
) -> GenerationOutputMeta | None:
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        tokens_in = getattr(usage, "input_tokens", None)
        tokens_out = getattr(usage, "output_tokens", None)
        cached_input_tokens = getattr(usage, "cache_read_input_tokens", None)
        stop_reason = getattr(response, "stop_reason", None)
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
            tokens_per_second=tps,
            stop_reason=stop_reason,
        )
    except Exception as exc:
        debug_anthropic_failure("output meta extraction", exc)
        return None


def build_api_meta(response: object) -> ApiMeta | None:
    try:
        resolved_model_id = getattr(response, "model", None)
        if not resolved_model_id:
            return None
        return ApiMeta(resolved_model_id=resolved_model_id)
    except Exception as exc:
        debug_anthropic_failure("api meta extraction", exc)
        return None


def resolve_handle(
    model_id: str,
    messages_obj: object,
    model_handles: dict[str, ModelHandle],
    client: object,
) -> ModelHandle | None:
    if model_id not in model_handles:
        try:
            model_handles[model_id] = client.register_model(  # type: ignore[attr-defined]
                messages_obj, model_id=model_id, source="anthropic"
            )
        except Exception as exc:
            debug_anthropic_failure("model registration", exc)
    return model_handles.get(model_id)


def make_anthropic_stream_callbacks(
    handle: ModelHandle,
    messages: list,
) -> tuple:
    """Return (on_chunk, on_done) callbacks for an Anthropic streaming response.

    on_chunk dispatches on event type to accumulate token counts and stop reason.
    on_done is called with (duration_ms, ttft_ms) when the stream is exhausted.
    """
    tokens_in: list[int | None] = [None]
    tokens_out: list[int | None] = [None]
    cached_tokens: list[int | None] = [None]
    stop_reason: list[str | None] = [None]
    resolved_model: list[str | None] = [None]

    def on_chunk(event: object) -> None:
        event_type = type(event).__name__
        if event_type == "RawMessageStartEvent":
            msg = getattr(event, "message", None)
            if msg is not None:
                usage = getattr(msg, "usage", None)
                if usage is not None:
                    tokens_in[0] = getattr(usage, "input_tokens", None)
                    cached_tokens[0] = getattr(usage, "cache_read_input_tokens", None)
                resolved_model[0] = getattr(msg, "model", None)
        elif event_type == "RawMessageDeltaEvent":
            usage = getattr(event, "usage", None)
            if usage is not None:
                tokens_out[0] = getattr(usage, "output_tokens", None)
            delta = getattr(event, "delta", None)
            if delta is not None:
                reason = getattr(delta, "stop_reason", None)
                if reason:
                    stop_reason[0] = reason

    def on_done(duration_ms: int, ttft_ms: int | None) -> None:
        ti, to, cr, sr = tokens_in[0], tokens_out[0], cached_tokens[0], stop_reason[0]
        tps = round(to / duration_ms * 1000, 1) if duration_ms > 0 and to else None
        api_meta = (
            ApiMeta(resolved_model_id=resolved_model[0]) if resolved_model[0] else None
        )
        handle.track_inference(
            duration_ms=duration_ms,
            input_modality="text",
            output_modality="generation",
            success=True,
            input_meta=build_input_meta(messages, ti),
            output_meta=GenerationOutputMeta(
                task="generation",
                tokens_in=ti,
                tokens_out=to,
                cached_input_tokens=cr,
                time_to_first_token_ms=ttft_ms,
                tokens_per_second=tps,
                stop_reason=sr,
            ),
            api_meta=api_meta,
        )

    return on_chunk, on_done


def wrap_sync_messages(messages_obj: object, client_ref: object) -> None:
    original_create = messages_obj.create  # type: ignore[attr-defined]
    model_handles: dict[str, ModelHandle] = {}

    def patched_create(*args, **kwargs):
        model_id: str | None = kwargs.get("model") or (args[0] if args else None)
        messages: list = kwargs.get("messages", [])
        is_streaming: bool = bool(kwargs.get("stream", False))
        c = client_ref()  # type: ignore[call-arg]
        if c is None or c.closed or not model_id:
            return original_create(*args, **kwargs)
        handle = resolve_handle(model_id, messages_obj, model_handles, c)
        t0 = time.perf_counter()
        try:
            result = original_create(*args, **kwargs)
            if handle is not None:
                if is_streaming:
                    on_chunk, on_done = make_anthropic_stream_callbacks(
                        handle, messages
                    )
                    return SyncStreamWrapper(result, handle, t0, on_chunk, on_done)
                else:
                    duration = elapsed_ms(t0)
                    usage = getattr(result, "usage", None)
                    tokens_in = getattr(usage, "input_tokens", None) if usage else None
                    handle.track_inference(
                        duration_ms=duration,
                        input_modality="text",
                        output_modality="generation",
                        success=True,
                        input_meta=build_input_meta(messages, tokens_in),
                        output_meta=build_output_meta(result, duration),
                        api_meta=build_api_meta(result),
                    )
            return result
        except Exception as exc:
            if handle is not None:
                handle.track_error(
                    error_code="UNKNOWN",
                    error_message=str(exc)[: constants.ERROR_MSG_MAX_LEN],
                )
            raise

    messages_obj.create = patched_create  # type: ignore[attr-defined]


def wrap_async_messages(messages_obj: object, client_ref: object) -> None:
    original_create = messages_obj.create  # type: ignore[attr-defined]
    model_handles: dict[str, ModelHandle] = {}

    async def patched_create(*args, **kwargs):
        model_id: str | None = kwargs.get("model") or (args[0] if args else None)
        messages: list = kwargs.get("messages", [])
        is_streaming: bool = bool(kwargs.get("stream", False))
        c = client_ref()  # type: ignore[call-arg]
        if c is None or c.closed or not model_id:
            return await original_create(*args, **kwargs)
        handle = resolve_handle(model_id, messages_obj, model_handles, c)
        t0 = time.perf_counter()
        try:
            result = await original_create(*args, **kwargs)
            if handle is not None:
                if is_streaming:
                    on_chunk, on_done = make_anthropic_stream_callbacks(
                        handle, messages
                    )
                    return AsyncStreamWrapper(result, handle, t0, on_chunk, on_done)
                else:
                    duration = elapsed_ms(t0)
                    usage = getattr(result, "usage", None)
                    tokens_in = getattr(usage, "input_tokens", None) if usage else None
                    handle.track_inference(
                        duration_ms=duration,
                        input_modality="text",
                        output_modality="generation",
                        success=True,
                        input_meta=build_input_meta(messages, tokens_in),
                        output_meta=build_output_meta(result, duration),
                        api_meta=build_api_meta(result),
                    )
            return result
        except Exception as exc:
            if handle is not None:
                handle.track_error(
                    error_code="UNKNOWN",
                    error_message=str(exc)[: constants.ERROR_MSG_MAX_LEN],
                )
            raise

    messages_obj.create = patched_create  # type: ignore[attr-defined]


class AnthropicExtractor(BaseExtractor):
    def can_handle(self, obj: object) -> bool:
        return type(obj).__name__ in (
            "Anthropic",
            "AsyncAnthropic",
            "Messages",
            "AsyncMessages",
        )

    def extract_info(
        self, obj: object, overrides: dict
    ) -> tuple[str | None, ModelInfo]:
        model_id = overrides.pop("id", None)
        info = ModelInfo(
            model_name=model_id or "anthropic-model",
            model_version=overrides.pop("version", "unknown"),
            model_source=overrides.pop("source", "anthropic"),
            model_format="api",
            model_family=overrides.pop("family", None),
            quantization=None,
        )
        return model_id, info

    def install_hooks(self, obj: object, handle: ModelHandle) -> None:
        pass

    @classmethod
    def install_auto_load_patch(cls, client_ref: object) -> None:
        """Patch anthropic.Anthropic and anthropic.AsyncAnthropic to wrap messages.create."""
        global _anthropic_patched
        if _anthropic_patched or _anthropic is None:
            return

        with _ANTHROPIC_PATCH_LOCK:
            if _anthropic_patched:
                return

            original_sync_init = _anthropic.Anthropic.__init__
            original_async_init = _anthropic.AsyncAnthropic.__init__

            if (
                getattr(original_sync_init, "__wildedge_patch_name__", None)
                == ANTHROPIC_INIT_PATCH_NAME
            ):
                _anthropic_patched = True
                return

            def patched_sync_init(self_inner, *args, **kwargs):  # type: ignore[no-untyped-def]
                original_sync_init(self_inner, *args, **kwargs)
                c = client_ref()  # type: ignore[call-arg]
                if c is not None and not c.closed:
                    try:
                        wrap_sync_messages(self_inner.messages, client_ref)
                    except Exception as exc:
                        debug_anthropic_failure("sync client wrap", exc)

            def patched_async_init(self_inner, *args, **kwargs):  # type: ignore[no-untyped-def]
                original_async_init(self_inner, *args, **kwargs)
                c = client_ref()  # type: ignore[call-arg]
                if c is not None and not c.closed:
                    try:
                        wrap_async_messages(self_inner.messages, client_ref)
                    except Exception as exc:
                        debug_anthropic_failure("async client wrap", exc)

            patched_sync_init.__wildedge_patch_name__ = ANTHROPIC_INIT_PATCH_NAME  # type: ignore[attr-defined]
            patched_sync_init.__wildedge_original_call__ = original_sync_init  # type: ignore[attr-defined]
            patched_async_init.__wildedge_patch_name__ = ANTHROPIC_INIT_PATCH_NAME  # type: ignore[attr-defined]
            patched_async_init.__wildedge_original_call__ = original_async_init  # type: ignore[attr-defined]

            _anthropic.Anthropic.__init__ = patched_sync_init
            _anthropic.AsyncAnthropic.__init__ = patched_async_init
            _anthropic_patched = True
