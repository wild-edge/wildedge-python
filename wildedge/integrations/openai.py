"""OpenAI / OpenRouter integration."""

from __future__ import annotations

import functools
import threading
import time
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from wildedge import constants
from wildedge.events.inference import ApiMeta, GenerationOutputMeta, TextInputMeta
from wildedge.integrations.base import BaseExtractor
from wildedge.integrations.common import debug_failure
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


def source_from_base_url(base_url: str | None) -> str:
    if not base_url:
        return "openai"
    s = base_url.lower()
    if "openrouter" in s:
        return "openrouter"
    if "openai.com" in s:
        return "openai"
    try:
        return urlparse(s).hostname or s
    except Exception:
        return s


def build_input_meta(messages: list, tokens_in: int | None) -> TextInputMeta | None:
    if not messages:
        return None
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if not last_user:
        return None
    content = last_user.get("content", "")
    if not isinstance(content, str) or not content:
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

        if model_id not in model_handles:
            try:
                model_handles[model_id] = c.register_model(
                    completions, model_id=model_id, source=source
                )
            except Exception as exc:
                debug_openai_failure("model registration", exc)

        handle = model_handles.get(model_id)
        t0 = time.perf_counter()
        try:
            result = original_create(*args, **kwargs)
            if is_streaming or handle is None:
                return result
            duration = elapsed_ms(t0)
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
            return result
        except Exception as exc:
            if handle is not None:
                handle.track_error(
                    error_code="UNKNOWN",
                    error_message=str(exc)[: constants.ERROR_MSG_MAX_LEN],
                )
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

        if model_id not in model_handles:
            try:
                model_handles[model_id] = c.register_model(
                    completions, model_id=model_id, source=source
                )
            except Exception as exc:
                debug_openai_failure("model registration", exc)

        handle = model_handles.get(model_id)
        t0 = time.perf_counter()
        try:
            result = await original_create(*args, **kwargs)
            if is_streaming or handle is None:
                return result
            duration = elapsed_ms(t0)
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
            return result
        except Exception as exc:
            if handle is not None:
                handle.track_error(
                    error_code="UNKNOWN",
                    error_message=str(exc)[: constants.ERROR_MSG_MAX_LEN],
                )
            raise

    completions.create = patched_create  # type: ignore[attr-defined]


class OpenAIExtractor(BaseExtractor):
    def can_handle(self, obj: object) -> bool:
        return type(obj).__name__ in (
            "OpenAI",
            "AsyncOpenAI",
            "Completions",
            "AsyncCompletions",
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
        """Patch openai.OpenAI and openai.AsyncOpenAI to wrap chat.completions.create."""
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
                    source = source_from_base_url(
                        str(getattr(self_inner, "base_url", None) or "")
                    )
                    try:
                        wrap_sync_completions(
                            self_inner.chat.completions, source, client_ref
                        )
                    except Exception as exc:
                        debug_openai_failure("sync client wrap", exc)

            def patched_async_init(self_inner, *args, **kwargs):  # type: ignore[no-untyped-def]
                original_async_init(self_inner, *args, **kwargs)
                c = client_ref()  # type: ignore[call-arg]
                if c is not None and not c.closed:
                    source = source_from_base_url(
                        str(getattr(self_inner, "base_url", None) or "")
                    )
                    try:
                        wrap_async_completions(
                            self_inner.chat.completions, source, client_ref
                        )
                    except Exception as exc:
                        debug_openai_failure("async client wrap", exc)

            patched_sync_init.__wildedge_patch_name__ = OPENAI_INIT_PATCH_NAME  # type: ignore[attr-defined]
            patched_sync_init.__wildedge_original_call__ = original_sync_init  # type: ignore[attr-defined]
            patched_async_init.__wildedge_patch_name__ = OPENAI_INIT_PATCH_NAME  # type: ignore[attr-defined]
            patched_async_init.__wildedge_original_call__ = original_async_init  # type: ignore[attr-defined]

            _openai.OpenAI.__init__ = patched_sync_init
            _openai.AsyncOpenAI.__init__ = patched_async_init
            _openai_patched = True
