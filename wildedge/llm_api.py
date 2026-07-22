"""Tracking for LLM calls made over an HTTP API.

For applications that call an LLM API with a plain HTTP client (httpx or
requests against OpenRouter, vLLM, Ollama, or any OpenAI-compatible endpoint)
instead of the openai/anthropic client libraries that auto-instrumentation
patches. For models running inside your own process (llama.cpp, transformers,
MLX), use the framework integrations instead; they capture model metadata
this API boundary cannot see. Times the call, normalizes usage payloads from
either provider shape, and emits the same inference events as the
integrations:

    with wildedge.llm_api(model="openai/gpt-4o-mini", provider="openrouter") as call:
        data = post_chat_completion(...)
        call.response(data)

Correlates with surrounding ``wildedge.trace`` / ``wildedge.span`` blocks
like any other event, and is a silent no-op without a DSN.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from wildedge import constants
from wildedge.events.inference import ApiMeta, GenerationOutputMeta, TextInputMeta
from wildedge.integrations.common import build_input_meta, source_from_base_url
from wildedge.logging import logger
from wildedge.timing import elapsed_ms

if TYPE_CHECKING:
    from wildedge.model import ModelHandle


def _get(obj: object, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _first(obj: object, *names: str) -> Any:
    for name in names:
        value = _get(obj, name)
        if value is not None:
            return value
    return None


class LLMCall:
    """Mutable record of one LLM API call; emits an inference event on exit.

    Fields may be set directly (``call.stop_reason = ...``, ``call.success =
    False``) or through :meth:`usage` / :meth:`response` before the block
    exits. An exception escaping the block records an error event instead,
    with the exception class as the error code.
    """

    def __init__(
        self,
        *,
        model: str,
        provider: str | None = None,
        base_url: str | None = None,
        prompt: str | None = None,
        messages: list | None = None,
    ):
        self.model = model
        self.source = provider or (
            source_from_base_url(base_url) if base_url else "api"
        )
        self.prompt = prompt
        self.messages = messages
        self.success = True
        self.stop_reason: str | None = None
        self.tokens_in: int | None = None
        self.tokens_out: int | None = None
        self.cached_tokens: int | None = None
        self.reasoning_tokens: int | None = None
        self.resolved_model_id: str | None = None
        self.system_fingerprint: str | None = None
        self.service_tier: str | None = None
        self._t0: float | None = None
        self._ttft_ms: int | None = None

    def first_token(self) -> None:
        """Mark time-to-first-token for streaming responses."""
        if self._t0 is not None and self._ttft_ms is None:
            self._ttft_ms = elapsed_ms(self._t0)

    def usage(self, payload: object = None, **fields: int | None) -> LLMCall:
        """Record token usage from a raw usage payload or explicit fields.

        ``payload`` may be a dict or attribute object in OpenAI shape
        (``prompt_tokens``/``completion_tokens`` with nested details) or
        Anthropic shape (``input_tokens``/``output_tokens``,
        ``cache_read_input_tokens``). Explicit keyword fields (``tokens_in``,
        ``tokens_out``, ``cached_tokens``, ``reasoning_tokens``) win over the
        payload.
        """
        if payload is not None:
            self.tokens_in = _first(payload, "prompt_tokens", "input_tokens")
            self.tokens_out = _first(payload, "completion_tokens", "output_tokens")
            cached = _get(_get(payload, "prompt_tokens_details"), "cached_tokens")
            if cached is None:
                cached = _get(payload, "cache_read_input_tokens")
            self.cached_tokens = cached
            self.reasoning_tokens = _get(
                _get(payload, "completion_tokens_details"), "reasoning_tokens"
            )
        for name in ("tokens_in", "tokens_out", "cached_tokens", "reasoning_tokens"):
            if fields.get(name) is not None:
                setattr(self, name, fields[name])
        return self

    def response(self, payload: object) -> LLMCall:
        """Record usage, stop reason and API metadata from a full response.

        Accepts a chat-completion response as a dict or SDK object, in OpenAI
        shape (``usage``, ``choices[0].finish_reason``, ``model``) or
        Anthropic shape (``usage``, top-level ``stop_reason``).
        """
        usage = _get(payload, "usage")
        if usage is not None:
            self.usage(usage)
        choices = _get(payload, "choices") or []
        stop_reason = _get(choices[0], "finish_reason") if choices else None
        if stop_reason is None:
            stop_reason = _get(payload, "stop_reason")
        if stop_reason is not None:
            self.stop_reason = stop_reason
        for field, name in (
            ("resolved_model_id", "model"),
            ("system_fingerprint", "system_fingerprint"),
            ("service_tier", "service_tier"),
        ):
            value = _get(payload, name)
            if value is not None:
                setattr(self, field, value)
        return self

    def __enter__(self) -> LLMCall:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._t0 is None:
            return False
        duration_ms = elapsed_ms(self._t0)
        handle = self._handle()
        if handle is None:
            return False
        if exc_type is not None:
            handle.track_error(
                error_code=exc_type.__name__,
                error_message=str(exc_val)[: constants.ERROR_MSG_MAX_LEN],
            )
            return False
        handle.track_inference(
            duration_ms=duration_ms,
            input_modality="text",
            output_modality="generation",
            success=self.success,
            input_meta=self._input_meta(),
            output_meta=self._output_meta(duration_ms),
            api_meta=self._api_meta(),
        )
        return False

    async def __aenter__(self) -> LLMCall:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        return self.__exit__(exc_type, exc_val, exc_tb)

    def _handle(self) -> ModelHandle | None:
        from wildedge.defaults import get_client  # noqa: PLC0415  (import cycle)

        try:
            return get_client().register_model(
                None, model_id=self.model, source=self.source, model_format="api"
            )
        except Exception as exc:
            logger.debug("wildedge: llm model registration failed: %s", exc)
            return None

    def _input_meta(self) -> TextInputMeta | None:
        if self.messages:
            return build_input_meta(self.messages, self.tokens_in)
        if self.prompt:
            return TextInputMeta(
                char_count=len(self.prompt),
                word_count=len(self.prompt.split()),
                token_count=self.tokens_in,
                prompt_type="chat",
            )
        return None

    def _output_meta(self, duration_ms: int) -> GenerationOutputMeta | None:
        known = (
            self.tokens_in,
            self.tokens_out,
            self.cached_tokens,
            self.reasoning_tokens,
            self.stop_reason,
            self._ttft_ms,
        )
        if all(value is None for value in known):
            return None
        tps = (
            round(self.tokens_out / duration_ms * 1000, 1)
            if duration_ms > 0 and self.tokens_out
            else None
        )
        return GenerationOutputMeta(
            task="generation",
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            cached_input_tokens=self.cached_tokens,
            reasoning_tokens_out=self.reasoning_tokens,
            time_to_first_token_ms=self._ttft_ms,
            tokens_per_second=tps,
            stop_reason=self.stop_reason,
        )

    def _api_meta(self) -> ApiMeta | None:
        if not any(
            [self.resolved_model_id, self.system_fingerprint, self.service_tier]
        ):
            return None
        return ApiMeta(
            resolved_model_id=self.resolved_model_id,
            system_fingerprint=self.system_fingerprint,
            service_tier=self.service_tier,
        )


def llm_api(
    *,
    model: str,
    provider: str | None = None,
    base_url: str | None = None,
    prompt: str | None = None,
    messages: list | None = None,
) -> LLMCall:
    """Track one LLM API call made with any HTTP client.

    ``model`` is the model id the endpoint expects. ``provider`` names the
    event source directly; alternatively pass ``base_url`` to derive it.
    ``prompt`` or ``messages`` (chat format) enable input metadata. Usable as
    a sync or async context manager.
    """
    return LLMCall(
        model=model,
        provider=provider,
        base_url=base_url,
        prompt=prompt,
        messages=messages,
    )
