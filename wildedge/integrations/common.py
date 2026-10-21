"""Shared utilities used across framework integrations."""

from __future__ import annotations

import asyncio
import contextvars
import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from wildedge import accelerators, constants, quantizations
from wildedge.events.error import ErrorCode
from wildedge.events.inference import TextInputMeta
from wildedge.logging import logger
from wildedge.platforms import CURRENT_PLATFORM
from wildedge.timing import elapsed_ms

if TYPE_CHECKING:
    from wildedge.model import ModelHandle


def debug_failure(framework: str, context: str, exc: BaseException) -> None:
    logger.debug("wildedge: %s %s failed: %s", framework, context, exc)


# Folder prefix of a repo in the Hugging Face hub cache: models--org--name.
HF_CACHE_REPO_PREFIX = "models--"


def hf_repo_from_cache_path(path: str) -> str | None:
    """``org/name`` for a path inside the Hugging Face hub cache, else None."""
    for part in path.replace("\\", "/").split("/"):
        if part.startswith(HF_CACHE_REPO_PREFIX):
            return part[len(HF_CACHE_REPO_PREFIX) :].replace("--", "/", 1)
    return None


def torch_module_accelerator(module: object) -> str | None:
    """Device type of a torch module's first parameter, such as cpu, cuda,
    mps or xpu. None when the module has no parameters."""
    try:
        first = next(module.parameters())  # type: ignore[attr-defined]
    except StopIteration:
        return None
    device_type = getattr(first.device, "type", None)
    return str(device_type) if device_type else None


def tf_weights_accelerator(model: object) -> str:
    """The platform's GPU when a TensorFlow/Keras model's first weight is on a
    GPU, else CPU. TF names devices like ``/job:localhost/.../device:GPU:0``."""
    weights = getattr(model, "weights", None) or []
    if weights:
        device = str(getattr(weights[0], "device", "") or "").upper()
        if "GPU" in device or "CUDA" in device:
            return CURRENT_PLATFORM.gpu_accelerator_for_offload()
    return accelerators.CPU


def mlx_accelerator() -> str:
    """MLX runs on the Apple GPU unless the app switched its default to CPU."""
    try:
        import mlx.core as mx  # noqa: PLC0415 -- optional runtime

        return accelerators.CPU if mx.default_device() == mx.cpu else accelerators.MPS
    except Exception as exc:
        debug_failure("mlx", "device detection", exc)
        return accelerators.MPS


def mlx_layer_quantization(model: object) -> str | None:
    """Quantization of an MLX model's layers: the bit width of the first
    quantized layer, ``quantized`` when the width is unknown, or None when no
    layer is quantized."""
    for _, layer in model.named_modules():  # type: ignore[attr-defined]
        bits = getattr(layer, "bits", None)
        if isinstance(bits, int):
            return quantizations.int_bits(bits)
        if "quantized" in type(layer).__name__.lower():
            return quantizations.QUANTIZED
    return None


def stream_error_code(outcome: str) -> str | None:
    """None for a stream that completed, else ``STREAM_<OUTCOME>``."""
    return None if outcome == "completed" else "STREAM_" + outcome.upper()


# ---------------------------------------------------------------------------
# Chat API helpers, shared by the openai integration and wildedge.llm_api
# ---------------------------------------------------------------------------

SOURCE_BY_HOSTNAME: dict[str, str] = {
    "api.openai.com": "openai",
    "openrouter.ai": "openrouter",
    "api.anthropic.com": "anthropic",
    "api.mistral.ai": "mistral",
    "api.groq.com": "groq",
    "api.together.xyz": "together",
    "api.deepseek.com": "deepseek",
    "api.x.ai": "xai",
    "generativelanguage.googleapis.com": "google",
    "api.fireworks.ai": "fireworks",
    "api.cerebras.ai": "cerebras",
    "api.perplexity.ai": "perplexity",
    "integrate.api.nvidia.com": "nvidia",
    "router.huggingface.co": "huggingface",
    "inference.baseten.co": "baseten",
}

# Providers whose endpoints live on per-resource subdomains.
SOURCE_BY_HOSTNAME_SUFFIX: list[tuple[str, str]] = [
    (".openai.azure.com", "azure-openai"),
    (".api.baseten.co", "baseten"),
]


def source_from_base_url(base_url: str | None) -> str:
    hostname = urlparse(base_url.lower()).hostname if base_url else ""
    if not hostname:
        return "openai"
    exact = SOURCE_BY_HOSTNAME.get(hostname)
    if exact is not None:
        return exact
    for suffix, source in SOURCE_BY_HOSTNAME_SUFFIX:
        if hostname.endswith(suffix):
            return source
    return hostname


def http_status_of(exc: BaseException) -> int | None:
    """HTTP error status carried by an API client exception, if any.

    openai and anthropic set ``status_code`` on the exception; httpx and
    requests put it on ``exc.response``. A status below 400 is ignored: an
    error sent mid-stream arrives on a 200 response, which says nothing
    about the failure.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) and status >= 400 else None


def provider_error_code_of(exc: BaseException) -> str | None:
    """Error code or type from the provider's error body, such as ``insufficient_quota``.

    openai sets ``exc.code`` and keeps the error object as ``exc.body``;
    anthropic has no ``code`` and wraps it, as in
    ``{"type": "error", "error": {"type": "overloaded_error"}}``.
    OpenRouter sends an HTTP-like int as the code.
    """
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else None
    if not isinstance(error, dict):
        error = {}
    code = getattr(exc, "code", None) or error.get("code") or error.get("type")
    if code is None:
        return None
    return str(code)[: constants.PROVIDER_ERROR_CODE_MAX_LEN]


def no_response_error_code(exc: BaseException) -> ErrorCode | None:
    """Timeout or connection failure, recognised by exception class name.

    Covers the openai, anthropic, httpx and requests clients without
    importing them.
    """
    names = [cls.__name__ for cls in type(exc).__mro__]
    # Timeout first: openai's APITimeoutError subclasses APIConnectionError.
    if any("Timeout" in name for name in names):
        return ErrorCode.INFERENCE_TIMEOUT
    if any("Connect" in name for name in names):
        return ErrorCode.CONNECTION_ERROR
    return None


def track_api_error(
    handle: ModelHandle,
    exc: BaseException,
    duration_ms: int,
    fallback_code: str | ErrorCode = ErrorCode.UNKNOWN,
) -> None:
    """Record a failed model API call, which took ``duration_ms``, as an error event.

    When the provider responded, the event carries its HTTP status and error
    code with ``UNKNOWN``, and the server derives the error code from them.
    Otherwise the SDK classifies the failure, or sends ``fallback_code``.
    """
    http_status = http_status_of(exc)
    provider_error_code = provider_error_code_of(exc)
    if http_status is not None or provider_error_code is not None:
        error_code = ErrorCode.UNKNOWN
    else:
        error_code = no_response_error_code(exc) or fallback_code
    handle.track_error(
        error_code=error_code,
        error_message=str(exc)[: constants.ERROR_MSG_MAX_LEN],
        http_status=http_status,
        provider_error_code=provider_error_code,
        duration_ms=duration_ms,
    )


def _msg_role(m) -> str | None:
    return m.get("role") if isinstance(m, dict) else getattr(m, "role", None)


def _msg_content(m) -> str | None:
    return m.get("content") if isinstance(m, dict) else getattr(m, "content", None)


def build_input_meta(messages: list, tokens_in: int | None) -> TextInputMeta | None:
    if not messages:
        return None
    last_user = next((m for m in reversed(messages) if _msg_role(m) == "user"), None)
    if not last_user:
        return None
    content = _msg_content(last_user) or ""
    if not isinstance(content, str) or not content:
        return None
    return TextInputMeta(
        char_count=len(content),
        word_count=len(content.split()),
        token_count=tokens_in,
        prompt_type="chat",
    )


# Substrings of framework dtype names (torch.float16, tensor(float16),
# mlx.core.bfloat16, torch.qint8). Checked in order, so bfloat16 must come
# before float16, which it contains.
DTYPE_QUANTIZATION_MAP: dict[str, str] = {
    "bfloat16": quantizations.BF16,
    "float16": quantizations.F16,
    "float32": quantizations.F32,
    "int8": quantizations.INT8,
    "qint": quantizations.INT8,
    "quint": quantizations.INT8,
    "uint8": quantizations.INT8,
}


def dtype_to_quantization(dtype_str: str) -> str | None:
    """Map a dtype string to a quantization label, or None if unrecognised."""
    for needle, label in DTYPE_QUANTIZATION_MAP.items():
        if needle in dtype_str:
            return label
    return None


def image_brightness_histogram(norm_flat: Any) -> tuple[float, float, list[int]]:
    """Brightness stats from a flat normalised (0-1) array (numpy or torch).

    Returns (mean, stddev, buckets) for ranges [0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1].
    """
    brightness_mean = round(float(norm_flat.mean()), 4)
    brightness_stddev = round(float(norm_flat.std()), 4)

    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    buckets = [
        int(((norm_flat >= lo) & (norm_flat < hi)).sum())
        for lo, hi in itertools.pairwise(edges)
    ]
    buckets[-1] += int((norm_flat == 1.0).sum())

    return brightness_mean, brightness_stddev, buckets


# ---------------------------------------------------------------------------
# Modality detection helpers
# ---------------------------------------------------------------------------

# Well-known input tensor names used by HuggingFace / ONNX exported models.
TEXT_INPUT_KEYS: frozenset[str] = frozenset(
    {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
)
AUDIO_INPUT_KEYS: frozenset[str] = frozenset({"input_values", "input_features"})

# Keras/TF layer class names that signal input domain.
CONV_LAYER_TYPES: frozenset[str] = frozenset(
    {
        "Conv1D",
        "Conv2D",
        "Conv3D",
        "DepthwiseConv2D",
        "SeparableConv2D",
        "SeparableConv1D",
        "Conv2DTranspose",
        "Conv3DTranspose",
    }
)
SEQUENCE_LAYER_TYPES: frozenset[str] = frozenset(
    {"Embedding", "LSTM", "GRU", "SimpleRNN", "Bidirectional", "TextVectorization"}
)


def infer_input_modality_from_shape(shape: tuple) -> str | None:
    """Return "image" for 4D tensors (NCHW or NHWC), else None."""
    if len(shape) == 4:
        return "image"
    return None


def infer_input_modality_from_names(names: list[str]) -> str | None:
    """Detect text or audio modality from named tensor/feed keys."""
    if any(n in TEXT_INPUT_KEYS for n in names):
        return "text"
    if any(n in AUDIO_INPUT_KEYS for n in names):
        return "audio"
    return None


def infer_input_modality_from_layer_types(class_names: list[str]) -> str | None:
    """Infer input modality from Keras/TF layer class names.

    Conv layers map to image; sequence/embedding layers map to text.
    """
    names_set = set(class_names)
    if names_set & CONV_LAYER_TYPES:
        return "image"
    if names_set & SEQUENCE_LAYER_TYPES:
        return "text"
    return None


def num_classes_from_output_shape(shape: tuple) -> int:
    """Return the number of output classes from a shape like (batch, N), or 0."""
    if len(shape) >= 2 and isinstance(shape[-1], int) and shape[-1] > 1:
        return int(shape[-1])
    return 0


# ---------------------------------------------------------------------------
# Generic streaming wrappers
# ---------------------------------------------------------------------------
# Each integration provides:
#   on_chunk(chunk) -> None        : update mutable state from a single chunk
#   on_done(duration_ms, ttft_ms, *, outcome="completed") : record one outcome
#
# The wrappers handle TTFT capture, error tracking, context-manager delegation,
# and attribute proxying so callers get a drop-in replacement for the raw stream.


def stream_chunk_has_output(chunk: object) -> bool:
    """Ignore role, usage, and lifecycle chunks when measuring first output."""

    def field(value: object, name: str) -> Any:
        return (
            value.get(name) if isinstance(value, dict) else getattr(value, name, None)
        )

    choices = field(chunk, "choices")
    if choices is not None:
        for choice in choices:
            delta = field(choice, "delta")
            if field(choice, "text") or any(
                field(delta, key)
                for key in ("content", "refusal", "tool_calls", "function_call")
            ):
                return True
        return False
    event_type = field(chunk, "type")
    if event_type is not None:
        if event_type == "content_block_delta":
            delta = field(chunk, "delta")
            return any(
                field(delta, key)
                for key in ("text", "thinking", "partial_json", "data")
            )
        return event_type in (
            "response.output_text.delta",
            "response.refusal.delta",
            "response.function_call_arguments.delta",
            "response.output_audio.delta",
        ) and bool(field(chunk, "delta"))
    return True


class StreamState:
    """One lifecycle per stream, with telemetry isolated from application code."""

    def __init__(
        self,
        handle: ModelHandle,
        t0: float,
        on_chunk: Callable[[object], None] | None,
        on_done: Callable[..., None],
    ) -> None:
        self._handle = handle
        self._t0 = t0
        self._on_chunk = on_chunk
        self._on_done = on_done
        self._finished = False
        self._ttft_ms = None
        self._context = contextvars.copy_context()

    @property
    def finished(self) -> bool:
        return self._finished

    def _safe(self, callback: Callable, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._context.run(callback, *args, **kwargs)
        except Exception as exc:
            debug_failure("stream", "telemetry callback", exc)
            return None

    def _chunk(self, chunk: object) -> None:
        if self._ttft_ms is None and self._safe(stream_chunk_has_output, chunk):
            self._ttft_ms = elapsed_ms(self._t0)
        if self._on_chunk is not None:
            self._safe(self._on_chunk, chunk)

    def _finish(self, outcome: str = "completed") -> None:
        if self._finished:
            return
        self._finished = True
        if outcome == "completed":
            self._safe(self._on_done, elapsed_ms(self._t0), self._ttft_ms)
        else:
            self._safe(
                self._on_done, elapsed_ms(self._t0), self._ttft_ms, outcome=outcome
            )

    def _error(self, exc: Exception) -> None:
        if not self._finished:
            self._finished = True
            self._safe(track_api_error, self._handle, exc, elapsed_ms(self._t0))


class SyncStreamWrapper(StreamState):
    """A single-pass iterator tracking exhaustion, failure, and explicit closure."""

    def __init__(
        self,
        original: Any,
        handle: ModelHandle,
        t0: float,
        on_chunk: Callable[[object], None] | None,
        on_done: Callable[..., None],
    ) -> None:
        super().__init__(handle, t0, on_chunk, on_done)
        self._original = original
        self._iterator = None

    @property
    def started(self) -> bool:
        """Whether iteration has begun."""
        return self._iterator is not None

    def __iter__(self) -> SyncStreamWrapper:
        return self

    def __next__(self) -> Any:
        if self._finished:
            raise StopIteration
        try:
            if self._iterator is None:
                self._iterator = iter(self._original)
            chunk = next(self._iterator)
        except StopIteration:
            self._finish()
            raise
        except Exception as exc:
            self._error(exc)
            raise
        self._chunk(chunk)
        return chunk

    def close(self) -> Any:
        try:
            close = getattr(self._original, "close", None)
            if close is not None:
                return close()
            return None
        finally:
            self._finish("closed")

    def __enter__(self) -> SyncStreamWrapper:
        if hasattr(self._original, "__enter__"):
            self._original.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        try:
            if hasattr(self._original, "__exit__"):
                return self._original.__exit__(*args)
            close = getattr(self._original, "close", None)
            if close is not None:
                close()
            return None
        finally:
            self._finish("closed")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class AsyncStreamWrapper(StreamState):
    """Async counterpart, including cancellation while waiting for a chunk."""

    def __init__(
        self,
        original: Any,
        handle: ModelHandle,
        t0: float,
        on_chunk: Callable[[object], None] | None,
        on_done: Callable[..., None],
    ) -> None:
        super().__init__(handle, t0, on_chunk, on_done)
        self._original = original
        self._iterator = None

    def __aiter__(self) -> AsyncStreamWrapper:
        return self

    async def __anext__(self) -> Any:
        if self._finished:
            raise StopAsyncIteration
        try:
            if self._iterator is None:
                self._iterator = self._original.__aiter__()
            chunk = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._finish()
            raise
        except asyncio.CancelledError:
            self._finish("cancelled")
            raise
        except Exception as exc:
            self._error(exc)
            raise
        self._chunk(chunk)
        return chunk

    async def close(self) -> Any:
        try:
            close = getattr(self._original, "aclose", None) or getattr(
                self._original, "close", None
            )
            if close is not None:
                return await close()
            return None
        finally:
            self._finish("closed")

    async def aclose(self) -> Any:
        return await self.close()

    async def __aenter__(self) -> AsyncStreamWrapper:
        if hasattr(self._original, "__aenter__"):
            await self._original.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> Any:
        try:
            if hasattr(self._original, "__aexit__"):
                return await self._original.__aexit__(*args)
            close = getattr(self._original, "aclose", None) or getattr(
                self._original, "close", None
            )
            if close is not None:
                await close()
            return None
        finally:
            self._finish(
                "cancelled"
                if args
                and args[0] is not None
                and issubclass(args[0], asyncio.CancelledError)
                else "closed"
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)
