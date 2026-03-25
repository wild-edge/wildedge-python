"""Shared utilities used across framework integrations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from wildedge import constants
from wildedge.logging import logger
from wildedge.timing import elapsed_ms

if TYPE_CHECKING:
    from wildedge.model import ModelHandle


def debug_failure(framework: str, context: str, exc: BaseException) -> None:
    logger.debug("wildedge: %s %s failed: %s", framework, context, exc)


DTYPE_QUANTIZATION_MAP: dict[str, str] = {
    "bfloat16": "bf16",
    "float16": "f16",
    "float32": "f32",
    "int8": "int8",
    "qint": "int8",
    "quint": "int8",
    "uint8": "int8",
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
        for lo, hi in zip(edges, edges[1:])
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
#   on_done(duration_ms, ttft_ms)  : record inference once the stream is exhausted
#
# The wrappers handle TTFT capture, error tracking, context-manager delegation,
# and attribute proxying so callers get a drop-in replacement for the raw stream.


class SyncStreamWrapper:
    """Wraps a sync iterable stream to capture TTFT and record inference on exhaustion."""

    def __init__(
        self,
        original: object,
        handle: ModelHandle,
        t0: float,
        on_chunk: Callable[[object], None] | None,
        on_done: Callable[[int, int | None], None],
    ) -> None:
        self._original = original
        self._handle = handle
        self._t0 = t0
        self._on_chunk = on_chunk
        self._on_done = on_done

    def __iter__(self):
        return self._track()

    def _track(self):
        ttft_ms: int | None = None
        try:
            for chunk in self._original:  # type: ignore[union-attr]
                if ttft_ms is None:
                    ttft_ms = elapsed_ms(self._t0)
                if self._on_chunk is not None:
                    self._on_chunk(chunk)
                yield chunk
        except Exception as exc:
            self._handle.track_error(
                error_code="UNKNOWN",
                error_message=str(exc)[: constants.ERROR_MSG_MAX_LEN],
            )
            raise
        else:
            self._on_done(elapsed_ms(self._t0), ttft_ms)

    def __enter__(self) -> SyncStreamWrapper:
        if hasattr(self._original, "__enter__"):
            self._original.__enter__()  # type: ignore[union-attr]
        return self

    def __exit__(self, *args: object) -> object:
        if hasattr(self._original, "__exit__"):
            return self._original.__exit__(*args)  # type: ignore[union-attr]
        return None

    def __getattr__(self, name: str) -> object:
        return getattr(self._original, name)


class AsyncStreamWrapper:
    """Wraps an async iterable stream to capture TTFT and record inference on exhaustion."""

    def __init__(
        self,
        original: object,
        handle: ModelHandle,
        t0: float,
        on_chunk: Callable[[object], None] | None,
        on_done: Callable[[int, int | None], None],
    ) -> None:
        self._original = original
        self._handle = handle
        self._t0 = t0
        self._on_chunk = on_chunk
        self._on_done = on_done

    def __aiter__(self):
        return self._track()

    async def _track(self):
        ttft_ms: int | None = None
        try:
            async for chunk in self._original:  # type: ignore[union-attr]
                if ttft_ms is None:
                    ttft_ms = elapsed_ms(self._t0)
                if self._on_chunk is not None:
                    self._on_chunk(chunk)
                yield chunk
        except Exception as exc:
            self._handle.track_error(
                error_code="UNKNOWN",
                error_message=str(exc)[: constants.ERROR_MSG_MAX_LEN],
            )
            raise
        else:
            self._on_done(elapsed_ms(self._t0), ttft_ms)

    async def __aenter__(self) -> AsyncStreamWrapper:
        if hasattr(self._original, "__aenter__"):
            await self._original.__aenter__()  # type: ignore[union-attr]
        return self

    async def __aexit__(self, *args: object) -> object:
        if hasattr(self._original, "__aexit__"):
            return await self._original.__aexit__(*args)  # type: ignore[union-attr]
        return None

    def __getattr__(self, name: str) -> object:
        return getattr(self._original, name)
