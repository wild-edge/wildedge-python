"""Shared utilities used across framework integrations."""

from __future__ import annotations

from typing import Any

from wildedge.logging import logger


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
    """Map a dtype string to a quantization label, or None if unrecognised.

    Handles PyTorch repr strings (``"torch.float16"``), ONNX type strings
    (``"tensor(float16)"``), and plain names (``"float16"``).
    """
    for needle, label in DTYPE_QUANTIZATION_MAP.items():
        if needle in dtype_str:
            return label
    return None


def image_brightness_histogram(norm_flat: Any) -> tuple[float, float, list[int]]:
    """Compute brightness stats from a flat min-max-normalised (0–1) array.

    Compatible with NumPy arrays and PyTorch tensors.
    Returns (brightness_mean, brightness_stddev, buckets) where buckets covers
    [0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0] (last bucket
    inclusive on both ends).
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
    """Return "image" for 4-D tensors (N, C, H, W) or (N, H, W, C), else None."""
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
    """Scan Keras/TF layer class names to guess input domain.

    Conv layers → image; sequence/embedding layers → text.
    """
    names_set = set(class_names)
    if names_set & CONV_LAYER_TYPES:
        return "image"
    if names_set & SEQUENCE_LAYER_TYPES:
        return "text"
    return None


def num_classes_from_output_shape(shape: tuple) -> int:
    """Return the number of output classes from a model output shape, or 0.

    Expects shapes like ``(None, 10)`` or ``(batch, num_classes)``.
    Returns 0 when the shape does not look like a classification head.
    """
    if len(shape) >= 2 and isinstance(shape[-1], int) and shape[-1] > 1:
        return int(shape[-1])
    return 0
