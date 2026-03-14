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
