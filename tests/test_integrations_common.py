"""Tests for wildedge.integrations.common."""

from __future__ import annotations

import logging

import pytest

from wildedge.integrations.common import (
    debug_failure,
    dtype_to_quantization,
    image_brightness_histogram,
)

# ---------------------------------------------------------------------------
# debug_failure
# ---------------------------------------------------------------------------


def test_debug_failure_logs_correctly(caplog):
    with caplog.at_level(logging.DEBUG, logger="wildedge"):
        debug_failure("pytorch", "quantization detection", ValueError("boom"))

    assert len(caplog.records) == 1
    assert "pytorch" in caplog.records[0].message
    assert "quantization detection" in caplog.records[0].message
    assert "boom" in caplog.records[0].message


# ---------------------------------------------------------------------------
# dtype_to_quantization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype_str, expected",
    [
        # PyTorch repr strings
        ("torch.float16", "f16"),
        ("torch.bfloat16", "bf16"),
        ("torch.float32", "f32"),
        ("torch.qint8", "int8"),
        ("torch.quint8", "int8"),
        # ONNX type strings
        ("tensor(float16)", "f16"),
        ("tensor(float)", None),  # ONNX plain float doesn't contain "float32"
        ("tensor(int8)", "int8"),
        ("tensor(uint8)", "int8"),
        # Plain names
        ("float16", "f16"),
        ("bfloat16", "bf16"),
        ("int8", "int8"),
        ("float32", "f32"),
        # bfloat16 must not match float16
        ("bfloat16", "bf16"),
        # Unknown
        ("float64", None),
        ("complex64", None),
        ("", None),
    ],
)
def test_dtype_to_quantization(dtype_str, expected):
    assert dtype_to_quantization(dtype_str) == expected


# ---------------------------------------------------------------------------
# image_brightness_histogram
# ---------------------------------------------------------------------------


def test_histogram_all_zeros():
    np = pytest.importorskip("numpy")
    flat = np.zeros(100, dtype=np.float32)
    mean, std, buckets = image_brightness_histogram(flat)
    assert mean == 0.0
    assert std == 0.0
    assert buckets[0] == 100
    assert sum(buckets[1:]) == 0


def test_histogram_all_ones():
    np = pytest.importorskip("numpy")
    flat = np.ones(100, dtype=np.float32)
    mean, std, buckets = image_brightness_histogram(flat)
    assert mean == 1.0
    assert std == 0.0
    # all pixels == 1.0 land in last bucket (inclusive)
    assert buckets[-1] == 100
    assert sum(buckets[:-1]) == 0


def test_histogram_uniform_distribution():
    np = pytest.importorskip("numpy")
    # 500 pixels evenly spread [0, 0.2, 0.4, 0.6, 0.8, 1.0)
    flat = np.linspace(0.0, 1.0, 500, dtype=np.float32)
    mean, std, buckets = image_brightness_histogram(flat)
    assert abs(mean - 0.5) < 0.01
    # Each bucket should have roughly equal counts
    for b in buckets:
        assert b > 0
    assert sum(buckets) == 500


def test_histogram_five_buckets():
    np = pytest.importorskip("numpy")
    flat = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    _, _, buckets = image_brightness_histogram(flat)
    assert len(buckets) == 5
    assert buckets == [1, 1, 1, 1, 1]


def test_histogram_last_bucket_inclusive():
    np = pytest.importorskip("numpy")
    flat = np.array([1.0, 1.0, 0.85], dtype=np.float32)
    _, _, buckets = image_brightness_histogram(flat)
    # 0.85 is in [0.8, 1.0); 1.0 added via the inclusive correction
    assert buckets[-1] == 3


def test_histogram_mean_and_std_rounded_to_4dp():
    np = pytest.importorskip("numpy")
    flat = np.array([0.0, 1.0], dtype=np.float32)
    mean, std, _ = image_brightness_histogram(flat)
    assert mean == round(mean, 4)
    assert std == round(std, 4)
