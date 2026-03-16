"""Tests for wildedge.integrations.common."""

from __future__ import annotations

import logging

import pytest

from wildedge.integrations.common import (
    debug_failure,
    dtype_to_quantization,
    image_brightness_histogram,
    infer_input_modality_from_layer_types,
    infer_input_modality_from_names,
    infer_input_modality_from_shape,
    num_classes_from_output_shape,
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


# ---------------------------------------------------------------------------
# infer_input_modality_from_shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((4, 3, 224, 224), "image"),  # NCHW image batch
        ((1, 3, 64, 64), "image"),  # single image
        ((32, 512), None),  # 2D, no signal
        ((128,), None),  # 1D vector
        ((4, 3, 224, 224, 2), None),  # 5D, not handled
        ((), None),  # scalar
    ],
)
def test_infer_input_modality_from_shape(shape, expected):
    assert infer_input_modality_from_shape(shape) == expected


# ---------------------------------------------------------------------------
# infer_input_modality_from_names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "names, expected",
    [
        (["input_ids", "attention_mask"], "text"),
        (["input_ids"], "text"),
        (["token_type_ids", "attention_mask"], "text"),
        (["position_ids"], "text"),
        (["input_values"], "audio"),
        (["input_features"], "audio"),
        # text takes priority over audio if both present
        (["input_ids", "input_values"], "text"),
        # generic names → no signal
        (["input", "output"], None),
        (["pixel_values", "labels"], None),
        ([], None),
    ],
)
def test_infer_input_modality_from_names(names, expected):
    assert infer_input_modality_from_names(names) == expected


# ---------------------------------------------------------------------------
# infer_input_modality_from_layer_types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layer_types, expected",
    [
        (["InputLayer", "Conv2D", "Dense"], "image"),
        (["InputLayer", "DepthwiseConv2D", "GlobalAveragePooling2D", "Dense"], "image"),
        (["InputLayer", "Embedding", "LSTM", "Dense"], "text"),
        (["InputLayer", "Bidirectional", "Dense"], "text"),
        (["InputLayer", "Dense", "Dense"], None),  # fully connected, no signal
        ([], None),
        # Conv wins over sequence when both present (unusual but well-defined)
        (["Conv2D", "Embedding"], "image"),
    ],
)
def test_infer_input_modality_from_layer_types(layer_types, expected):
    assert infer_input_modality_from_layer_types(layer_types) == expected


# ---------------------------------------------------------------------------
# num_classes_from_output_shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((None, 10), 10),  # standard classification head
        ((None, 1000), 1000),  # ImageNet
        ((32, 10), 10),  # concrete batch size
        ((None, 1), 0),  # binary output, not multi-class
        ((None,), 0),  # 1D, no class axis
        ((), 0),  # scalar
        ((None, None), 0),  # dynamic last dim
        ((None, 10, 5), 5),  # sequence output with 5 classes per token
    ],
)
def test_num_classes_from_output_shape(shape, expected):
    assert num_classes_from_output_shape(shape) == expected
