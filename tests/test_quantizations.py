"""Quantization labels are shared across integrations."""

from types import SimpleNamespace

import pytest

from wildedge import quantizations
from wildedge.integrations.common import dtype_to_quantization, mlx_layer_quantization
from wildedge.integrations.faster_whisper import COMPUTE_TYPE_QUANTIZATION
from wildedge.integrations.gguf import file_type_quantization, parse_quantization


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ("torch.bfloat16", quantizations.BF16),
        ("torch.float16", quantizations.F16),
        ("tensor(float16)", quantizations.F16),
        ("mlx.core.float32", quantizations.F32),
        ("torch.qint8", quantizations.INT8),
    ],
)
def test_dtype_names(dtype, expected):
    assert dtype_to_quantization(dtype) == expected


@pytest.mark.parametrize(
    ("compute_type", "expected"),
    [
        ("int8_float16", quantizations.INT8),
        ("int8_bfloat16", quantizations.INT8),
        ("float16", quantizations.F16),
        ("int16", "int16"),
    ],
)
def test_ctranslate2_compute_types_use_weight_precision(compute_type, expected):
    assert COMPUTE_TYPE_QUANTIZATION[compute_type] == expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [("llama-7b.Q4_K_M", "q4_k_m"), ("model-f16", "f16"), ("model.i8", "int8")],
)
def test_gguf_filename_suffixes(filename, expected):
    assert parse_quantization(filename) == expected


@pytest.mark.parametrize(
    ("file_type", "expected"),
    [("15", "q4_k_m"), (0, quantizations.F32), ("32", quantizations.BF16)],
)
def test_gguf_file_types(file_type, expected):
    assert file_type_quantization(file_type) == expected


@pytest.mark.parametrize("file_type", ["1024", "not-a-number", "5"])
def test_unknown_gguf_file_types_are_not_reported(file_type):
    assert file_type_quantization(file_type) is None


class QuantizedLinear:
    pass


@pytest.mark.parametrize(
    ("layers", "expected"),
    [
        ([SimpleNamespace(bits=4)], "int4"),
        ([QuantizedLinear()], quantizations.QUANTIZED),
        ([SimpleNamespace()], None),
    ],
)
def test_mlx_layers(layers, expected):
    model = SimpleNamespace(named_modules=lambda: [("", layer) for layer in layers])
    assert mlx_layer_quantization(model) == expected
