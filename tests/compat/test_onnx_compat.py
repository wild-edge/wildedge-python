from __future__ import annotations

import pytest


def test_onnxruntime_import_and_instrument(compat_client):
    onnxruntime = pytest.importorskip("onnxruntime")
    assert getattr(onnxruntime, "__version__", None)
    compat_client.instrument("onnx")
