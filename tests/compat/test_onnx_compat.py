from __future__ import annotations

import numpy as np
import onnxruntime as ort
from conftest import FIXTURES


def test_inference_session_load_and_run_are_tracked(compat_client):
    compat_client.instrument("onnx")
    session = ort.InferenceSession(
        str(FIXTURES / "identity.onnx"), providers=["CPUExecutionProvider"]
    )
    x = np.arange(4, dtype=np.float32).reshape(1, 4)
    (y,) = session.run(None, {"x": x})
    assert np.array_equal(x, y)

    # Auto-load takes the id from the file name, the rest from the session.
    load = compat_client.one("model_load")
    assert load["model_id"] == "identity"
    info = compat_client.registry.models["identity"]
    assert info.model_format == "onnx"
    assert info.model_name == "compat_identity"
    assert info.model_version == "3"
    inference = compat_client.one("inference")
    assert inference["model_id"] == "identity"
    assert inference["inference"]["success"] is True
