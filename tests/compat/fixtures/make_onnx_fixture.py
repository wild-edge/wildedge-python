"""Regenerate identity.onnx, the model the onnx compat test loads.

    uv run --with onnx python tests/compat/fixtures/make_onnx_fixture.py

IR version 8 and opset 13 keep it loadable by the oldest onnxruntime in the
compatibility matrix.
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper

graph = helper.make_graph(
    [helper.make_node("Identity", ["x"], ["y"])],
    "compat_identity",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
model.model_version = 3
onnx.checker.check_model(model)
onnx.save(model, Path(__file__).with_name("identity.onnx"))
