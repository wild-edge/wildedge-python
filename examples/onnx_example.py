# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk", "onnxruntime", "huggingface_hub", "numpy"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""ONNX Runtime integration example. Run with: uv run onnx_example.py"""

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

import wildedge

client = wildedge.WildEdge(
    app_version="1.0.0",  # uses WILDEDGE_DSN if set; otherwise no-op
)
client.instrument("onnx", hubs=["huggingface"])

model_path = hf_hub_download("Xenova/resnet-50", "onnx/model.onnx")
session = ort.InferenceSession(model_path)
batch = np.random.randn(4, 3, 224, 224).astype(np.float32)

for _ in range(3):
    outputs = session.run(None, {"pixel_values": batch})
    print("output shape:", outputs[0].shape)
