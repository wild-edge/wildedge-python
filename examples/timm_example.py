# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk", "torch", "timm"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""
timm integration example — run with: uv run timm_example.py

timm models are standard PyTorch nn.Module subclasses — wildedge patches
timm.create_model at client initialisation, so load timing, download tracking,
and unload tracking happen automatically. Inference tracking uses the existing
PyTorch forward hooks.
"""

import timm
import torch

import wildedge

client = wildedge.WildEdge(
    app_version="1.0.0",  # set WILDEDGE_DSN env var
)
client.instrument("timm", hubs=["huggingface", "torchhub"])

model = timm.create_model("resnet18", pretrained=True)
model.eval()

batch = torch.randn(4, 3, 224, 224)

with torch.inference_mode():
    for _ in range(3):
        output = model(batch)
        print("output shape:", output.shape)
