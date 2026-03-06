# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk", "torch", "timm"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""CLI wrapper example.

Run:
  WILDEDGE_DSN="https://<secret>@ingest.wildedge.dev/<key>" \
  wildedge run --print-startup-report --integrations timm -- \
  python examples/cli_wrapper_example.py
"""

import timm
import torch

model = timm.create_model("resnet18", pretrained=False).eval()
batch = torch.randn(1, 3, 224, 224)

with torch.inference_mode():
    output = model(batch)

print("output shape:", tuple(output.shape))
