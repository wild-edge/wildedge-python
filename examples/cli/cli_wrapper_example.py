"""CLI wrapper example.

Run this example with:
  export WILDEDGE_DSN="https://<secret>@ingest.wildedge.dev/<key>"
  ./examples/cli/demo.sh
"""

import timm
import torch

model = timm.create_model("resnet18", pretrained=False).eval()
batch = torch.randn(1, 3, 224, 224)
iterations = 500

for _ in range(iterations):
  with torch.inference_mode():
      output = model(batch)

# print("output shape:", tuple(output.shape))
