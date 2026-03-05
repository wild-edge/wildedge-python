# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge", "torch"]
#
# [tool.uv.sources]
# wildedge = { path = "..", editable = true }
# ///
"""PyTorch integration example — run with: uv run pytorch_example.py"""

import torch
import torch.nn as nn

import wildedge

client = wildedge.WildEdge(
    app_version="1.0.0",  # set WILDEDGE_DSN env var
)


class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


model = client.load(SimpleClassifier)
device = next(model.parameters()).device
batch = torch.randn(8, 128, device=device)

model.eval()

with torch.inference_mode():
    for _ in range(3):
        output = model(batch)
        print("output shape:", output.shape)
