# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk", "torch", "timm"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""
Feedback example. Run with: uv run feedback_example.py

Simulates an automated quality gate: after each inference, the top-1 confidence
score drives a thumbs_up / thumbs_down feedback event with no human input.
This pattern fits batch pipelines where downstream validation is programmatic.
"""

import timm
import torch

import wildedge
from wildedge import FeedbackType

CONFIDENCE_THRESHOLD = 0.6

client = wildedge.init(
    app_version="1.0.0",  # uses WILDEDGE_DSN if set; otherwise no-op
    integrations="timm",
)

model = timm.create_model("resnet18", pretrained=True)
model.eval()
handle = client.register_model(
    model
)  # auto-instrumented already; returns existing handle

batch = torch.randn(4, 3, 224, 224)

with torch.inference_mode():
    for i in range(3):
        output = model(batch)

        confidence = float(torch.softmax(output, dim=-1).max(dim=-1).values.mean())
        # Simulate feedback based on confidence threshold with no human input.
        handle.feedback(
            FeedbackType.THUMBS_UP
            if confidence >= CONFIDENCE_THRESHOLD
            else FeedbackType.THUMBS_DOWN
        )
        print(
            f"run {i + 1}: confidence={confidence:.3f} → {handle.last_inference_id[:8]}..."
        )
