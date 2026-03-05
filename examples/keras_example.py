# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge", "keras"]
#
# [tool.uv.sources]
# wildedge = { path = "..", editable = true }
# ///
"""
Keras integration example — run with: uv run keras_example.py

Keras models are user-defined subclasses, so wildedge cannot patch the
constructor directly like with timm or PyTorch. Use client.load() to
register the model and track load/unload timing automatically. Inference
tracking uses Keras callbacks.
"""

import numpy as np

import wildedge

# TensorFlow/Keras may not be available; skip gracefully
try:
    from keras.applications import MobileNetV2
    from keras.preprocessing.image import preprocess_input
except ImportError:
    print("Keras not installed. Install with: pip install keras tensorflow")
    exit(1)

client = wildedge.WildEdge(
    app_version="1.0.0",  # set WILDEDGE_DSN env var
)

# Load a pre-trained MobileNetV2 model using client to track construction and lifecycle
model = client.load(lambda: MobileNetV2(weights="imagenet"))
model.eval()

# Generate dummy image batch (batch_size=4, 224x224x3)
batch = np.random.randn(4, 224, 224, 3).astype(np.float32) * 255
batch = preprocess_input(batch)

# Run inference
for _ in range(3):
    output = model.predict(batch, verbose=0)
    print("output shape:", output.shape)
