# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk", "tensorflow", "numpy"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""TensorFlow integration example. Run with: uv run tensorflow_example.py."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf

import wildedge

client = wildedge.WildEdge(
    app_version="1.0.0",  # uses WILDEDGE_DSN if set; otherwise no-op
)
client.instrument("tensorflow")


def build_and_save_model(save_path: Path) -> None:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(16,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(8),
        ]
    )
    # Trigger variable creation before save.
    _ = model(np.zeros((1, 16), dtype=np.float32))
    model.save(save_path)


with TemporaryDirectory() as temp_dir:
    model_path = Path(temp_dir) / "demo_model.keras"
    build_and_save_model(model_path)

    # load_model is auto-instrumented by client.instrument("tensorflow")
    loaded = tf.keras.models.load_model(model_path)

    batch = np.random.randn(4, 16).astype(np.float32)
    output = loaded(batch, training=False)
    print("output shape:", tuple(output.shape))


client.close()
