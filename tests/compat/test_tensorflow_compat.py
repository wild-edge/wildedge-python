from __future__ import annotations

import numpy as np
import tensorflow as tf


def test_keras_model_load_and_call_are_tracked(compat_client):
    compat_client.instrument("tensorflow")
    model = compat_client.load(
        tf.keras.Sequential,
        [
            tf.keras.layers.Input(shape=(8,)),
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Dense(2),
        ],
    )
    out = model(np.random.randn(2, 8).astype(np.float32), training=False)
    assert tuple(out.shape) == (2, 2)

    load = compat_client.one("model_load")
    assert load["model_id"]
    inference = compat_client.one("inference")
    assert inference["model_id"] == load["model_id"]
    assert inference["inference"]["success"] is True
