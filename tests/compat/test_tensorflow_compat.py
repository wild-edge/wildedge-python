from __future__ import annotations

import pytest


def test_tensorflow_import_and_instrumentation(compat_client):
    np = pytest.importorskip("numpy")
    tf = pytest.importorskip("tensorflow")

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
