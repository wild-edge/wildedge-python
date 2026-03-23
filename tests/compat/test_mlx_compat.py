from __future__ import annotations

import pytest


def test_mlx_import_and_instrumentation(compat_client):
    pytest.importorskip("mlx_lm")
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

    compat_client.instrument("mlx")

    model = nn.Linear(4, 2)
    x = mx.ones((3, 4))
    y = model(x)
    mx.eval(y)
    assert y.shape == (3, 2)
