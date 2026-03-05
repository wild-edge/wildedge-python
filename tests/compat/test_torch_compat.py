from __future__ import annotations

import pytest


def test_torch_import_and_load_tracking(compat_client):
    torch = pytest.importorskip("torch")
    model = compat_client.load(torch.nn.Linear, 4, 2)
    x = torch.randn(3, 4)
    y = model(x)
    assert tuple(y.shape) == (3, 2)
