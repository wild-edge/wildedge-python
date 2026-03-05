from __future__ import annotations

import pytest


def test_timm_import_and_instrumentation(compat_client):
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")

    compat_client.instrument("timm")
    model = timm.create_model("resnet18", pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape[0] == 1
