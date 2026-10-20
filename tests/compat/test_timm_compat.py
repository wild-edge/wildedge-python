from __future__ import annotations

import pytest
import timm
import torch


def test_timm_create_model_and_forward_are_tracked(compat_client):
    compat_client.instrument("timm")
    model = timm.create_model("resnet18", pretrained=False)
    y = model(torch.randn(1, 3, 64, 64))
    assert y.shape[0] == 1

    load = compat_client.one("model_load")
    assert load["load"]["duration_ms"] >= 0
    inference = compat_client.one("inference")
    assert inference["model_id"] == load["model_id"]
    assert inference["inference"]["input_modality"] == "image"


@pytest.mark.xfail(
    strict=True,
    reason="timm models are registered by class name, so architectures collide (#73)",
)
def test_timm_architectures_get_their_own_ids(compat_client):
    compat_client.instrument("timm")
    # Neither is loaded by another test, so each should add one model_load.
    timm.create_model("resnet34", pretrained=False)
    timm.create_model("resnet50", pretrained=False)
    ids = {e["model_id"] for e in compat_client.of_type("model_load")}
    assert len(ids) == 2
