from __future__ import annotations

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


def test_timm_architectures_get_their_own_ids(compat_client):
    compat_client.instrument("timm")
    # Neither is loaded by another test, so each should add one model_load.
    models = [
        timm.create_model(name, pretrained=False).eval()
        for name in ("resnet34", "resnet50")
    ]
    with torch.no_grad():
        for model in models:
            model(torch.randn(1, 3, 64, 64))
    ids = {e["model_id"] for e in compat_client.of_type("model_load")}
    assert ids == {"resnet34", "resnet50"}
    assert {e["model_id"] for e in compat_client.of_type("inference")} == ids
    assert {compat_client.registry.models[name].model_name for name in ids} == ids


def test_repeated_architecture_reuses_identity(compat_client):
    compat_client.instrument("timm")
    models = [
        timm.create_model(model_name="resnet18", pretrained=False).eval()
        for _ in range(2)
    ]
    with torch.no_grad():
        for model in models:
            model(torch.randn(1, 3, 64, 64))
    events = compat_client.of_type("inference")
    assert len(events) == 2
    assert {event["model_id"] for event in events} == {"resnet18"}
    assert compat_client.registry.models["resnet18"].model_name == "resnet18"
