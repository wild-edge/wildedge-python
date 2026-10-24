from __future__ import annotations

import torch


def test_torch_load_and_inference_are_tracked(compat_client):
    model = compat_client.load(torch.nn.Linear, 4, 2)
    y = model(torch.randn(3, 4))
    assert tuple(y.shape) == (3, 2)

    load = compat_client.one("model_load")
    assert load["load"]["duration_ms"] >= 0
    inference = compat_client.one("inference")
    assert inference["model_id"] == load["model_id"]
    assert inference["inference"]["success"] is True
    assert inference["inference"]["batch_size"] == 3
