from __future__ import annotations

import pytest


def test_transformers_import_and_instrumentation(compat_client):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    compat_client.instrument("transformers")

    config = transformers.BertConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=2,
    )
    model = transformers.BertForSequenceClassification(config)
    inputs = {
        "input_ids": torch.zeros((1, 4), dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    out = model(**inputs)
    assert tuple(out.logits.shape) == (1, 2)
