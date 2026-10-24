from __future__ import annotations

import torch
import transformers


def _save_tiny_bert(path) -> None:
    config = transformers.BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=2,
    )
    transformers.BertForSequenceClassification(config).save_pretrained(path)


def test_from_pretrained_load_and_forward_are_tracked(compat_client, tmp_path):
    _save_tiny_bert(tmp_path)
    compat_client.instrument("transformers")

    model = transformers.AutoModelForSequenceClassification.from_pretrained(tmp_path)
    out = model(
        input_ids=torch.zeros((1, 4), dtype=torch.long),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
    )
    assert tuple(out.logits.shape) == (1, 2)

    load = compat_client.one("model_load")
    inference = compat_client.one("inference")
    assert inference["model_id"] == load["model_id"]
    assert inference["inference"]["success"] is True


def test_pipeline_load_and_call_are_tracked(compat_client, tmp_path):
    _save_tiny_bert(tmp_path)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    transformers.BertTokenizer(str(tmp_path / "vocab.txt")).save_pretrained(tmp_path)
    compat_client.instrument("transformers")

    classifier = transformers.pipeline(
        "text-classification", model=str(tmp_path), tokenizer=str(tmp_path)
    )
    result = classifier("hello world")
    assert result[0]["label"] in {"LABEL_0", "LABEL_1"}

    load = compat_client.one("model_load")
    inference = compat_client.one("inference")
    assert inference["model_id"] == load["model_id"]
    assert inference["inference"]["input_modality"] == "text"
