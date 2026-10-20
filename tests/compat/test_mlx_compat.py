from __future__ import annotations

import json

import mlx.core as mx
import mlx_lm
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import llama
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

VOCAB = ["<unk>", "<s>", "</s>", "once", "upon", "a", "time", "there", "was"]


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """A random-weight two-layer Llama with a word-level tokenizer, built locally."""
    path = tmp_path_factory.mktemp("tiny-llama")
    config = {
        "model_type": "llama",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "vocab_size": len(VOCAB),
        "tie_word_embeddings": True,
    }
    (path / "config.json").write_text(json.dumps(config))
    model = llama.Model(llama.ModelArgs.from_dict(config))
    mx.save_safetensors(
        str(path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )

    tokenizer = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
    ).save_pretrained(path)
    return path


def test_mlx_lm_load_and_generate_are_tracked(compat_client, tiny_model_dir):
    compat_client.instrument("mlx")
    model, tokenizer = mlx_lm.load(str(tiny_model_dir))
    text = mlx_lm.generate(model, tokenizer, prompt="once upon a time", max_tokens=3)
    assert isinstance(text, str)

    load = compat_client.one("model_load")
    inference = compat_client.one("inference")
    assert inference["model_id"] == load["model_id"]
    output = inference["inference"]["output_meta"]
    assert output["tokens_out"] >= 1
