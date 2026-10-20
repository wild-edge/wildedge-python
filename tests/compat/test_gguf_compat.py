from __future__ import annotations

import llama_cpp
import pytest
from conftest import downloaded_fixture

# The 260K-parameter TinyStories Llama from karpathy/llama2.c (MIT), converted
# to GGUF by ggml-org. 1.2 MB. If the link breaks, point it at another copy.
MODEL_URL = (
    "https://huggingface.co/ggml-org/models-moved/resolve/"
    "499bc8821c6b12b4e53c5bffcb21ec206f212d81/tinyllamas/stories260K.gguf"
)
MODEL_SHA256 = "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d"


@pytest.fixture
def llm(compat_client):
    model = downloaded_fixture("stories260K-f32.gguf", MODEL_URL, MODEL_SHA256)
    compat_client.instrument("gguf")
    return llama_cpp.Llama(model_path=str(model), n_ctx=128, seed=1, verbose=False)


def test_llama_load_is_tracked(compat_client, llm):
    load = compat_client.one("model_load")
    assert load["model_id"] == "stories260K-f32"
    info = compat_client.registry.models["stories260K-f32"]
    assert info.model_format == "gguf"
    assert info.model_family == "llama"
    assert info.quantization == "f32"


def test_llama_call_is_tracked(compat_client, llm):
    result = llm("Once upon a time", max_tokens=4)
    assert result["choices"][0]["text"] is not None

    inference = compat_client.one("inference")
    assert inference["model_id"] == "stories260K-f32"
    output = inference["inference"]["output_meta"]
    assert output["tokens_in"] == result["usage"]["prompt_tokens"]
    assert output["tokens_out"] == result["usage"]["completion_tokens"]


def test_create_completion_is_tracked_with_its_config(compat_client, llm):
    result = llm.create_completion(
        "Once upon a time", max_tokens=4, temperature=0.3, stop=["\n"]
    )

    inference = compat_client.one("inference")["inference"]
    output = inference["output_meta"]
    assert output["tokens_in"] == result["usage"]["prompt_tokens"]
    assert output["tokens_out"] == result["usage"]["completion_tokens"]
    assert output["stop_reason"] == result["choices"][0]["finish_reason"]
    config = inference["generation_config"]
    assert config["temperature"] == 0.3
    assert config["max_tokens"] == 4
    assert config["stop_sequences_count"] == 1
    # Arguments not passed are reported with llama-cpp-python's defaults.
    assert "top_p" in config
    assert "top_k" in config


def test_chat_completion_is_tracked_once_as_chat(compat_client, llm):
    result = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": "Tell me a story"},
            {"role": "assistant", "content": "Once upon a time"},
            {"role": "user", "content": "Go on"},
        ],
        max_tokens=4,
    )

    inference = compat_client.one("inference")["inference"]
    assert inference["input_meta"]["prompt_type"] == "chat"
    assert inference["input_meta"]["turn_index"] == 1
    assert inference["input_meta"]["word_count"] == 2
    output = inference["output_meta"]
    assert output["tokens_in"] == result["usage"]["prompt_tokens"]
    assert output["stop_reason"] == result["choices"][0]["finish_reason"]
    assert inference["generation_config"]["max_tokens"] == 4


def test_streamed_chat_completion_is_tracked_once(compat_client, llm):
    stream = llm.create_chat_completion(
        messages=[{"role": "user", "content": "Tell me a story"}],
        max_tokens=4,
        stream=True,
    )
    chunks = list(stream)
    assert chunks

    inference = compat_client.one("inference")["inference"]
    assert inference["input_meta"]["prompt_type"] == "chat"
    output = inference["output_meta"]
    assert output["time_to_first_token_ms"] >= 0
    assert output["stop_reason"] in {"stop", "length"}
