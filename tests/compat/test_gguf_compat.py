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


@pytest.mark.xfail(
    strict=True,
    reason="create_completion and create_chat_completion are not tracked (#64)",
)
def test_chat_and_completion_calls_are_tracked(compat_client, llm):
    llm.create_completion("Once upon a time", max_tokens=4)
    llm.create_chat_completion(
        messages=[{"role": "user", "content": "Tell me a story"}], max_tokens=4
    )
    assert len(compat_client.of_type("inference")) == 2
