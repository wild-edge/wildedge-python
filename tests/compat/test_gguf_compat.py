from __future__ import annotations

import llama_cpp
import pytest
from conftest import FIXTURES

MODEL = FIXTURES / "stories260K-f32.gguf"


@pytest.fixture
def llm(compat_client):
    compat_client.instrument("gguf")
    return llama_cpp.Llama(model_path=str(MODEL), n_ctx=128, seed=1, verbose=False)


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
