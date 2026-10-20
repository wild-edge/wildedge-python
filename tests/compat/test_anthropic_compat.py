from __future__ import annotations

import anthropic
from conftest import sdk_http_module

httpx = sdk_http_module("anthropic")

MESSAGE = {
    "id": "msg_compat",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "Hello there"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 3},
}


def _client(handler) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key="sk-ant-compat",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_message_is_tracked(compat_client):
    compat_client.instrument("anthropic")
    client = _client(lambda request: httpx.Response(200, json=MESSAGE))

    result = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=32,
        messages=[{"role": "user", "content": "Say hello"}],
    )
    assert result.content[0].text == "Hello there"

    inference = compat_client.one("inference")
    assert inference["model_id"] == "claude-haiku-4-5"
    output = inference["inference"]["output_meta"]
    assert output["tokens_in"] == 12
    assert output["tokens_out"] == 3
    assert output["stop_reason"] == "end_turn"
    assert compat_client.registry.models["claude-haiku-4-5"].model_format == "api"


def test_api_error_is_tracked(compat_client):
    compat_client.instrument("anthropic")
    client = _client(
        lambda request: httpx.Response(
            529,
            json={
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
            },
        )
    )
    try:
        client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=8,
            messages=[{"role": "user", "content": "x"}],
        )
    except anthropic.APIStatusError:
        pass
    else:
        raise AssertionError("expected APIStatusError")

    error = compat_client.one("error")
    assert error["error"]["http_status"] == 529
    assert error["error"]["provider_error_code"] == "overloaded_error"
