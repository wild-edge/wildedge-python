from __future__ import annotations

import json

import anthropic
import pytest
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


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("early_close", [False, True])
async def test_stream_lifecycle(compat_client, asynchronous, early_close):
    compat_client.instrument("anthropic")
    events = [
        {
            "type": "message_start",
            "message": {
                **MESSAGE,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 12, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 3},
        },
        {"type": "message_stop"},
    ]
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)

    def handler(request):
        return httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )

    if asynchronous:
        async with anthropic.AsyncAnthropic(
            api_key="sk-ant-compat",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            async with await client.messages.create(
                model="claude-haiku-4-5", max_tokens=32, messages=[], stream=True
            ) as stream:
                if early_close:
                    await anext(stream)
                else:
                    assert len([event async for event in stream]) == 4
            await stream.close()
    else:
        with _client(handler) as client:
            with client.messages.create(
                model="claude-haiku-4-5", max_tokens=32, messages=[], stream=True
            ) as stream:
                if early_close:
                    next(stream)
                else:
                    assert len(list(stream)) == 4
            stream.close()
    inference = compat_client.one("inference")["inference"]
    assert inference["success"] is not early_close
    assert inference["output_meta"]["tokens_in"] == 12
    if early_close:
        assert inference["error_code"] == "STREAM_CLOSED"
        assert inference["output_meta"].get("time_to_first_token_ms") is None
    else:
        assert inference["output_meta"]["tokens_out"] == 3
        assert inference["output_meta"]["time_to_first_token_ms"] is not None
