from __future__ import annotations

import json

import openai
from conftest import sdk_http_module

httpx = sdk_http_module("openai")

COMPLETION = {
    "id": "chatcmpl-compat",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
}


def _client(handler) -> openai.OpenAI:
    return openai.OpenAI(
        api_key="sk-compat",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_chat_completion_is_tracked(compat_client):
    compat_client.instrument("openai")
    client = _client(lambda request: httpx.Response(200, json=COMPLETION))

    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        temperature=0.2,
    )
    assert result.choices[0].message.content == "Hello there"

    inference = compat_client.one("inference")
    assert inference["model_id"] == "gpt-4o-mini"
    output = inference["inference"]["output_meta"]
    assert output["tokens_in"] == 9
    assert output["tokens_out"] == 2
    assert output["stop_reason"] == "stop"


def test_streamed_chat_completion_is_tracked(compat_client):
    compat_client.instrument("openai")
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
        },
    ]
    base = {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
    }
    body = "".join(f"data: {json.dumps({**base, **c})}\n\n" for c in chunks)
    body += "data: [DONE]\n\n"
    client = _client(
        lambda request: httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )
    )

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hi"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    assert (
        "".join(c.choices[0].delta.content or "" for c in stream if c.choices) == "Hi"
    )

    inference = compat_client.one("inference")
    output = inference["inference"]["output_meta"]
    assert output["tokens_out"] == 1
    assert output["stop_reason"] == "stop"
    assert output["time_to_first_token_ms"] >= 0


def test_api_error_is_tracked(compat_client):
    compat_client.instrument("openai")
    client = _client(
        lambda request: httpx.Response(
            429,
            json={
                "error": {
                    "message": "slow down",
                    "type": "rate_limit",
                    "code": "rate_limit_exceeded",
                }
            },
        )
    )
    try:
        client.with_options(max_retries=0).chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}]
        )
    except openai.RateLimitError:
        pass
    else:
        raise AssertionError("expected RateLimitError")

    error = compat_client.one("error")
    assert error["error"]["http_status"] == 429
    assert error["error"]["provider_error_code"] == "rate_limit_exceeded"
