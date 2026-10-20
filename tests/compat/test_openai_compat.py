from __future__ import annotations

import json
import runpy
from pathlib import Path

import openai
import pytest
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


RESPONSE = {
    "id": "resp-compat",
    "object": "response",
    "created_at": 1700000000,
    "model": "gpt-4o-mini",
    "status": "completed",
    "error": None,
    "incomplete_details": None,
    "output": [],
    "parallel_tool_calls": True,
    "tools": [],
    "tool_choice": "auto",
    "usage": {
        "input_tokens": 9,
        "output_tokens": 2,
        "total_tokens": 11,
        "input_tokens_details": {"cached_tokens": 3},
        "output_tokens_details": {"reasoning_tokens": 1},
    },
}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_responses_is_tracked(compat_client, asynchronous, streaming):
    compat_client.instrument("openai")
    with openai.OpenAI(api_key="sk-compat") as probe:
        if not hasattr(probe, "responses"):
            return  # The minimum supported openai version predates Responses.

    def handler(request):
        assert request.url.path == "/v1/responses"
        if not streaming:
            return httpx.Response(200, json=RESPONSE)
        events = [
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {**RESPONSE, "status": "in_progress", "usage": None},
            },
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": "msg-1",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hi",
                "logprobs": [],
            },
            {"type": "response.completed", "sequence_number": 2, "response": RESPONSE},
        ]
        body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
        return httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )

    if asynchronous:
        client = openai.AsyncOpenAI(
            api_key="sk-compat",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        async with client:
            result = await client.responses.create(
                model="gpt-4o-mini", input="Say hello", stream=streaming
            )
            if streaming:
                assert [e.type async for e in result][-1] == "response.completed"
            else:
                assert result.id == RESPONSE["id"]
    else:
        with _client(handler) as client:
            result = client.responses.create(
                model="gpt-4o-mini", input="Say hello", stream=streaming
            )
            if streaming:
                assert [e.type for e in result][-1] == "response.completed"
            else:
                assert result.id == RESPONSE["id"]
    event = compat_client.one("inference")
    assert event["model_id"] == "gpt-4o-mini"
    output = event["inference"]["output_meta"]
    assert output["tokens_in"] == 9
    assert output["tokens_out"] == 2
    assert output["cached_input_tokens"] == 3
    assert output["reasoning_tokens_out"] == 1
    assert output["stop_reason"] == "completed"


async def test_responses_example(compat_client):
    compat_client.instrument("openai")
    with openai.OpenAI(api_key="sk-compat") as probe:
        if not hasattr(probe, "responses"):
            return
    example = runpy.run_path(
        str(Path(__file__).parents[2] / "examples" / "openai_responses_example.py")
    )
    payload = {
        **RESPONSE,
        "output": [
            {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello from Responses",
                        "annotations": [],
                    }
                ],
            }
        ],
    }

    def handler(request):
        if not json.loads(request.content).get("stream"):
            return httpx.Response(200, json=payload)
        event = {
            "type": "response.completed",
            "sequence_number": 0,
            "response": payload,
        }
        return httpx.Response(
            200,
            text=f"event: response.completed\ndata: {json.dumps(event)}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    example["run_sync"].__globals__["OpenAI"] = lambda: _client(handler)
    example["run_async"].__globals__["AsyncOpenAI"] = lambda: openai.AsyncOpenAI(
        api_key="sk-compat",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with compat_client.trace(run_id="responses-example-test"):
        example["run_sync"]()
        await example["run_async"]()
    events = compat_client.of_type("inference")
    assert len(events) == 4
    assert all(e["run_id"] == "responses-example-test" for e in events)
    assert len({e["parent_span_id"] for e in events}) == 4


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("encoding", ["float", "base64"])
async def test_embeddings_is_tracked(compat_client, asynchronous, encoding):
    compat_client.instrument("openai")

    def handler(request):
        assert request.url.path == "/v1/embeddings"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": [0.25, 0.5, 0.75]
                        if encoding == "float"
                        else "AACAPgAAAD8AAEA/",
                    }
                    for i in range(2)
                ],
                "usage": {"prompt_tokens": 7, "total_tokens": 7},
            },
        )

    if asynchronous:
        client = openai.AsyncOpenAI(
            api_key="sk-compat",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        async with client:
            result = await client.embeddings.create(
                model="text-embedding-3-small",
                input=["hello", "world"],
                encoding_format=encoding,
            )
    else:
        with _client(handler) as client:
            result = client.embeddings.create(
                model="text-embedding-3-small",
                input=["hello", "world"],
                encoding_format=encoding,
            )
    assert len(result.data) == 2
    event = compat_client.one("inference")
    inference = event["inference"]
    assert inference["batch_size"] == 2
    assert inference["input_meta"]["token_count"] == 7
    assert inference["output_meta"] == {
        "task": "embedding",
        "dimensions": 3,
        "tokens_in": 7,
    }
    assert inference["api_meta"]["resolved_model_id"] == "text-embedding-3-small"
    assert "embedding" not in inference["output_meta"]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_early_stream_close_is_tracked_once(compat_client, asynchronous):
    compat_client.instrument("openai")
    chunk = {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {"content": "Hi"}}],
    }
    body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"

    def handler(request):
        return httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )

    if asynchronous:
        async with openai.AsyncOpenAI(
            api_key="sk-compat",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            async with await client.chat.completions.create(
                model="gpt-4o-mini", messages=[], stream=True
            ) as stream:
                assert (await anext(stream)).choices[0].delta.content == "Hi"
            await stream.close()
    else:
        with _client(handler) as client:
            with client.chat.completions.create(
                model="gpt-4o-mini", messages=[], stream=True
            ) as stream:
                assert next(stream).choices[0].delta.content == "Hi"
            stream.close()
    inference = compat_client.one("inference")["inference"]
    assert inference["success"] is False
    assert inference["error_code"] == "STREAM_CLOSED"
    assert inference["output_meta"]["time_to_first_token_ms"] is not None
