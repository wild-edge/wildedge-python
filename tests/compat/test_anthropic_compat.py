from __future__ import annotations

import json

import anthropic
import pytest
from conftest import make_compat_client, sdk_http_module

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
@pytest.mark.parametrize("truncated", [False, True])
async def test_stream_lifecycle(compat_client, asynchronous, early_close, truncated):
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
    if truncated:
        events = events[:-1]
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
                    assert len([event async for event in stream]) == len(events)
            await stream.close()
    else:
        with _client(handler) as client:
            with client.messages.create(
                model="claude-haiku-4-5", max_tokens=32, messages=[], stream=True
            ) as stream:
                if early_close:
                    next(stream)
                else:
                    assert len(list(stream)) == len(events)
            stream.close()
    inference = compat_client.one("inference")["inference"]
    assert inference["success"] == (not early_close and not truncated)
    assert inference["output_meta"]["tokens_in"] == 12
    if early_close:
        assert inference["error_code"] == "STREAM_CLOSED"
        assert inference["output_meta"].get("time_to_first_token_ms") is None
    else:
        if truncated:
            assert inference["error_code"] == "STREAM_INCOMPLETE"
        assert inference["output_meta"]["tokens_out"] == 3
        assert inference["output_meta"]["time_to_first_token_ms"] is not None


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["events", "text", "final", "close", "truncated"])
async def test_native_stream_helper(compat_client, asynchronous, mode):
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
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello there"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 3},
        },
        {"type": "message_stop"},
    ]
    if mode == "truncated":
        events = events[:-1]
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )

    if asynchronous:
        async with (
            anthropic.AsyncAnthropic(
                api_key="sk-ant-compat",
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            ) as client,
            client.messages.stream(
                model="claude-haiku-4-5",
                max_tokens=32,
                messages=[{"role": "user", "content": "hi"}],
            ) as stream,
        ):
            if mode == "text":
                assert (
                    "".join([text async for text in stream.text_stream])
                    == "Hello there"
                )
            elif mode in ("final", "truncated"):
                assert (await stream.get_final_message()).content[
                    0
                ].text == "Hello there"
            elif mode == "events":
                assert [e.type async for e in stream][-1] == "message_stop"
            else:
                await anext(stream)
    else:
        with (
            _client(handler) as client,
            client.messages.stream(
                model="claude-haiku-4-5",
                max_tokens=32,
                messages=[{"role": "user", "content": "hi"}],
            ) as stream,
        ):
            if mode == "text":
                assert "".join(stream.text_stream) == "Hello there"
            elif mode in ("final", "truncated"):
                assert stream.get_final_message().content[0].text == "Hello there"
            elif mode == "events":
                assert [e.type for e in stream][-1] == "message_stop"
            else:
                next(stream)
    assert len(requests) == 1
    inference = compat_client.one("inference")["inference"]
    assert inference["success"] == (mode not in ("close", "truncated"))
    assert inference["output_meta"]["tokens_in"] == 12
    if mode == "close":
        assert inference["error_code"] == "STREAM_CLOSED"
    else:
        if mode == "truncated":
            assert inference["error_code"] == "STREAM_INCOMPLETE"
        assert inference["output_meta"]["tokens_out"] == 3
        assert inference["output_meta"]["time_to_first_token_ms"] is not None


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_native_stream_request_error(compat_client, asynchronous):
    compat_client.instrument("anthropic")

    def handler(request):
        return httpx.Response(
            429,
            json={
                "type": "error",
                "error": {"type": "rate_limit_error", "message": "slow down"},
            },
        )

    async def invoke_async():
        async with (
            anthropic.AsyncAnthropic(
                api_key="sk-ant-compat",
                max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            ) as client,
            client.messages.stream(
                model="claude-haiku-4-5", max_tokens=32, messages=[]
            ) as stream,
        ):
            await stream.get_final_message()

    def invoke_sync():
        with (
            _client(handler) as client,
            client.messages.stream(
                model="claude-haiku-4-5", max_tokens=32, messages=[]
            ) as stream,
        ):
            stream.get_final_message()

    if asynchronous:
        with pytest.raises(anthropic.RateLimitError):
            await invoke_async()
    else:
        with pytest.raises(anthropic.RateLimitError):
            invoke_sync()
    error = compat_client.one("error")["error"]
    assert error["http_status"] == 429
    assert error["provider_error_code"] == "rate_limit_error"
    assert not compat_client.of_type("inference")


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_native_stream_provider_error_midstream(compat_client, asynchronous):
    compat_client.instrument("anthropic")
    events = [
        {
            "type": "message_start",
            "message": {**MESSAGE, "content": [], "stop_reason": None},
        },
        {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "overloaded"},
        },
    ]
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)

    def handler(request):
        return httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )

    async def invoke_async():
        async with (
            anthropic.AsyncAnthropic(
                api_key="sk-ant-compat",
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            ) as client,
            client.messages.stream(
                model="claude-haiku-4-5", max_tokens=32, messages=[]
            ) as stream,
        ):
            await stream.get_final_message()

    def invoke_sync():
        with (
            _client(handler) as client,
            client.messages.stream(
                model="claude-haiku-4-5", max_tokens=32, messages=[]
            ) as stream,
        ):
            stream.get_final_message()

    if asynchronous:
        with pytest.raises(anthropic.APIError):
            await invoke_async()
    else:
        with pytest.raises(anthropic.APIError):
            invoke_sync()
    error = compat_client.one("error")["error"]
    assert error["provider_error_code"] == "overloaded_error"
    assert not compat_client.of_type("inference")


def test_replacement_client_keeps_native_helper_ownership(compat_client):
    compat_client.instrument("anthropic")
    old = _client(lambda request: httpx.Response(200, json=MESSAGE))
    old.messages.create(model="claude-haiku-4-5", max_tokens=32, messages=[])
    compat_client.close()
    replacement = make_compat_client()
    try:
        replacement.instrument("anthropic")
        old.messages.create(model="claude-haiku-4-5", max_tokens=32, messages=[])
        events = [
            {
                "type": "message_start",
                "message": {**MESSAGE, "content": [], "stop_reason": None},
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
            {"type": "message_stop"},
        ]
        body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
        with (
            _client(
                lambda request: httpx.Response(
                    200, text=body, headers={"content-type": "text/event-stream"}
                )
            ) as new,
            new.messages.stream(
                model="claude-haiku-4-5", max_tokens=32, messages=[]
            ) as stream,
        ):
            assert stream.get_final_message().usage.output_tokens == 3
        assert len(compat_client.of_type("inference")) == 1
        replacement.one("inference")
    finally:
        old.close()
        replacement.close()
