from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from wildedge.integrations.openai import (
    make_responses_stream_callbacks,
    wrap_async_responses,
    wrap_sync_responses,
)


def response(status="completed"):
    return SimpleNamespace(
        model="resolved",
        status=status,
        error=None,
        incomplete_details=SimpleNamespace(reason="max_output_tokens")
        if status == "incomplete"
        else None,
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=4,
            input_tokens_details=SimpleNamespace(cached_tokens=3),
            output_tokens_details=SimpleNamespace(reasoning_tokens=2),
        ),
    )


class AsyncItems:
    def __init__(self, items):
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("status", ["completed", "incomplete", "failed"])
async def test_responses(asynchronous, streaming, status):
    result = response(status)
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(type="response.output_text.delta", delta="hello"),
        SimpleNamespace(type=f"response.{status}", response=result),
    ]
    returned = (
        (AsyncItems(events) if asynchronous else iter(events)) if streaming else result
    )
    original = (AsyncMock if asynchronous else MagicMock)(return_value=returned)
    resource = SimpleNamespace(create=original)
    handle = MagicMock()
    client = SimpleNamespace(
        closed=False, register_model=MagicMock(return_value=handle)
    )
    (wrap_async_responses if asynchronous else wrap_sync_responses)(
        resource, "openai", lambda: client
    )
    call = resource.create(model="model", input="hello world", stream=streaming)
    actual = await call if asynchronous else call
    if streaming:
        assert (
            [event async for event in actual] if asynchronous else list(actual)
        ) == events
    else:
        assert actual is result
    handle.track_inference.assert_called_once()
    data = handle.track_inference.call_args.kwargs
    assert data["success"] == (status != "failed")
    assert data["input_meta"].char_count == 11
    output = data["output_meta"]
    assert (
        output.tokens_in,
        output.tokens_out,
        output.cached_input_tokens,
        output.reasoning_tokens_out,
    ) == (12, 4, 3, 2)
    assert output.stop_reason == (
        "max_output_tokens" if status == "incomplete" else status
    )
    assert data["api_meta"].resolved_model_id == "resolved"


def test_ttft_and_once(monkeypatch):
    handle = MagicMock()
    on_chunk, on_done = make_responses_stream_callbacks(handle, [], 0)
    on_chunk(SimpleNamespace(type="response.created"))
    monkeypatch.setattr("wildedge.integrations.openai.elapsed_ms", lambda _: 42)
    on_chunk(SimpleNamespace(type="response.output_text.delta", delta="hi"))
    on_chunk(SimpleNamespace(type="response.completed", response=response()))
    on_done(100, 1)
    on_done(101, 1)
    handle.track_inference.assert_called_once()
    assert (
        handle.track_inference.call_args.kwargs["output_meta"].time_to_first_token_ms
        == 42
    )


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_error_and_closed_client(asynchronous):
    error = TimeoutError("timeout")
    original = (AsyncMock if asynchronous else MagicMock)(side_effect=error)
    resource = SimpleNamespace(create=original)
    handle = MagicMock()
    client = SimpleNamespace(
        closed=False, register_model=MagicMock(return_value=handle)
    )
    (wrap_async_responses if asynchronous else wrap_sync_responses)(
        resource, "openai", lambda: client
    )

    async def invoke():
        call = resource.create(model="model", input="hello")
        return await call if asynchronous else call

    with pytest.raises(TimeoutError) as caught:
        await invoke()
    assert caught.value is error
    handle.track_error.assert_called_once()
    handle.track_inference.assert_not_called()
    client.closed = True
    original.side_effect = None
    original.return_value = "unchanged"
    call = resource.create(model="model", input="hello")
    assert (await call if asynchronous else call) == "unchanged"
    assert client.register_model.call_count == 1


def test_unfinished_and_error_streams():
    handle = MagicMock()
    on_chunk, on_done = make_responses_stream_callbacks(handle, [], 0)
    on_chunk(SimpleNamespace(type="response.created"))
    on_done(100, 1)
    handle.track_inference.assert_not_called()
    on_chunk(SimpleNamespace(type="error", code="server_error"))
    on_done(100, 1)
    handle.track_error.assert_called_once()
    handle.track_inference.assert_not_called()
