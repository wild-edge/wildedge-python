from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from wildedge.integrations.anthropic import make_anthropic_stream_callbacks
from wildedge.integrations.common import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
    stream_chunk_has_output,
)
from wildedge.integrations.gguf import make_gguf_stream_callbacks
from wildedge.integrations.openai import (
    make_openai_stream_callbacks,
    make_responses_stream_callbacks,
)
from wildedge.tracing import get_trace_context, trace_context


def wrapper(original, on_chunk=None, on_done=None):
    handle = MagicMock()
    done = on_done or MagicMock()
    return (
        SyncStreamWrapper(original, handle, time.perf_counter(), on_chunk, done),
        handle,
        done,
    )


def test_single_pass_completion_and_close():
    stream, handle, done = wrapper(iter([1, 2]))
    assert next(stream) == 1
    assert list(stream) == [2]
    assert list(stream) == []
    stream.close()
    done.assert_called_once()
    handle.track_error.assert_not_called()


def test_explicit_close_and_context_exit():
    original = MagicMock()
    original.__iter__.return_value = iter([1, 2])
    stream, _, done = wrapper(original)
    with stream:
        assert next(stream) == 1
    original.__exit__.assert_called_once()
    stream.close()
    done.assert_called_once()
    assert done.call_args.kwargs == {"outcome": "closed"}


def test_midstream_error_preserved_once():
    error = TimeoutError("broken stream")

    def chunks():
        yield 1
        raise error

    stream, handle, done = wrapper(chunks())
    assert next(stream) == 1
    with pytest.raises(TimeoutError) as caught:
        next(stream)
    assert caught.value is error
    stream.close()
    handle.track_error.assert_called_once()
    done.assert_not_called()


def test_callback_failures_do_not_break_iteration():
    stream, handle, _ = wrapper(
        iter([1, 2]),
        MagicMock(side_effect=ValueError()),
        MagicMock(side_effect=ValueError()),
    )
    assert list(stream) == [1, 2]
    handle.track_error.assert_not_called()


def test_correlation_uses_creation_context():
    seen = []
    with trace_context(trace_id="origin"):
        stream, _, _ = wrapper(
            iter([1]), on_done=lambda *_: seen.append(get_trace_context().trace_id)
        )
    with trace_context(trace_id="consumer"):
        list(stream)
    assert seen == ["origin"]


class AsyncChunks:
    def __init__(self, items, error=None):
        self.items = iter(items)
        self.error = error
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration:
            if self.error is not None:
                raise self.error from None
            raise StopAsyncIteration from None

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize(
    "error", [None, TimeoutError("timeout"), asyncio.CancelledError()]
)
async def test_async_lifecycle(error):
    original = AsyncChunks([1], error)
    handle, done = MagicMock(), MagicMock()
    stream = AsyncStreamWrapper(original, handle, time.perf_counter(), None, done)
    assert await anext(stream) == 1
    if error is None:
        assert [item async for item in stream] == []
        done.assert_called_once()
    else:
        with pytest.raises(type(error)) as caught:
            await anext(stream)
        assert caught.value is error
        if isinstance(error, asyncio.CancelledError):
            assert done.call_args.kwargs == {"outcome": "cancelled"}
        else:
            handle.track_error.assert_called_once()
            done.assert_not_called()
    await stream.aclose()
    assert original.closed
    assert [item async for item in stream] == []
    assert done.call_count + handle.track_error.call_count == 1


async def test_async_context_early_exit():
    original = AsyncChunks([1, 2])
    done = MagicMock()
    stream = AsyncStreamWrapper(original, MagicMock(), time.perf_counter(), None, done)
    async with stream:
        assert await anext(stream) == 1
    assert original.closed
    assert done.call_args.kwargs == {"outcome": "closed"}


@pytest.mark.parametrize(
    "factory",
    [
        lambda h: make_openai_stream_callbacks(h, []),
        lambda h: make_anthropic_stream_callbacks(h, []),
        lambda h: make_gguf_stream_callbacks(h, "hello"),
        lambda h: make_responses_stream_callbacks(h, [], time.perf_counter()),
    ],
)
@pytest.mark.parametrize("outcome", ["closed", "cancelled"])
def test_all_integrations_record_interruption(factory, outcome):
    handle = MagicMock()
    _, on_done = factory(handle)
    on_done(100, None, outcome=outcome)
    handle.track_inference.assert_called_once()
    event = handle.track_inference.call_args.kwargs
    assert event["success"] is False
    assert event["error_code"] == "STREAM_" + outcome.upper()


@pytest.mark.parametrize(
    ("chunk", "expected"),
    [
        ({"choices": [{"delta": {"role": "assistant"}}]}, False),
        ({"choices": [], "usage": {}}, False),
        ({"choices": [{"delta": {"content": "hello"}}]}, True),
        ({"choices": [{"text": "hello"}]}, True),
        (SimpleNamespace(type="message_start"), False),
        (
            SimpleNamespace(
                type="content_block_delta", delta=SimpleNamespace(text="hello")
            ),
            True,
        ),
        (SimpleNamespace(type="response.created"), False),
    ],
)
def test_first_output_detection(chunk, expected):
    assert stream_chunk_has_output(chunk) is expected


async def test_task_cancellation_while_waiting():
    entered = asyncio.Event()

    async def chunks():
        entered.set()
        await asyncio.Event().wait()
        yield 1

    handle, done = MagicMock(), MagicMock()
    stream = AsyncStreamWrapper(chunks(), handle, time.perf_counter(), None, done)
    task = asyncio.create_task(anext(stream))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await stream.aclose()
    done.assert_called_once()
    assert done.call_args.kwargs == {"outcome": "cancelled"}
    handle.track_error.assert_not_called()


def test_ttft_waits_for_output(monkeypatch):
    clock = [1]
    monkeypatch.setattr("wildedge.integrations.common.elapsed_ms", lambda _: clock[0])
    chunks = iter(
        [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "hello"}}]},
        ]
    )
    stream, _, done = wrapper(chunks)
    next(stream)
    clock[0] = 42
    next(stream)
    clock[0] = 100
    list(stream)
    assert done.call_args.args == (100, 42)


def test_underlying_close_error_is_preserved():
    original = MagicMock()
    original.close.side_effect = ValueError("close failed")
    stream, _, done = wrapper(original)
    with pytest.raises(ValueError, match="close failed"):
        stream.close()
    done.assert_called_once()
    assert done.call_args.kwargs == {"outcome": "closed"}


async def test_async_telemetry_and_context():
    seen = []
    with trace_context(trace_id="origin"):
        stream = AsyncStreamWrapper(
            AsyncChunks([1]),
            MagicMock(),
            time.perf_counter(),
            MagicMock(side_effect=ValueError()),
            lambda *_: seen.append(get_trace_context().trace_id),
        )
    with trace_context(trace_id="consumer"):
        assert [item async for item in stream] == [1]
    assert seen == ["origin"]


@pytest.mark.parametrize(
    "factory", [make_openai_stream_callbacks, make_anthropic_stream_callbacks]
)
def test_remote_eof_without_completion_is_incomplete(factory):
    handle = MagicMock()
    callbacks = factory(handle, [])
    stream = SyncStreamWrapper(iter([]), handle, time.perf_counter(), *callbacks)
    list(stream)
    stream.close()
    handle.track_inference.assert_called_once()
    event = handle.track_inference.call_args.kwargs
    assert event["success"] is False
    assert event["error_code"] == "STREAM_INCOMPLETE"
