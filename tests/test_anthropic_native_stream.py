import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from wildedge.integrations.anthropic import AnthropicStreamManager, wrap_native_stream


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_request_failure_preserves_exception_when_telemetry_fails(asynchronous):
    original_error = TimeoutError("request timeout")
    handle = MagicMock()
    handle.track_error.side_effect = ValueError("telemetry failed")
    original = MagicMock()
    original.__enter__.side_effect = original_error
    original.__aenter__.side_effect = original_error
    manager = AnthropicStreamManager(original, handle, [])

    async def enter():
        return await manager.__aenter__() if asynchronous else manager.__enter__()

    with pytest.raises(TimeoutError) as caught:
        await enter()
    assert caught.value is original_error
    handle.track_error.assert_called_once()


async def test_cancelled_before_entry_preserves_cancellation():
    original = MagicMock()
    original.__aenter__.side_effect = asyncio.CancelledError()
    handle = MagicMock()
    handle.track_inference.side_effect = ValueError("telemetry failed")
    manager = AnthropicStreamManager(original, handle, [])
    with pytest.raises(asyncio.CancelledError):
        await manager.__aenter__()
    assert handle.track_inference.call_args.kwargs["error_code"] == "STREAM_CANCELLED"


async def test_cancelled_inside_native_context():
    async def chunks():
        yield SimpleNamespace(type="message_start")

    native = SimpleNamespace(_iterator=chunks(), close=AsyncMock())
    original = MagicMock()
    original.__aenter__.return_value = native

    async def exit_context(*args):
        await native.close()

    original.__aexit__.side_effect = exit_context
    handle = MagicMock()
    manager = AnthropicStreamManager(original, handle, [])

    async def consume():
        async with manager as stream:
            assert stream is native
            await anext(stream._iterator)
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await consume()
    handle.track_inference.assert_called_once()
    assert handle.track_inference.call_args.kwargs["error_code"] == "STREAM_CANCELLED"


def test_closed_client_returns_original_manager():
    original_manager = object()
    resource = SimpleNamespace(stream=MagicMock(return_value=original_manager))
    wrap_native_stream(resource, lambda: SimpleNamespace(closed=True))
    assert resource.stream(model="model") is original_manager
