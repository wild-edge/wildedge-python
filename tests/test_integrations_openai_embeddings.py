from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from wildedge.integrations.openai import wrap_async_embeddings, wrap_sync_embeddings


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("value", "batch_size", "chars"),
    [
        ("hello", 1, 5),
        (["hello", "world"], 2, 10),
        ([1, 2, 3], 1, None),
        ([[1, 2], [3]], 2, None),
    ],
)
async def test_metadata_and_passthrough(asynchronous, value, batch_size, chars):
    result = SimpleNamespace(
        model="resolved-model",
        usage=SimpleNamespace(prompt_tokens=7),
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])],
    )
    resource = SimpleNamespace(
        create=(AsyncMock if asynchronous else MagicMock)(return_value=result)
    )
    handle = MagicMock()
    client = SimpleNamespace(
        closed=False, register_model=MagicMock(return_value=handle)
    )
    (wrap_async_embeddings if asynchronous else wrap_sync_embeddings)(
        resource, "openai", lambda: client
    )
    call = resource.create(model="model", input=value)
    assert (await call if asynchronous else call) is result
    handle.track_inference.assert_called_once()
    event = handle.track_inference.call_args.kwargs
    assert event["batch_size"] == batch_size
    assert event["input_meta"].token_count == 7
    assert event["input_meta"].char_count == chars
    assert event["output_meta"].tokens_in == 7
    assert event["output_meta"].dimensions == 3
    assert event["api_meta"].resolved_model_id == "resolved-model"
    assert "hello" not in str(event)
    client.closed = True
    call = resource.create(model="model", input=value)
    assert (await call if asynchronous else call) is result
    assert handle.track_inference.call_count == 1


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_error_and_telemetry_failure(asynchronous):
    original = (AsyncMock if asynchronous else MagicMock)(
        side_effect=TimeoutError("timeout")
    )
    resource = SimpleNamespace(create=original)
    handle = MagicMock()
    client = SimpleNamespace(
        closed=False, register_model=MagicMock(return_value=handle)
    )
    (wrap_async_embeddings if asynchronous else wrap_sync_embeddings)(
        resource, "openai", lambda: client
    )

    async def invoke():
        call = resource.create(model="model", input="hello")
        return await call if asynchronous else call

    with pytest.raises(TimeoutError, match="timeout"):
        await invoke()
    handle.track_error.assert_called_once()
    handle.track_inference.assert_not_called()
    result = SimpleNamespace(model="model", usage=None, data=[])
    original.side_effect = None
    original.return_value = result
    handle.track_inference.side_effect = RuntimeError("telemetry broken")
    assert await invoke() is result


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_input_iterator_is_not_consumed(asynchronous):
    def values():
        raise AssertionError("instrumentation must not iterate inputs")
        yield "unreachable"

    result = SimpleNamespace(model="model", usage=None, data=[])
    resource = SimpleNamespace(
        create=(AsyncMock if asynchronous else MagicMock)(return_value=result)
    )
    handle = MagicMock()
    client = SimpleNamespace(
        closed=False, register_model=MagicMock(return_value=handle)
    )
    (wrap_async_embeddings if asynchronous else wrap_sync_embeddings)(
        resource, "openai", lambda: client
    )
    call = resource.create(model="model", input=values())
    assert (await call if asynchronous else call) is result
    event = handle.track_inference.call_args.kwargs
    assert event["batch_size"] is None
    assert event["output_meta"].dimensions is None
