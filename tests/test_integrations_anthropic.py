"""Tests for the Anthropic SDK integration."""

from __future__ import annotations

import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import wildedge.integrations.anthropic as anthropic_mod
from wildedge.integrations.anthropic import (
    AnthropicExtractor,
    build_api_meta,
    build_input_meta,
    build_output_meta,
    wrap_async_messages,
    wrap_sync_messages,
)
from wildedge.integrations.common import AsyncStreamWrapper, SyncStreamWrapper
from wildedge.model import ModelHandle, ModelInfo

# ---------------------------------------------------------------------------
# Fake objects — no anthropic library required
# ---------------------------------------------------------------------------


class FakeUsage:
    input_tokens = 10
    output_tokens = 20
    cache_read_input_tokens = 4


class FakeUsageNoCache:
    input_tokens = 10
    output_tokens = 20
    cache_read_input_tokens = None


class FakeResponse:
    model = "claude-opus-4-6-20251101"
    stop_reason = "end_turn"
    usage = FakeUsage()


class FakeResponseNoUsage:
    model = None
    stop_reason = None
    usage = None


class FakeMessages:
    def __init__(self, response=None):
        self._response = response or FakeResponse()

    def create(self, *args, **kwargs):
        return self._response


class FakeAsyncMessages:
    def __init__(self, response=None):
        self._response = response or FakeResponse()

    async def create(self, *args, **kwargs):
        return self._response


class RawMessageStartEvent:
    def __init__(self, input_tokens=10, cached=4, model="claude-opus-4-6"):
        usage = SimpleNamespace(
            input_tokens=input_tokens, cache_read_input_tokens=cached
        )
        self.message = SimpleNamespace(usage=usage, model=model)


class RawMessageDeltaEvent:
    def __init__(self, output_tokens=20, stop_reason="end_turn"):
        self.usage = SimpleNamespace(output_tokens=output_tokens)
        self.delta = SimpleNamespace(stop_reason=stop_reason)


class RawContentBlockDeltaEvent:
    def __init__(self, text="hi"):
        self.delta = SimpleNamespace(type="text_delta", text=text)


def make_message_start_event(input_tokens=10, cached=4, model="claude-opus-4-6"):
    return RawMessageStartEvent(input_tokens=input_tokens, cached=cached, model=model)


def make_message_delta_event(output_tokens=20, stop_reason="end_turn"):
    return RawMessageDeltaEvent(output_tokens=output_tokens, stop_reason=stop_reason)


def make_content_block_delta_event(text="hi"):
    return RawContentBlockDeltaEvent(text=text)


class FakeStreamingMessages:
    def __init__(self, events):
        self._events = events

    def create(self, *args, **kwargs):
        if kwargs.get("stream"):
            return iter(self._events)
        return FakeResponse()


class FakeAsyncStreamingMessages:
    def __init__(self, events):
        self._events = events

    async def create(self, *args, **kwargs):
        if kwargs.get("stream"):
            return FakeAsyncIterator(self._events)
        return FakeResponse()


class FakeAsyncIterator:
    def __init__(self, items):
        self._iter = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


# Named "Anthropic" / "AsyncAnthropic" so can_handle sees the right type name.
class Anthropic:
    def __init__(self, api_key=None):
        pass


class AsyncAnthropic:
    def __init__(self, api_key=None):
        pass


def make_handle(publish_spy) -> ModelHandle:
    info = ModelInfo(
        model_name="test",
        model_version="1.0",
        model_source="anthropic",
        model_format="api",
    )
    return ModelHandle(model_id="claude-opus-4-6", info=info, publish=publish_spy)


def make_fake_client(closed=False):
    client = SimpleNamespace(closed=closed, handles={})

    def register_model(obj, *, model_id=None, source=None, **kwargs):
        if model_id not in client.handles:
            client.handles[model_id] = SimpleNamespace(
                model_id=model_id,
                track_inference=MagicMock(),
                track_error=MagicMock(),
            )
        return client.handles[model_id]

    client.register_model = register_model
    return client


# ---------------------------------------------------------------------------
# build_input_meta
# ---------------------------------------------------------------------------


def test_build_input_meta_picks_last_user_message():
    messages = [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "Reply"},
        {"role": "user", "content": "Hello world"},
    ]
    meta = build_input_meta(messages, tokens_in=5)
    assert meta is not None
    assert meta.char_count == len("Hello world")
    assert meta.word_count == 2
    assert meta.token_count == 5
    assert meta.prompt_type == "chat"


def test_build_input_meta_content_block_list():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello world"},
                {"type": "image", "source": {}},
            ],
        }
    ]
    meta = build_input_meta(messages, tokens_in=None)
    assert meta is not None
    assert meta.char_count == len("Hello world")
    assert meta.word_count == 2


def test_build_input_meta_no_user_message_returns_none():
    assert (
        build_input_meta([{"role": "assistant", "content": "Hi"}], tokens_in=None)
        is None
    )


def test_build_input_meta_empty_messages_returns_none():
    assert build_input_meta([], tokens_in=None) is None


def test_build_input_meta_empty_content_returns_none():
    assert build_input_meta([{"role": "user", "content": ""}], tokens_in=None) is None


# ---------------------------------------------------------------------------
# build_output_meta
# ---------------------------------------------------------------------------


def test_build_output_meta_extracts_tokens_and_stop_reason():
    meta = build_output_meta(FakeResponse(), duration_ms=500)
    assert meta is not None
    assert meta.tokens_in == 10
    assert meta.tokens_out == 20
    assert meta.stop_reason == "end_turn"
    assert meta.tokens_per_second == pytest.approx(40.0)


def test_build_output_meta_extracts_cached_tokens():
    meta = build_output_meta(FakeResponse(), duration_ms=500)
    assert meta is not None
    assert meta.cached_input_tokens == 4


def test_build_output_meta_none_cached_when_absent():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=5, cache_read_input_tokens=None
        ),
        stop_reason="end_turn",
    )
    meta = build_output_meta(response, duration_ms=100)
    assert meta is not None
    assert meta.cached_input_tokens is None


def test_build_output_meta_none_when_no_usage():
    assert build_output_meta(FakeResponseNoUsage(), duration_ms=500) is None


def test_build_output_meta_zero_duration_gives_no_tps():
    meta = build_output_meta(FakeResponse(), duration_ms=0)
    assert meta is not None
    assert meta.tokens_per_second is None


# ---------------------------------------------------------------------------
# build_api_meta
# ---------------------------------------------------------------------------


def test_build_api_meta_extracts_resolved_model_id():
    meta = build_api_meta(FakeResponse())
    assert meta is not None
    assert meta.resolved_model_id == "claude-opus-4-6-20251101"


def test_build_api_meta_none_when_model_absent():
    assert build_api_meta(FakeResponseNoUsage()) is None


def test_build_api_meta_to_dict_contains_resolved_model_id():
    meta = build_api_meta(FakeResponse())
    assert meta is not None
    assert "resolved_model_id" in meta.to_dict()


# ---------------------------------------------------------------------------
# AnthropicExtractor
# ---------------------------------------------------------------------------


class TestAnthropicExtractor:
    extractor = AnthropicExtractor()

    def test_can_handle_anthropic(self):
        assert self.extractor.can_handle(Anthropic())

    def test_can_handle_async_anthropic(self):
        assert self.extractor.can_handle(AsyncAnthropic())

    def test_can_handle_rejects_other_types(self):
        assert not self.extractor.can_handle(object())
        assert not self.extractor.can_handle("string")

    def test_extract_info_uses_override_model_id(self):
        obj = Anthropic()
        model_id, info = self.extractor.extract_info(obj, {"id": "claude-opus-4-6"})
        assert model_id == "claude-opus-4-6"
        assert info.model_name == "claude-opus-4-6"
        assert info.model_format == "api"

    def test_extract_info_source_defaults_to_anthropic(self):
        _, info = self.extractor.extract_info(Anthropic(), {"id": "claude-opus-4-6"})
        assert info.model_source == "anthropic"

    def test_extract_info_returns_none_model_id_when_not_provided(self):
        model_id, _ = self.extractor.extract_info(Anthropic(), {})
        assert model_id is None

    def test_install_hooks_is_noop(self, publish_spy):
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(Anthropic(), handle)  # must not raise


# ---------------------------------------------------------------------------
# wrap_sync_messages
# ---------------------------------------------------------------------------


class TestWrapSyncMessages:
    def setup(self, response=None, closed=False):
        messages = FakeMessages(response)
        client = make_fake_client(closed=closed)
        wrap_sync_messages(messages, lambda: client)
        return messages, client

    def test_returns_response(self):
        messages, _ = self.setup()
        result = messages.create(
            model="claude-opus-4-6", messages=[{"role": "user", "content": "hi"}]
        )
        assert isinstance(result, FakeResponse)

    def test_registers_model_on_first_call(self):
        messages, client = self.setup()
        messages.create(model="claude-opus-4-6", messages=[])
        assert "claude-opus-4-6" in client.handles

    def test_lazy_registration_only_once(self):
        messages, client = self.setup()
        messages.create(model="claude-opus-4-6", messages=[])
        messages.create(model="claude-opus-4-6", messages=[])
        assert len(client.handles) == 1

    def test_tracks_inference_with_token_counts(self):
        messages, client = self.setup()
        messages.create(
            model="claude-opus-4-6", messages=[{"role": "user", "content": "hello"}]
        )
        handle = client.handles["claude-opus-4-6"]
        handle.track_inference.assert_called_once()
        kwargs = handle.track_inference.call_args.kwargs
        assert kwargs["input_modality"] == "text"
        assert kwargs["output_modality"] == "generation"
        assert kwargs["success"] is True
        assert kwargs["output_meta"].tokens_out == 20

    def test_tracks_api_meta(self):
        messages, client = self.setup()
        messages.create(model="claude-opus-4-6", messages=[])
        kwargs = client.handles["claude-opus-4-6"].track_inference.call_args.kwargs
        assert kwargs["api_meta"] is not None
        assert kwargs["api_meta"].resolved_model_id == "claude-opus-4-6-20251101"

    def test_tracks_error_and_reraises(self):
        class ErrorMessages:
            def create(self, *args, **kwargs):
                raise RuntimeError("api error")

        client = make_fake_client()
        messages = ErrorMessages()
        wrap_sync_messages(messages, lambda: client)

        with pytest.raises(RuntimeError, match="api error"):
            messages.create(model="claude-opus-4-6", messages=[])

        client.handles["claude-opus-4-6"].track_error.assert_called_once()
        client.handles["claude-opus-4-6"].track_inference.assert_not_called()

    def test_streaming_returns_sync_stream_wrapper(self):
        events = [make_content_block_delta_event("hi"), make_message_delta_event()]
        messages = FakeStreamingMessages(events)
        client = make_fake_client()
        wrap_sync_messages(messages, lambda: client)
        result = messages.create(model="claude-opus-4-6", messages=[], stream=True)
        assert isinstance(result, SyncStreamWrapper)

    def test_streaming_records_inference_on_exhaustion(self):
        events = [
            make_message_start_event(
                input_tokens=10, cached=4, model="claude-opus-4-6-20251101"
            ),
            make_content_block_delta_event("Hello world"),
            make_message_delta_event(output_tokens=20, stop_reason="end_turn"),
        ]
        messages = FakeStreamingMessages(events)
        client = make_fake_client()
        wrap_sync_messages(messages, lambda: client)
        stream = messages.create(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        list(stream)
        handle = client.handles["claude-opus-4-6"]
        handle.track_inference.assert_called_once()
        kwargs = handle.track_inference.call_args.kwargs
        assert kwargs["output_meta"].time_to_first_token_ms is not None
        assert kwargs["output_meta"].stop_reason == "end_turn"
        assert kwargs["output_meta"].tokens_out == 20
        assert kwargs["input_modality"] == "text"
        assert kwargs["success"] is True

    def test_streaming_captures_token_counts_from_events(self):
        events = [
            make_message_start_event(input_tokens=8, cached=2),
            make_message_delta_event(output_tokens=15, stop_reason="end_turn"),
        ]
        messages = FakeStreamingMessages(events)
        client = make_fake_client()
        wrap_sync_messages(messages, lambda: client)
        list(messages.create(model="claude-opus-4-6", messages=[], stream=True))
        out = client.handles["claude-opus-4-6"].track_inference.call_args.kwargs[
            "output_meta"
        ]
        assert out.tokens_in == 8
        assert out.tokens_out == 15
        assert out.cached_input_tokens == 2

    def test_streaming_error_during_iteration_tracks_error(self):
        def bad_iter():
            yield make_content_block_delta_event("hi")
            raise RuntimeError("stream error")

        class ErrorStreamMessages:
            def create(self, *args, **kwargs):
                return bad_iter()

        client = make_fake_client()
        messages = ErrorStreamMessages()
        wrap_sync_messages(messages, lambda: client)
        stream = messages.create(model="claude-opus-4-6", messages=[], stream=True)
        with pytest.raises(RuntimeError, match="stream error"):
            list(stream)
        client.handles["claude-opus-4-6"].track_error.assert_called_once()
        client.handles["claude-opus-4-6"].track_inference.assert_not_called()

    def test_closed_client_passes_through(self):
        messages, client = self.setup(closed=True)
        result = messages.create(model="claude-opus-4-6", messages=[])
        assert isinstance(result, FakeResponse)
        assert "claude-opus-4-6" not in client.handles

    def test_different_models_get_separate_handles(self):
        messages, client = self.setup()
        messages.create(model="claude-opus-4-6", messages=[])
        messages.create(model="claude-haiku-4-5", messages=[])
        assert "claude-opus-4-6" in client.handles
        assert "claude-haiku-4-5" in client.handles


# ---------------------------------------------------------------------------
# wrap_async_messages
# ---------------------------------------------------------------------------


class TestWrapAsyncMessages:
    def setup(self, response=None, closed=False):
        messages = FakeAsyncMessages(response)
        client = make_fake_client(closed=closed)
        wrap_async_messages(messages, lambda: client)
        return messages, client

    async def test_returns_response(self):
        messages, _ = self.setup()
        result = await messages.create(
            model="claude-opus-4-6", messages=[{"role": "user", "content": "hi"}]
        )
        assert isinstance(result, FakeResponse)

    async def test_registers_model_on_first_call(self):
        messages, client = self.setup()
        await messages.create(model="claude-opus-4-6", messages=[])
        assert "claude-opus-4-6" in client.handles

    async def test_tracks_inference(self):
        messages, client = self.setup()
        await messages.create(
            model="claude-opus-4-6", messages=[{"role": "user", "content": "hello"}]
        )
        handle = client.handles["claude-opus-4-6"]
        handle.track_inference.assert_called_once()
        assert handle.track_inference.call_args.kwargs["output_meta"].tokens_out == 20

    async def test_tracks_error_and_reraises(self):
        class ErrorAsyncMessages:
            async def create(self, *args, **kwargs):
                raise RuntimeError("timeout")

        client = make_fake_client()
        messages = ErrorAsyncMessages()
        wrap_async_messages(messages, lambda: client)

        with pytest.raises(RuntimeError, match="timeout"):
            await messages.create(model="claude-opus-4-6", messages=[])

        client.handles["claude-opus-4-6"].track_error.assert_called_once()

    async def test_streaming_returns_async_stream_wrapper(self):
        events = [make_content_block_delta_event("hi"), make_message_delta_event()]
        messages = FakeAsyncStreamingMessages(events)
        client = make_fake_client()
        wrap_async_messages(messages, lambda: client)
        result = await messages.create(
            model="claude-opus-4-6", messages=[], stream=True
        )
        assert isinstance(result, AsyncStreamWrapper)

    async def test_streaming_records_inference_on_exhaustion(self):
        events = [
            make_message_start_event(input_tokens=10, model="claude-opus-4-6-20251101"),
            make_content_block_delta_event("Hello world"),
            make_message_delta_event(output_tokens=20, stop_reason="end_turn"),
        ]
        messages = FakeAsyncStreamingMessages(events)
        client = make_fake_client()
        wrap_async_messages(messages, lambda: client)
        stream = await messages.create(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        async for _ in stream:
            pass
        handle = client.handles["claude-opus-4-6"]
        handle.track_inference.assert_called_once()
        kwargs = handle.track_inference.call_args.kwargs
        assert kwargs["output_meta"].time_to_first_token_ms is not None
        assert kwargs["output_meta"].stop_reason == "end_turn"
        assert kwargs["success"] is True


# ---------------------------------------------------------------------------
# install_auto_load_patch
# ---------------------------------------------------------------------------


def test_install_auto_load_patch_is_idempotent(monkeypatch):
    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            pass

    class FakeAsyncAnthropic:
        def __init__(self, *args, **kwargs):
            pass

    fake_anthropic = types.SimpleNamespace(
        Anthropic=FakeAnthropic, AsyncAnthropic=FakeAsyncAnthropic
    )
    monkeypatch.setattr(anthropic_mod, "_anthropic", fake_anthropic)
    monkeypatch.setattr(anthropic_mod, "_anthropic_patched", False)

    AnthropicExtractor.install_auto_load_patch(lambda: None)
    first_sync = fake_anthropic.Anthropic.__init__
    first_async = fake_anthropic.AsyncAnthropic.__init__

    AnthropicExtractor.install_auto_load_patch(lambda: None)
    assert fake_anthropic.Anthropic.__init__ is first_sync
    assert fake_anthropic.AsyncAnthropic.__init__ is first_async


def test_install_auto_load_patch_skips_when_anthropic_missing(monkeypatch):
    monkeypatch.setattr(anthropic_mod, "_anthropic", None)
    monkeypatch.setattr(anthropic_mod, "_anthropic_patched", False)

    AnthropicExtractor.install_auto_load_patch(lambda: None)
    assert not anthropic_mod._anthropic_patched


def test_install_auto_load_patch_wraps_new_client_instances(monkeypatch):
    class FakeMessagesInner:
        def create(self, *args, **kwargs):
            return FakeResponseNoUsage()

    class FakeAnthropic:
        messages = FakeMessagesInner()

        def __init__(self, *args, **kwargs):
            pass

    class FakeAsyncAnthropic:
        messages = FakeMessagesInner()

        def __init__(self, *args, **kwargs):
            pass

    fake_anthropic = types.SimpleNamespace(
        Anthropic=FakeAnthropic, AsyncAnthropic=FakeAsyncAnthropic
    )
    monkeypatch.setattr(anthropic_mod, "_anthropic", fake_anthropic)
    monkeypatch.setattr(anthropic_mod, "_anthropic_patched", False)

    client = make_fake_client()
    AnthropicExtractor.install_auto_load_patch(lambda: client)

    instance = FakeAnthropic()
    assert instance.messages.create is not FakeMessagesInner.create
