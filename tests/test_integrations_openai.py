"""Tests for the OpenAI / OpenRouter integration."""

from __future__ import annotations

import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import wildedge.integrations.openai as openai_mod
from wildedge.integrations.openai import (
    OpenAIExtractor,
    build_api_meta,
    build_input_meta,
    build_output_meta,
    source_from_base_url,
    wrap_async_completions,
    wrap_sync_completions,
)
from wildedge.model import ModelHandle, ModelInfo

# ---------------------------------------------------------------------------
# Fake objects — no openai library required
# ---------------------------------------------------------------------------


class FakePromptDetails:
    cached_tokens = 5


class FakeCompletionDetails:
    reasoning_tokens = 3


class FakeUsage:
    prompt_tokens = 10
    completion_tokens = 20
    prompt_tokens_details = FakePromptDetails()
    completion_tokens_details = FakeCompletionDetails()


class FakeUsageNoDetails:
    prompt_tokens = 10
    completion_tokens = 20
    prompt_tokens_details = None
    completion_tokens_details = None


class FakeChoice:
    finish_reason = "stop"


class FakeResponse:
    model = "gpt-4o-2024-08-06"
    system_fingerprint = "fp_abc123"
    service_tier = "default"
    usage = FakeUsage()
    choices = [FakeChoice()]


class FakeResponseNoUsage:
    model = None
    system_fingerprint = None
    service_tier = None
    usage = None
    choices = []


class FakeCompletions:
    def __init__(self, response=None):
        self._response = response or FakeResponse()

    def create(self, *args, **kwargs):
        return self._response


class FakeAsyncCompletions:
    def __init__(self, response=None):
        self._response = response or FakeResponse()

    async def create(self, *args, **kwargs):
        return self._response


# Named "OpenAI" / "AsyncOpenAI" so can_handle sees the right type name.
class OpenAI:
    def __init__(self, base_url="https://api.openai.com/v1", api_key=None):
        self.base_url = base_url


class AsyncOpenAI:
    def __init__(self, base_url="https://api.openai.com/v1", api_key=None):
        self.base_url = base_url


def make_handle(publish_spy) -> ModelHandle:
    info = ModelInfo(
        model_name="test",
        model_version="1.0",
        model_source="openai",
        model_format="api",
    )
    return ModelHandle(model_id="gpt-4o", info=info, publish=publish_spy)


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
# source_from_base_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        (None, "openai"),
        ("", "openai"),
        ("https://openrouter.ai/api/v1", "openrouter"),
        ("https://api.openai.com/v1", "openai"),
        ("https://api.together.xyz/v1", "api.together.xyz"),
        ("https://localhost:11434/v1", "localhost"),
    ],
)
def test_source_from_base_url(url, expected):
    assert source_from_base_url(url) == expected


# ---------------------------------------------------------------------------
# build_input_meta
# ---------------------------------------------------------------------------


def test_build_input_meta_picks_last_user_message():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello world"},
    ]
    meta = build_input_meta(messages, tokens_in=5)
    assert meta is not None
    assert meta.char_count == len("Hello world")
    assert meta.word_count == 2
    assert meta.token_count == 5
    assert meta.prompt_type == "chat"


def test_build_input_meta_no_user_message_returns_none():
    assert (
        build_input_meta([{"role": "system", "content": "sys"}], tokens_in=None) is None
    )


def test_build_input_meta_empty_messages_returns_none():
    assert build_input_meta([], tokens_in=None) is None


def test_build_input_meta_non_string_content_returns_none():
    assert (
        build_input_meta(
            [{"role": "user", "content": [{"type": "image_url"}]}], tokens_in=None
        )
        is None
    )


# ---------------------------------------------------------------------------
# build_output_meta
# ---------------------------------------------------------------------------


def test_build_output_meta_extracts_tokens_and_stop_reason():
    meta = build_output_meta(FakeResponse(), duration_ms=500)
    assert meta is not None
    assert meta.tokens_in == 10
    assert meta.tokens_out == 20
    assert meta.stop_reason == "stop"
    assert meta.tokens_per_second == pytest.approx(40.0)


def test_build_output_meta_extracts_cached_and_reasoning_tokens():
    meta = build_output_meta(FakeResponse(), duration_ms=500)
    assert meta is not None
    assert meta.cached_input_tokens == 5
    assert meta.reasoning_tokens_out == 3


def test_build_output_meta_none_cached_when_no_details():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
        choices=[SimpleNamespace(finish_reason="stop")],
    )
    meta = build_output_meta(response, duration_ms=100)
    assert meta is not None
    assert meta.cached_input_tokens is None
    assert meta.reasoning_tokens_out is None


def test_build_output_meta_none_when_no_usage():
    assert build_output_meta(FakeResponseNoUsage(), duration_ms=500) is None


def test_build_output_meta_zero_duration_gives_no_tps():
    meta = build_output_meta(FakeResponse(), duration_ms=0)
    assert meta is not None
    assert meta.tokens_per_second is None


# ---------------------------------------------------------------------------
# build_api_meta
# ---------------------------------------------------------------------------


def test_build_api_meta_extracts_all_fields():
    meta = build_api_meta(FakeResponse())
    assert meta is not None
    assert meta.resolved_model_id == "gpt-4o-2024-08-06"
    assert meta.system_fingerprint == "fp_abc123"
    assert meta.service_tier == "default"


def test_build_api_meta_none_when_all_fields_absent():
    assert build_api_meta(FakeResponseNoUsage()) is None


def test_build_api_meta_partial_fields():
    response = SimpleNamespace(
        model="gpt-4o", system_fingerprint=None, service_tier=None
    )
    meta = build_api_meta(response)
    assert meta is not None
    assert meta.resolved_model_id == "gpt-4o"
    assert meta.system_fingerprint is None


def test_build_api_meta_to_dict_omits_none_fields():
    response = SimpleNamespace(
        model="gpt-4o", system_fingerprint=None, service_tier=None
    )
    meta = build_api_meta(response)
    assert meta is not None
    d = meta.to_dict()
    assert "resolved_model_id" in d
    assert "system_fingerprint" not in d
    assert "service_tier" not in d


# ---------------------------------------------------------------------------
# OpenAIExtractor
# ---------------------------------------------------------------------------


class TestOpenAIExtractor:
    extractor = OpenAIExtractor()

    def test_can_handle_openai(self):
        assert self.extractor.can_handle(OpenAI())

    def test_can_handle_async_openai(self):
        assert self.extractor.can_handle(AsyncOpenAI())

    def test_can_handle_rejects_other_types(self):
        assert not self.extractor.can_handle(object())
        assert not self.extractor.can_handle("string")

    def test_extract_info_uses_override_model_id_and_source(self):
        obj = OpenAI(base_url="https://openrouter.ai/api/v1")
        model_id, info = self.extractor.extract_info(
            obj, {"id": "qwen/qwen3-235b", "source": "openrouter"}
        )
        assert model_id == "qwen/qwen3-235b"
        assert info.model_source == "openrouter"
        assert info.model_format == "api"
        assert info.model_name == "qwen/qwen3-235b"

    def test_extract_info_derives_source_from_base_url(self):
        obj = OpenAI(base_url="https://openrouter.ai/api/v1")
        _, info = self.extractor.extract_info(obj, {"id": "qwen/qwen3-235b"})
        assert info.model_source == "openrouter"

    def test_extract_info_returns_none_model_id_when_not_provided(self):
        model_id, _ = self.extractor.extract_info(OpenAI(), {})
        assert model_id is None

    def test_install_hooks_is_noop(self, publish_spy):
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(OpenAI(), handle)  # must not raise


# ---------------------------------------------------------------------------
# wrap_sync_completions
# ---------------------------------------------------------------------------


class TestWrapSyncCompletions:
    def setup(self, response=None, closed=False):
        completions = FakeCompletions(response)
        client = make_fake_client(closed=closed)
        wrap_sync_completions(completions, "openai", lambda: client)
        return completions, client

    def test_returns_response(self):
        completions, _ = self.setup()
        result = completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )
        assert isinstance(result, FakeResponse)

    def test_registers_model_on_first_call(self):
        completions, client = self.setup()
        completions.create(model="gpt-4o", messages=[])
        assert "gpt-4o" in client.handles

    def test_lazy_registration_only_once(self):
        completions, client = self.setup()
        completions.create(model="gpt-4o", messages=[])
        completions.create(model="gpt-4o", messages=[])
        assert len(client.handles) == 1

    def test_tracks_inference_with_token_counts(self):
        completions, client = self.setup()
        completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hello"}]
        )
        handle = client.handles["gpt-4o"]
        handle.track_inference.assert_called_once()
        kwargs = handle.track_inference.call_args.kwargs
        assert kwargs["input_modality"] == "text"
        assert kwargs["output_modality"] == "generation"
        assert kwargs["success"] is True
        assert kwargs["output_meta"].tokens_out == 20

    def test_tracks_api_meta(self):
        completions, client = self.setup()
        completions.create(model="gpt-4o", messages=[])
        kwargs = client.handles["gpt-4o"].track_inference.call_args.kwargs
        assert kwargs["api_meta"] is not None
        assert kwargs["api_meta"].resolved_model_id == "gpt-4o-2024-08-06"
        assert kwargs["api_meta"].system_fingerprint == "fp_abc123"

    def test_tracks_error_and_reraises(self):
        class ErrorCompletions:
            def create(self, *args, **kwargs):
                raise RuntimeError("api error")

        client = make_fake_client()
        completions = ErrorCompletions()
        wrap_sync_completions(completions, "openai", lambda: client)

        with pytest.raises(RuntimeError, match="api error"):
            completions.create(model="gpt-4o", messages=[])

        client.handles["gpt-4o"].track_error.assert_called_once()
        client.handles["gpt-4o"].track_inference.assert_not_called()

    def test_streaming_skips_tracking(self):
        completions, client = self.setup()
        completions.create(model="gpt-4o", messages=[], stream=True)
        if "gpt-4o" in client.handles:
            client.handles["gpt-4o"].track_inference.assert_not_called()

    def test_closed_client_passes_through(self):
        completions, client = self.setup(closed=True)
        result = completions.create(model="gpt-4o", messages=[])
        assert isinstance(result, FakeResponse)
        assert "gpt-4o" not in client.handles

    def test_different_models_get_separate_handles(self):
        completions, client = self.setup()
        completions.create(model="gpt-4o", messages=[])
        completions.create(model="gpt-4-turbo", messages=[])
        assert "gpt-4o" in client.handles
        assert "gpt-4-turbo" in client.handles


# ---------------------------------------------------------------------------
# wrap_async_completions
# ---------------------------------------------------------------------------


class TestWrapAsyncCompletions:
    def setup(self, response=None, closed=False):
        completions = FakeAsyncCompletions(response)
        client = make_fake_client(closed=closed)
        wrap_async_completions(completions, "openrouter", lambda: client)
        return completions, client

    async def test_returns_response(self):
        completions, _ = self.setup()
        result = await completions.create(
            model="qwen/qwen3-235b", messages=[{"role": "user", "content": "hi"}]
        )
        assert isinstance(result, FakeResponse)

    async def test_registers_model_on_first_call(self):
        completions, client = self.setup()
        await completions.create(model="qwen/qwen3-235b", messages=[])
        assert "qwen/qwen3-235b" in client.handles

    async def test_tracks_inference(self):
        completions, client = self.setup()
        await completions.create(
            model="qwen/qwen3-235b", messages=[{"role": "user", "content": "hello"}]
        )
        handle = client.handles["qwen/qwen3-235b"]
        handle.track_inference.assert_called_once()
        assert handle.track_inference.call_args.kwargs["output_meta"].tokens_out == 20

    async def test_tracks_error_and_reraises(self):
        class ErrorAsyncCompletions:
            async def create(self, *args, **kwargs):
                raise RuntimeError("timeout")

        client = make_fake_client()
        completions = ErrorAsyncCompletions()
        wrap_async_completions(completions, "openai", lambda: client)

        with pytest.raises(RuntimeError, match="timeout"):
            await completions.create(model="gpt-4o", messages=[])

        client.handles["gpt-4o"].track_error.assert_called_once()

    async def test_streaming_skips_tracking(self):
        completions, client = self.setup()
        await completions.create(model="qwen/qwen3-235b", messages=[], stream=True)
        if "qwen/qwen3-235b" in client.handles:
            client.handles["qwen/qwen3-235b"].track_inference.assert_not_called()


# ---------------------------------------------------------------------------
# install_auto_load_patch
# ---------------------------------------------------------------------------


def test_install_auto_load_patch_is_idempotent(monkeypatch):
    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            pass

    class FakeAsyncOpenAI:
        def __init__(self, *args, **kwargs):
            pass

    fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI, AsyncOpenAI=FakeAsyncOpenAI)
    monkeypatch.setattr(openai_mod, "_openai", fake_openai)
    monkeypatch.setattr(openai_mod, "_openai_patched", False)

    OpenAIExtractor.install_auto_load_patch(lambda: None)
    first_sync = fake_openai.OpenAI.__init__
    first_async = fake_openai.AsyncOpenAI.__init__

    OpenAIExtractor.install_auto_load_patch(lambda: None)
    assert fake_openai.OpenAI.__init__ is first_sync
    assert fake_openai.AsyncOpenAI.__init__ is first_async


def test_install_auto_load_patch_skips_when_openai_missing(monkeypatch):
    monkeypatch.setattr(openai_mod, "_openai", None)
    monkeypatch.setattr(openai_mod, "_openai_patched", False)

    OpenAIExtractor.install_auto_load_patch(lambda: None)
    assert not openai_mod._openai_patched


def test_install_auto_load_patch_wraps_new_client_instances(monkeypatch):
    class FakeCompletionsInner:
        def create(self, *args, **kwargs):
            return FakeResponseNoUsage()

    class FakeChat:
        completions = FakeCompletionsInner()

    class FakeOpenAI:
        base_url = "https://api.openai.com/v1"
        chat = FakeChat()

        def __init__(self, *args, **kwargs):
            pass

    class FakeAsyncOpenAI:
        base_url = "https://api.openai.com/v1"
        chat = FakeChat()

        def __init__(self, *args, **kwargs):
            pass

    fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI, AsyncOpenAI=FakeAsyncOpenAI)
    monkeypatch.setattr(openai_mod, "_openai", fake_openai)
    monkeypatch.setattr(openai_mod, "_openai_patched", False)

    client = make_fake_client()
    OpenAIExtractor.install_auto_load_patch(lambda: client)

    instance = FakeOpenAI()
    assert instance.chat.completions.create is not FakeCompletionsInner.create
