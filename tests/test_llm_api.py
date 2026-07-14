from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import wildedge
from wildedge import constants
from wildedge.defaults import set_default_client


class FakeHandle:
    def __init__(self):
        self.inferences: list[dict] = []
        self.errors: list[dict] = []

    def track_inference(self, **kwargs):
        self.inferences.append(kwargs)

    def track_error(self, error_code=None, *, error_message=None, **kwargs):
        self.errors.append({"error_code": error_code, "error_message": error_message})


class FakeClient:
    def __init__(self):
        self.handle = FakeHandle()
        self.registered: list[dict] = []

    def register_model(self, model_obj, **kwargs):
        self.registered.append({"model_obj": model_obj, **kwargs})
        return self.handle


@pytest.fixture
def client():
    fake = FakeClient()
    set_default_client(fake)
    return fake


def test_openai_shape_usage(client):
    with wildedge.llm_api(model="openai/gpt-4o-mini", provider="openrouter") as call:
        call.usage(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 25},
                "completion_tokens_details": {"reasoning_tokens": 10},
            }
        )
        call.stop_reason = "stop"

    assert client.registered[0]["model_id"] == "openai/gpt-4o-mini"
    assert client.registered[0]["source"] == "openrouter"
    event = client.handle.inferences[0]
    assert event["input_modality"] == "text"
    assert event["output_modality"] == "generation"
    assert event["success"] is True
    meta = event["output_meta"]
    assert meta.tokens_in == 100
    assert meta.tokens_out == 50
    assert meta.cached_input_tokens == 25
    assert meta.reasoning_tokens_out == 10
    assert meta.stop_reason == "stop"


def test_anthropic_shape_usage(client):
    with wildedge.llm_api(model="claude-sonnet-4-5", provider="anthropic") as call:
        call.usage(
            {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 3}
        )

    meta = client.handle.inferences[0]["output_meta"]
    assert meta.tokens_in == 10
    assert meta.tokens_out == 5
    assert meta.cached_input_tokens == 3


def test_attribute_object_usage(client):
    usage = SimpleNamespace(
        prompt_tokens=7,
        completion_tokens=3,
        prompt_tokens_details=SimpleNamespace(cached_tokens=2),
        completion_tokens_details=None,
    )
    with wildedge.llm_api(model="m", provider="p") as call:
        call.usage(usage)

    meta = client.handle.inferences[0]["output_meta"]
    assert meta.tokens_in == 7
    assert meta.tokens_out == 3
    assert meta.cached_input_tokens == 2


def test_explicit_field_usage_wins_over_payload(client):
    with wildedge.llm_api(model="m", provider="p") as call:
        call.usage({"prompt_tokens": 1}, tokens_in=11, tokens_out=22)

    meta = client.handle.inferences[0]["output_meta"]
    assert meta.tokens_in == 11
    assert meta.tokens_out == 22


def test_response_openai_shape(client):
    with wildedge.llm_api(model="m", provider="p") as call:
        call.response(
            {
                "model": "gpt-4o-mini-2024-07-18",
                "system_fingerprint": "fp_1",
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                "choices": [{"finish_reason": "length", "message": {}}],
            }
        )

    event = client.handle.inferences[0]
    assert event["output_meta"].stop_reason == "length"
    assert event["output_meta"].tokens_in == 4
    assert event["api_meta"].resolved_model_id == "gpt-4o-mini-2024-07-18"
    assert event["api_meta"].system_fingerprint == "fp_1"


def test_response_anthropic_shape(client):
    with wildedge.llm_api(model="m", provider="anthropic") as call:
        call.response(
            {
                "model": "claude-sonnet-4-5",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 9, "output_tokens": 1},
            }
        )

    meta = client.handle.inferences[0]["output_meta"]
    assert meta.stop_reason == "end_turn"
    assert meta.tokens_in == 9


def test_exception_records_error_not_inference(client):
    with pytest.raises(ValueError):
        with wildedge.llm_api(model="m", provider="p"):
            raise ValueError("bad payload")

    assert client.handle.inferences == []
    assert client.handle.errors == [
        {"error_code": "ValueError", "error_message": "bad payload"}
    ]


def test_prompt_input_meta(client):
    with wildedge.llm_api(
        model="m", provider="p", prompt="a knight with two swords"
    ) as call:
        call.usage(tokens_in=6)

    meta = client.handle.inferences[0]["input_meta"]
    assert meta.char_count == len("a knight with two swords")
    assert meta.word_count == 5
    assert meta.token_count == 6
    assert meta.prompt_type == "chat"


def test_messages_input_meta(client):
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello there"},
    ]
    with wildedge.llm_api(model="m", provider="p", messages=messages) as call:
        call.usage(tokens_in=3)

    meta = client.handle.inferences[0]["input_meta"]
    assert meta.char_count == len("hello there")
    assert meta.word_count == 2


def test_first_token_sets_ttft(client):
    with wildedge.llm_api(model="m", provider="p") as call:
        call.first_token()
        call.usage(tokens_out=1)

    assert client.handle.inferences[0]["output_meta"].time_to_first_token_ms is not None


def test_async_context_manager(client):
    async def run():
        async with wildedge.llm_api(model="m", provider="p") as call:
            call.usage(tokens_in=1)

    asyncio.run(run())
    assert len(client.handle.inferences) == 1


def test_source_derived_from_base_url(client):
    with wildedge.llm_api(model="m", base_url="https://openrouter.ai/api/v1"):
        pass

    assert client.registered[0]["source"] == "openrouter"


def test_success_flag_recorded(client):
    with wildedge.llm_api(model="m", provider="p") as call:
        call.usage(tokens_out=1)
        call.success = False

    assert client.handle.inferences[0]["success"] is False


def test_no_output_meta_when_nothing_known(client):
    with wildedge.llm_api(model="m", provider="p"):
        pass

    event = client.handle.inferences[0]
    assert event["output_meta"] is None
    assert event["api_meta"] is None


def test_noop_without_dsn_end_to_end(monkeypatch):
    monkeypatch.delenv(constants.ENV_DSN, raising=False)
    monkeypatch.delenv(constants.ENV_INTEGRATIONS, raising=False)

    with wildedge.llm_api(model="m", provider="p") as call:
        call.usage(tokens_in=1, tokens_out=2)

    assert wildedge.get_client().noop is True


def test_registers_model_as_api_format(client):
    with wildedge.llm_api(model="m", provider="p"):
        pass

    assert client.registered[0]["model_format"] == "api"
