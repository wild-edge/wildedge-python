"""Tests for the Apple Foundation Models (apple-fm-sdk) integration."""

from __future__ import annotations

import asyncio
import enum
from types import ModuleType, SimpleNamespace

import pytest

import wildedge.integrations.apple_fm as apple_fm_mod
from wildedge import accelerators
from wildedge.integrations.apple_fm import (
    MODEL_ID,
    AppleFmExtractor,
    build_input_meta,
    model_id_for,
    provider_error_code_of,
)
from wildedge.integrations.registry import find_extractor
from wildedge.model import ModelHandle

# ---------------------------------------------------------------------------
# Fake apple_fm_sdk, matching the signatures of apple-fm-sdk 0.2
# ---------------------------------------------------------------------------


class SystemLanguageModelUnavailableReason(enum.IntEnum):
    APPLE_INTELLIGENCE_NOT_ENABLED = 0
    DEVICE_NOT_ELIGIBLE = 1
    MODEL_NOT_READY = 2
    UNKNOWN = 255


class SystemLanguageModelUseCase(enum.IntEnum):
    GENERAL = 0
    CONTENT_TAGGING = 1


class FoundationModelsError(Exception):
    pass


class GenerationError(FoundationModelsError):
    pass


class GuardrailViolationError(GenerationError):
    pass


class ExceededContextWindowSizeError(GenerationError):
    pass


class DecodingFailureError(GenerationError):
    pass


def make_fake_sdk(
    *,
    reply="Hello there",
    snapshots=("Hel", "Hello", "Hello there"),
    error=None,
    stream_error=None,
    availability=(True, None),
) -> ModuleType:
    """A fresh module per test, since the integration patches its classes."""

    class SystemLanguageModel:
        def __init__(
            self,
            use_case=SystemLanguageModelUseCase.GENERAL,
            guardrails=None,
            _ptr=None,
        ):
            self._ptr = object()

        def is_available(self):
            return availability() if callable(availability) else availability

    class LanguageModelSession:
        def __init__(self, instructions=None, model=None, tools=None, _ptr=None):
            self.instructions = instructions

        @classmethod
        def from_transcript(cls, transcript, model=None, tools=None):
            return cls(_ptr=object())

        async def respond(
            self,
            prompt,
            generating=None,
            *,
            schema=None,
            json_schema=None,
            options=None,
        ):
            await asyncio.sleep(0)
            if error is not None:
                raise error
            return reply

        async def stream_response(self, prompt, options=None):
            for snapshot in snapshots:
                await asyncio.sleep(0)
                yield snapshot
            if stream_error is not None:
                raise stream_error

    SystemLanguageModel.__module__ = "apple_fm_sdk.core"
    LanguageModelSession.__module__ = "apple_fm_sdk.session"

    sdk = ModuleType("apple_fm_sdk")
    sdk.SystemLanguageModel = SystemLanguageModel
    sdk.LanguageModelSession = LanguageModelSession
    sdk.SystemLanguageModelUseCase = SystemLanguageModelUseCase
    return sdk


class FakeClient:
    """Registers through the real extractor so handles carry real ModelInfo."""

    def __init__(self, closed=False):
        self.closed = closed
        self.events: list[dict] = []
        self.handles: dict[str, ModelHandle] = {}

    def register_model(self, obj, *, model_id=None, **_kwargs):
        extractor = AppleFmExtractor()
        model_id, info = extractor.extract_info(obj, {"id": model_id})
        if model_id not in self.handles:
            handle = ModelHandle(model_id, info, self.events.append)
            extractor.install_hooks(obj, handle)
            self.handles[model_id] = handle
        return self.handles[model_id]

    def of_type(self, event_type):
        return [e for e in self.events if e["event_type"] == event_type]


@pytest.fixture
def install(monkeypatch):
    """Activate the integration on a fake SDK and return (sdk, client)."""

    def _install(client=None, **sdk_kwargs):
        sdk = make_fake_sdk(**sdk_kwargs)
        client = client or FakeClient()
        monkeypatch.setattr(apple_fm_mod, "_fm", sdk)
        monkeypatch.setattr(apple_fm_mod, "_apple_fm_patched", False)
        monkeypatch.setattr(apple_fm_mod, "_reported_availability", {})
        AppleFmExtractor.install_auto_load_patch(lambda: client)
        return sdk, client

    return _install


def inference_of(event):
    return event["inference"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_model_id_names_non_general_use_cases():
    assert model_id_for(None) == MODEL_ID
    assert model_id_for(SystemLanguageModelUseCase.GENERAL) == MODEL_ID
    assert (
        model_id_for(SystemLanguageModelUseCase.CONTENT_TAGGING)
        == "apple-foundation-model-content-tagging"
    )


def test_provider_error_code_follows_the_class_hierarchy():
    class AppGuardrailError(GuardrailViolationError):
        pass

    assert provider_error_code_of(GuardrailViolationError()) == "guardrail_violation"
    assert provider_error_code_of(AppGuardrailError()) == "guardrail_violation"
    assert provider_error_code_of(FoundationModelsError()) is None


def test_input_meta_counts_text_parts_only():
    modality, meta = build_input_meta(["Describe this", object(), "photo"])
    assert modality == "text"
    assert meta.char_count == len("Describe this photo")
    assert meta.word_count == 3

    modality, meta = build_input_meta(object())
    assert (modality, meta) == ("image", None)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


def test_extractor_is_found_for_sdk_objects_only():
    sdk = make_fake_sdk()
    assert isinstance(find_extractor(sdk.LanguageModelSession()), AppleFmExtractor)
    assert isinstance(find_extractor(sdk.SystemLanguageModel()), AppleFmExtractor)

    class LanguageModelSession:  # same name, different package
        pass

    assert not AppleFmExtractor().can_handle(LanguageModelSession())


def test_extract_info_describes_the_system_model(monkeypatch):
    monkeypatch.setattr(apple_fm_mod.platform, "mac_ver", lambda: ("26.6.2", "", ""))
    model_id, info = AppleFmExtractor().extract_info(
        make_fake_sdk().SystemLanguageModel(), {}
    )
    assert model_id == MODEL_ID
    assert info.model_version == "26.6.2"
    assert info.model_source == "apple"
    assert info.model_format == "apple_fm"


def test_registering_a_session_through_the_client(client_with_stubbed_runtime):
    session = make_fake_sdk().LanguageModelSession()
    handle = client_with_stubbed_runtime.register_model(session)
    assert handle.model_id == MODEL_ID
    assert handle.info.model_format == "apple_fm"
    assert handle.detected_accelerator == accelerators.COREML
    assert getattr(session, apple_fm_mod.APPLE_FM_HANDLE_ATTR) is handle
    client_with_stubbed_runtime.close()


# ---------------------------------------------------------------------------
# Auto-load patch
# ---------------------------------------------------------------------------


async def test_respond_records_one_inference(install):
    sdk, client = install()
    session = sdk.LanguageModelSession(instructions="Be brief.")
    options = SimpleNamespace(
        temperature=0.3,
        maximum_response_tokens=64,
        sampling=SimpleNamespace(top=40, probability_threshold=None, seed=7),
    )

    assert (
        await session.respond("How does rain form?", options=options) == "Hello there"
    )

    [event] = client.of_type("inference")
    inference = inference_of(event)
    assert event["model_id"] == MODEL_ID
    assert inference["success"] is True
    assert inference["input_modality"] == "text"
    assert inference["output_modality"] == "generation"
    assert inference["input_meta"]["word_count"] == 4
    assert inference["generation_config"] == {
        "temperature": 0.3,
        "top_k": 40,
        "max_tokens": 64,
        "seed": 7,
    }
    assert "tokens_out" not in inference["output_meta"]


async def test_sessions_share_a_handle_per_use_case(install):
    sdk, client = install()
    tagging = sdk.SystemLanguageModel(
        use_case=SystemLanguageModelUseCase.CONTENT_TAGGING
    )
    await sdk.LanguageModelSession().respond("a")
    await sdk.LanguageModelSession(model=sdk.SystemLanguageModel()).respond("b")
    await sdk.LanguageModelSession(model=tagging).respond("c")
    await sdk.LanguageModelSession.from_transcript(object()).respond("d")

    model_ids = [e["model_id"] for e in client.of_type("inference")]
    assert model_ids == [MODEL_ID, MODEL_ID, f"{MODEL_ID}-content-tagging", MODEL_ID]
    assert set(client.handles) == {MODEL_ID, f"{MODEL_ID}-content-tagging"}


async def test_respond_error_records_the_sdk_error_name(install):
    sdk, client = install(error=GuardrailViolationError("Guardrail violation occurred"))
    session = sdk.LanguageModelSession()

    with pytest.raises(GuardrailViolationError):
        await session.respond("something unsafe")

    [error] = client.of_type("error")
    assert error["error"]["error_code"] == "CONTENT_FILTERED"
    assert error["error"]["provider_error_code"] == "guardrail_violation"
    assert "duration_ms" in error["error"]
    assert client.of_type("inference") == []


async def test_stream_records_ttft_and_passes_snapshots_through(install):
    sdk, client = install()
    session = sdk.LanguageModelSession()

    chunks = [c async for c in session.stream_response("Tell me", None)]

    assert chunks == ["Hel", "Hello", "Hello there"]
    [event] = client.of_type("inference")
    inference = inference_of(event)
    assert inference["success"] is True
    assert "time_to_first_token_ms" in inference["output_meta"]


async def test_stream_closed_early_is_not_a_success(install):
    sdk, client = install()
    stream = sdk.LanguageModelSession().stream_response("Tell me")
    async for _ in stream:
        break
    await stream.aclose()

    [event] = client.of_type("inference")
    assert inference_of(event)["success"] is False
    assert inference_of(event)["error_code"] == "STREAM_CLOSED"


async def test_stream_error_records_the_sdk_error_name(install):
    sdk, client = install(stream_error=ExceededContextWindowSizeError("too long"))
    with pytest.raises(ExceededContextWindowSizeError):
        async for _ in sdk.LanguageModelSession().stream_response("Tell me"):
            pass

    [error] = client.of_type("error")
    assert error["error"]["error_code"] == "CONTEXT_LENGTH_EXCEEDED"
    assert error["error"]["provider_error_code"] == "exceeded_context_window_size"
    assert client.of_type("inference") == []


async def test_error_without_a_specific_code_is_unknown(install):
    sdk, client = install(error=DecodingFailureError("bad output"))
    with pytest.raises(DecodingFailureError):
        await sdk.LanguageModelSession().respond("hi")

    [error] = client.of_type("error")
    assert error["error"]["error_code"] == "UNKNOWN"
    assert error["error"]["provider_error_code"] == "decoding_failure"


def test_every_mapped_provider_code_is_a_known_sdk_error():
    assert set(apple_fm_mod.ERROR_CODES_BY_PROVIDER_CODE) <= set(
        apple_fm_mod.PROVIDER_ERROR_CODES.values()
    )


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def test_available_model_records_one_successful_load(install):
    sdk, client = install()
    model = sdk.SystemLanguageModel()

    assert model.is_available() == (True, None)
    model.is_available()
    sdk.SystemLanguageModel().is_available()

    [event] = client.of_type("model_load")
    assert event["model_id"] == MODEL_ID
    assert event["load"]["success"] is True
    assert event["load"]["error_code"] is None


def test_unavailable_model_records_the_reason(install):
    reason = SystemLanguageModelUnavailableReason.MODEL_NOT_READY
    sdk, client = install(availability=(False, reason))

    assert sdk.SystemLanguageModel().is_available() == (False, reason)

    [event] = client.of_type("model_load")
    assert event["load"]["success"] is False
    assert event["load"]["error_code"] == "MODEL_NOT_READY"


def test_availability_is_recorded_again_when_it_changes(install):
    not_ready = (False, SystemLanguageModelUnavailableReason.MODEL_NOT_READY)
    answers = iter([not_ready, not_ready, (True, None), (True, None)])
    sdk, client = install(availability=lambda: next(answers))
    model = sdk.SystemLanguageModel()

    for _ in range(4):
        model.is_available()

    loads = [e["load"] for e in client.of_type("model_load")]
    assert [(load["success"], load["error_code"]) for load in loads] == [
        (False, "MODEL_NOT_READY"),
        (True, None),
    ]


def test_closed_client_leaves_availability_untracked(install):
    sdk, client = install(client=FakeClient(closed=True))
    assert sdk.SystemLanguageModel().is_available() == (True, None)
    assert client.events == []


async def test_closed_client_leaves_sessions_untracked(install):
    sdk, client = install(client=FakeClient(closed=True))
    session = sdk.LanguageModelSession()
    assert await session.respond("hi") == "Hello there"
    assert not hasattr(session, apple_fm_mod.APPLE_FM_HANDLE_ATTR)
    assert client.events == []


def test_install_is_idempotent(install):
    sdk, _client = install()
    respond = sdk.LanguageModelSession.respond
    AppleFmExtractor.install_auto_load_patch(lambda: None)
    apple_fm_mod._apple_fm_patched = False
    AppleFmExtractor.install_auto_load_patch(lambda: None)
    assert sdk.LanguageModelSession.respond is respond


def test_install_skips_when_sdk_missing(monkeypatch):
    monkeypatch.setattr(apple_fm_mod, "_fm", None)
    monkeypatch.setattr(apple_fm_mod, "_apple_fm_patched", False)
    monkeypatch.setattr(apple_fm_mod, "optional_module", lambda _name: None)
    AppleFmExtractor.install_auto_load_patch(lambda: None)
    assert apple_fm_mod._apple_fm_patched is False
