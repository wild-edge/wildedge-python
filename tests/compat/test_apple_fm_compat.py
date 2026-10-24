"""Real apple-fm-sdk sessions.

CI runners can't turn on Apple Intelligence, so a call there fails with the
SDK's own error. Either way the availability check must produce one load
event, and the call exactly one event, for the model it ran on.
"""

import asyncio
import inspect

import apple_fm_sdk as fm


def test_patched_signatures_match_the_sdk():
    """The patches read these arguments by position."""
    model_params = list(inspect.signature(fm.SystemLanguageModel).parameters)
    session_params = list(inspect.signature(fm.LanguageModelSession).parameters)
    stream_params = list(
        inspect.signature(fm.LanguageModelSession.stream_response).parameters
    )
    respond = inspect.signature(fm.LanguageModelSession.respond).parameters
    assert model_params[0] == "use_case"
    assert session_params[1] == "model"
    assert stream_params[1:3] == ["prompt", "options"]
    assert respond["options"].kind is inspect.Parameter.KEYWORD_ONLY


def test_respond_records_one_event(compat_client):
    compat_client.instrument("apple_fm")
    model = fm.SystemLanguageModel(
        use_case=fm.SystemLanguageModelUseCase.CONTENT_TAGGING
    )
    available, reason = model.is_available()
    session = fm.LanguageModelSession(model=model)
    options = fm.GenerationOptions(maximum_response_tokens=64)

    async def respond():
        try:
            return await session.respond("Tag this: rain", options=options)
        except fm.FoundationModelsError as exc:
            return exc

    result = asyncio.run(respond())

    [load, event] = compat_client.events
    assert load["event_type"] == "model_load"
    assert load["load"]["success"] is available
    if not available:
        assert load["load"]["error_code"] == reason.name
    assert event["model_id"] == "apple-foundation-model-content-tagging"
    if available:
        assert isinstance(result, str), result
        assert event["event_type"] == "inference"
        assert event["inference"]["generation_config"]["max_tokens"] == 64
    else:
        assert isinstance(result, fm.FoundationModelsError), (reason, result)
        assert event["event_type"] == "error"
        assert event["error"]["provider_error_code"] is not None
