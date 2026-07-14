from __future__ import annotations

from unittest.mock import MagicMock

import wildedge
import wildedge.defaults as defaults
from wildedge import constants


class DummyClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instrument_calls: list[tuple[str | None, list[str] | None]] = []

    def instrument(self, integration, *, hubs=None):
        self.instrument_calls.append((integration, hubs))


def test_get_client_lazily_creates_noop_without_dsn(monkeypatch):
    monkeypatch.delenv(constants.ENV_DSN, raising=False)
    monkeypatch.delenv(constants.ENV_INTEGRATIONS, raising=False)
    client = wildedge.get_client()
    assert client.noop is True
    assert wildedge.get_client() is client


def test_set_default_client_wins():
    client = DummyClient()
    defaults.set_default_client(client)
    assert wildedge.get_client() is client
    assert defaults.peek_default_client() is client


def test_peek_does_not_create():
    assert defaults.peek_default_client() is None


def test_lazy_client_reads_integrations_env(monkeypatch):
    monkeypatch.setattr(defaults, "WildEdge", DummyClient)
    monkeypatch.setenv(constants.ENV_INTEGRATIONS, "openai")
    monkeypatch.delenv(constants.ENV_HUBS, raising=False)
    assert wildedge.get_client().instrument_calls == [("openai", None)]


def test_lazy_client_without_env_instruments_nothing(monkeypatch):
    monkeypatch.setattr(defaults, "WildEdge", DummyClient)
    monkeypatch.delenv(constants.ENV_INTEGRATIONS, raising=False)
    monkeypatch.delenv(constants.ENV_HUBS, raising=False)
    assert wildedge.get_client().instrument_calls == []


def test_lazy_client_hubs_only(monkeypatch):
    monkeypatch.setattr(defaults, "WildEdge", DummyClient)
    monkeypatch.delenv(constants.ENV_INTEGRATIONS, raising=False)
    monkeypatch.setenv(constants.ENV_HUBS, "huggingface")
    assert wildedge.get_client().instrument_calls == [(None, ["huggingface"])]


def test_lazy_client_env_errors_do_not_raise(monkeypatch):
    class Exploding(DummyClient):
        def instrument(self, integration, *, hubs=None):
            raise ValueError("unknown integration")

    monkeypatch.setattr(defaults, "WildEdge", Exploding)
    monkeypatch.setenv(constants.ENV_INTEGRATIONS, "bogus")
    client = wildedge.get_client()
    assert isinstance(client, Exploding)


def test_span_delegates_to_default_client():
    client = MagicMock()
    defaults.set_default_client(client)
    wildedge.span(kind="tool", name="lint", step_index=2)
    kwargs = client.span.call_args.kwargs
    assert kwargs["kind"] == "tool"
    assert kwargs["name"] == "lint"
    assert kwargs["step_index"] == 2


def test_track_span_delegates_to_default_client():
    client = MagicMock()
    defaults.set_default_client(client)
    wildedge.track_span(kind="tool", name="call", duration_ms=15, status="ok")
    kwargs = client.track_span.call_args.kwargs
    assert kwargs["duration_ms"] == 15


def test_register_model_delegates_to_default_client():
    client = MagicMock()
    defaults.set_default_client(client)
    model = object()
    wildedge.register_model(model, model_id="org/model", source="openrouter")
    args, kwargs = client.register_model.call_args
    assert args == (model,)
    assert kwargs["model_id"] == "org/model"
    assert kwargs["source"] == "openrouter"


def test_flush_delegates_to_default_client():
    client = MagicMock()
    defaults.set_default_client(client)
    wildedge.flush(timeout=2.0)
    client.flush.assert_called_once_with(timeout=2.0)


def test_trace_is_pure_context_and_creates_no_client(monkeypatch):
    monkeypatch.delenv(constants.ENV_DSN, raising=False)
    with wildedge.trace(run_id="r1", agent_id="a1"):
        ctx = wildedge.get_trace_context()
        assert ctx is not None
        assert ctx.run_id == "r1"
        assert ctx.agent_id == "a1"
    assert wildedge.get_trace_context() is None
    assert defaults.peek_default_client() is None


def test_module_api_works_end_to_end_without_dsn(monkeypatch):
    monkeypatch.delenv(constants.ENV_DSN, raising=False)
    monkeypatch.delenv(constants.ENV_INTEGRATIONS, raising=False)
    with wildedge.trace(run_id="r"):
        with wildedge.span(kind="eval", name="lint") as span:
            span.set_attributes(lint_errors=3)
            span.fail("3 lint errors")
    assert wildedge.get_client().noop is True
