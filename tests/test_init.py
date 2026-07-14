from __future__ import annotations

import wildedge
import wildedge.convenience as convenience


def test_init_calls_instrument_for_integrations(monkeypatch):

    calls: list[tuple[str | None, list[str] | None]] = []

    class DummyClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def instrument(self, integration, *, hubs=None):
            calls.append((integration, hubs))

    monkeypatch.setattr(convenience, "WildEdge", DummyClient)

    client = wildedge.init(
        dsn="https://secret@ingest.wildedge.dev/key",
        integrations=["onnx", "timm"],
        hubs=["huggingface"],
    )

    assert isinstance(client, DummyClient)
    assert calls == [("onnx", ["huggingface"]), ("timm", ["huggingface"])]


def test_init_hubs_only(monkeypatch):

    calls: list[tuple[str | None, list[str] | None]] = []

    class DummyClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def instrument(self, integration, *, hubs=None):
            calls.append((integration, hubs))

    monkeypatch.setattr(convenience, "WildEdge", DummyClient)

    client = wildedge.init(
        dsn="https://secret@ingest.wildedge.dev/key",
        hubs=["huggingface"],
    )

    assert isinstance(client, DummyClient)
    assert calls == [(None, ["huggingface"])]


def test_init_logs_debug_when_no_integrations_or_hubs(monkeypatch):
    logs: list[str] = []

    class DummyClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.debug = True

        def instrument(self, integration, *, hubs=None):
            raise AssertionError("instrument should not be called")

    monkeypatch.setattr(convenience, "WildEdge", DummyClient)
    monkeypatch.setattr(convenience.logger, "debug", lambda msg: logs.append(msg))

    client = wildedge.init(dsn="https://secret@ingest.wildedge.dev/key")

    assert isinstance(client, DummyClient)
    assert logs == ["wildedge: init called without integrations or hubs"]


def test_init_reuses_existing_default_client(monkeypatch):
    from wildedge.defaults import peek_default_client, set_default_client

    calls: list[tuple[str | None, list[str] | None]] = []

    class ExistingClient:
        def instrument(self, integration, *, hubs=None):
            calls.append((integration, hubs))

    existing = ExistingClient()
    set_default_client(existing)

    client = wildedge.init(integrations=["onnx"])

    assert client is existing
    assert calls == [("onnx", None)]
    assert peek_default_client() is existing


def test_init_with_dsn_replaces_default_client(monkeypatch):
    from wildedge.defaults import peek_default_client, set_default_client

    class ExistingClient:
        def instrument(self, integration, *, hubs=None):
            raise AssertionError("existing client must not be touched")

    class DummyClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def instrument(self, integration, *, hubs=None):
            pass

    existing = ExistingClient()
    set_default_client(existing)
    monkeypatch.setattr(convenience, "WildEdge", DummyClient)

    client = wildedge.init(dsn="https://secret@ingest.wildedge.dev/key")

    assert isinstance(client, DummyClient)
    assert client is not existing
    assert peek_default_client() is client


def test_init_reuse_warns_about_ignored_kwargs(monkeypatch):
    from wildedge.defaults import set_default_client

    class ExistingClient:
        def instrument(self, integration, *, hubs=None):
            pass

    logs: list[str] = []
    set_default_client(ExistingClient())
    monkeypatch.setattr(
        convenience.logger, "warning", lambda msg, *args: logs.append(msg % args)
    )

    wildedge.init(integrations=["onnx"], app_version="1.0.0")

    assert logs == [
        "wildedge: init() reusing the existing default client; ignoring app_version"
    ]


def test_init_registers_new_client_as_default(monkeypatch):
    from wildedge.defaults import peek_default_client

    class DummyClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def instrument(self, integration, *, hubs=None):
            pass

    monkeypatch.setattr(convenience, "WildEdge", DummyClient)

    client = wildedge.init(dsn="https://secret@ingest.wildedge.dev/key")

    assert peek_default_client() is client
