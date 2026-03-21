from __future__ import annotations

import wildedge


def test_init_calls_instrument_for_integrations(monkeypatch):
    import wildedge.convenience as convenience

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
    import wildedge.convenience as convenience

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
    import wildedge.convenience as convenience

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
