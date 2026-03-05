from __future__ import annotations

from types import SimpleNamespace

import pytest

from wildedge import cli
from wildedge.runtime import bootstrap


def test_cli_run_script_invokes_runner(monkeypatch):
    captured = {}

    def fake_run(cmd, env, check):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["env"] = env
        captured["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    rc = cli.main(
        [
            "run",
            "--dsn",
            "https://secret@ingest.wildedge.dev/key",
            "--app-version",
            "1.2.3",
            "--debug",
            "--",
            "python",
            "app.py",
            "--foo",
            "bar",
        ]
    )

    assert rc == 0
    assert captured["cmd"][1:8] == [
        "-m",
        "wildedge.runtime.runner",
        "--mode",
        "script",
        "--target",
        "app.py",
        "--",
    ]
    assert captured["cmd"][8:] == ["--foo", "bar"]
    assert captured["env"][bootstrap.RUN_DSN_ENV] == "https://secret@ingest.wildedge.dev/key"
    assert captured["env"][bootstrap.RUN_APP_VERSION_ENV] == "1.2.3"
    assert captured["env"][bootstrap.RUN_DEBUG_ENV] == "1"


def test_cli_rejects_non_python_command(capsys):
    rc = cli.main(["run", "--", "bash", "script.sh"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "unsupported command format" in captured.err


def test_install_runtime_requires_dsn(monkeypatch):
    monkeypatch.delenv(bootstrap.RUN_DSN_ENV, raising=False)
    monkeypatch.delenv("WILDEDGE_DSN", raising=False)
    with pytest.raises(RuntimeError):
        bootstrap.install_runtime()


def test_install_runtime_instruments_requested_integrations(monkeypatch):
    events: list[tuple[str, str]] = []

    class FakeWildEdge:
        SUPPORTED_INTEGRATIONS = {"onnx", "torch"}

        def __init__(self, *, dsn, app_version, debug):  # type: ignore[no-untyped-def]
            assert dsn == "https://secret@ingest.wildedge.dev/key"
            assert app_version == "2.0.0"
            assert debug is True

        def instrument(self, name):  # type: ignore[no-untyped-def]
            events.append(("instrument", name))

        def flush(self, timeout):  # type: ignore[no-untyped-def]
            events.append(("flush", str(timeout)))

        def close(self):  # type: ignore[no-untyped-def]
            events.append(("close", ""))

    monkeypatch.setattr(bootstrap, "WildEdge", FakeWildEdge)
    monkeypatch.setenv(bootstrap.RUN_DSN_ENV, "https://secret@ingest.wildedge.dev/key")
    monkeypatch.setenv(bootstrap.RUN_APP_VERSION_ENV, "2.0.0")
    monkeypatch.setenv(bootstrap.RUN_DEBUG_ENV, "1")
    monkeypatch.setenv(bootstrap.RUN_INTEGRATIONS_ENV, "torch")
    monkeypatch.setenv(bootstrap.RUN_FLUSH_TIMEOUT_ENV, "7.5")

    context = bootstrap.install_runtime()
    try:
        assert events == [("instrument", "torch")]
    finally:
        context.shutdown()

    assert ("flush", "7.5") in events
    assert ("close", "") in events
