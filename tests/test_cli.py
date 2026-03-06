from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from wildedge import cli
from wildedge.integrations.registry import IntegrationSpec
from wildedge.runtime import bootstrap
from wildedge.runtime import runner as runtime_runner


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
    assert (
        captured["env"][bootstrap.RUN_DSN_ENV]
        == "https://secret@ingest.wildedge.dev/key"
    )
    assert captured["env"][bootstrap.RUN_APP_VERSION_ENV] == "1.2.3"
    assert captured["env"][bootstrap.RUN_DEBUG_ENV] == "1"
    assert captured["env"][bootstrap.RUN_PROPAGATE_ENV] == "1"
    assert captured["env"][bootstrap.RUN_STRICT_INTEGRATIONS_ENV] == "0"
    assert captured["env"][bootstrap.RUN_PRINT_STARTUP_REPORT_ENV] == "0"


def test_cli_run_sets_no_propagate_and_strict(monkeypatch):
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
            "--strict-integrations",
            "--no-propagate",
            "--",
            "python",
            "-m",
            "pkg.main",
            "--foo",
        ]
    )

    assert rc == 0
    assert captured["cmd"][1:8] == [
        "-m",
        "wildedge.runtime.runner",
        "--mode",
        "module",
        "--target",
        "pkg.main",
        "--",
    ]
    assert captured["env"][bootstrap.RUN_PROPAGATE_ENV] == "0"
    assert captured["env"][bootstrap.RUN_STRICT_INTEGRATIONS_ENV] == "1"


def test_cli_run_sets_print_startup_report(monkeypatch):
    captured = {}

    def fake_run(cmd, env, check):  # type: ignore[no-untyped-def]
        captured["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    rc = cli.main(["run", "--print-startup-report", "--", "python", "app.py"])
    assert rc == 0
    assert captured["env"][bootstrap.RUN_PRINT_STARTUP_REPORT_ENV] == "1"


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
    monkeypatch.setattr(bootstrap.importlib.util, "find_spec", lambda _: object())

    context = bootstrap.install_runtime()
    try:
        assert events == [("instrument", "torch")]
    finally:
        context.shutdown()

    assert ("flush", "7.5") in events
    assert ("close", "") in events


def test_install_runtime_strict_integrations_raises(monkeypatch):
    class FakeWildEdge:
        SUPPORTED_INTEGRATIONS = {"onnx"}

        def __init__(self, *, dsn, app_version, debug):  # type: ignore[no-untyped-def]
            pass

        def instrument(self, name):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    monkeypatch.setattr(bootstrap, "WildEdge", FakeWildEdge)
    monkeypatch.setenv(bootstrap.RUN_DSN_ENV, "https://secret@ingest.wildedge.dev/key")
    monkeypatch.setenv(bootstrap.RUN_INTEGRATIONS_ENV, "onnx")
    monkeypatch.setenv(bootstrap.RUN_STRICT_INTEGRATIONS_ENV, "1")

    with pytest.raises(bootstrap.RuntimeStrictIntegrationError):
        bootstrap.install_runtime()


def test_doctor_reports_missing_dsn_and_unknown_integration(monkeypatch, capsys):
    monkeypatch.delenv("WILDEDGE_DSN", raising=False)
    rc = cli.main(["doctor", "--integrations", "madeup"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "dsn: FAIL" in out
    assert "integration[madeup]: FAIL" in out
    assert "doctor: FAIL" in out


def test_doctor_passes_for_available_module(monkeypatch, capsys):
    monkeypatch.setenv("WILDEDGE_DSN", "https://secret@ingest.wildedge.dev/key")
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(cli, "check_writable_dir", lambda _: (True, "ok"))
    rc = cli.main(["doctor", "--integrations", "huggingface"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "integration[huggingface]: OK" in out
    assert "doctor: PASS" in out


def test_doctor_json_output_schema(monkeypatch, capsys):
    monkeypatch.setenv("WILDEDGE_DSN", "https://secret@ingest.wildedge.dev/key")
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(cli, "check_writable_dir", lambda _: (True, "ok"))
    rc = cli.main(["doctor", "--format", "json", "--integrations", "onnx"])
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert rc == 0
    assert sorted(payload.keys()) == [
        "checks",
        "integrations",
        "platform",
        "python",
        "status",
    ]
    assert payload["status"] == "PASS"
    assert payload["integrations"][0]["name"] == "onnx"


def test_doctor_network_check_failure(monkeypatch, capsys):
    monkeypatch.setenv("WILDEDGE_DSN", "https://secret@ingest.wildedge.dev/key")
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(cli, "check_writable_dir", lambda _: (True, "ok"))
    monkeypatch.setattr(
        cli, "network_reachability_check", lambda _: (False, "unreachable")
    )

    rc = cli.main(["doctor", "--network-check", "--integrations", "onnx"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "network: FAIL (unreachable)" in out
    assert "doctor: FAIL" in out


def test_doctor_runtime_config_fail(monkeypatch, capsys):
    monkeypatch.setenv("WILDEDGE_DSN", "https://secret@ingest.wildedge.dev/key")
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(cli, "check_writable_dir", lambda _: (True, "ok"))
    rc = cli.main(["doctor", "--batch-size", "0", "--integrations", "onnx"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "runtime_config: FAIL (batch_size out of range)" in out


def test_runner_clears_runtime_env_when_no_propagate(monkeypatch):
    class FakeContext:
        debug = False
        print_startup_report = False

        def shutdown(self):  # type: ignore[no-untyped-def]
            return None

    monkeypatch.setenv(bootstrap.RUN_PROPAGATE_ENV, "0")
    monkeypatch.setenv(bootstrap.RUN_DSN_ENV, "https://secret@ingest.wildedge.dev/key")
    monkeypatch.setenv(bootstrap.RUN_INTEGRATIONS_ENV, "all")
    monkeypatch.setattr(runtime_runner, "install_runtime", lambda: FakeContext())
    monkeypatch.setattr(runtime_runner.runpy, "run_path", lambda *a, **k: None)

    rc = runtime_runner.main(["--mode", "script", "--target", "app.py"])
    assert rc == 0
    assert bootstrap.RUN_DSN_ENV not in os.environ
    assert bootstrap.RUN_INTEGRATIONS_ENV not in os.environ


def test_install_runtime_tracks_missing_dependency_status(monkeypatch):
    class FakeWildEdge:
        def __init__(self, *, dsn, app_version, debug):  # type: ignore[no-untyped-def]
            pass

        def instrument(self, name):  # type: ignore[no-untyped-def]
            raise AssertionError("instrument should not be called when deps missing")

        def flush(self, timeout):  # type: ignore[no-untyped-def]
            pass

        def close(self):  # type: ignore[no-untyped-def]
            pass

    monkeypatch.setattr(bootstrap, "WildEdge", FakeWildEdge)
    monkeypatch.setattr(
        bootstrap,
        "INTEGRATIONS_BY_NAME",
        {"x": IntegrationSpec("x", ("missing_mod",), "client_patch")},
    )
    monkeypatch.setattr(bootstrap.importlib.util, "find_spec", lambda _: None)
    monkeypatch.setenv(bootstrap.RUN_DSN_ENV, "https://secret@ingest.wildedge.dev/key")
    monkeypatch.setenv(bootstrap.RUN_INTEGRATIONS_ENV, "x")

    context = bootstrap.install_runtime()
    try:
        assert context.integration_statuses == [
            {
                "name": "x",
                "status": bootstrap.STATUS_SKIP_MISSING_DEP,
                "detail": "missing modules: missing_mod",
            }
        ]
    finally:
        context.shutdown()


def test_runner_returns_reserved_exit_codes(monkeypatch, capsys):
    monkeypatch.setattr(
        runtime_runner,
        "install_runtime",
        lambda: (_ for _ in ()).throw(bootstrap.RuntimeConfigError("bad config")),
    )
    assert runtime_runner.main(["--mode", "script", "--target", "app.py"]) == 120

    monkeypatch.setattr(
        runtime_runner,
        "install_runtime",
        lambda: (_ for _ in ()).throw(
            bootstrap.RuntimeStrictIntegrationError("strict fail")
        ),
    )
    assert runtime_runner.main(["--mode", "script", "--target", "app.py"]) == 121

    monkeypatch.setattr(
        runtime_runner,
        "install_runtime",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert runtime_runner.main(["--mode", "script", "--target", "app.py"]) == 122
    assert "wildedge:" in capsys.readouterr().err


def test_runner_prints_startup_report_when_enabled(monkeypatch, capsys):
    class FakeContext:
        debug = False
        print_startup_report = True
        integration_statuses = []

        def shutdown(self):  # type: ignore[no-untyped-def]
            return None

    monkeypatch.setattr(runtime_runner, "install_runtime", lambda: FakeContext())
    monkeypatch.setattr(runtime_runner, "format_startup_report", lambda _: "report")
    monkeypatch.setattr(runtime_runner.runpy, "run_path", lambda *a, **k: None)

    rc = runtime_runner.main(["--mode", "script", "--target", "app.py"])
    assert rc == 0
    assert "report" in capsys.readouterr().err
