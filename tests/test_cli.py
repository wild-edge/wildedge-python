from __future__ import annotations

import json
import os

import pytest

from wildedge import cli, constants
from wildedge.integrations.registry import IntegrationSpec
from wildedge.runtime import bootstrap
from wildedge.runtime import runner as runtime_runner


def _fake_execle(captured: dict):
    def _execle(path, *args):  # type: ignore[no-untyped-def]
        # Last positional arg is the env dict (os.execle convention).
        captured["path"] = path
        captured["argv"] = list(args[:-1])
        captured["env"] = args[-1]

    return _execle


def test_cli_run_execs_command_with_env(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(cli.os, "execle", _fake_execle(captured))
    monkeypatch.setattr(cli.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")

    rc = cli.main(
        [
            "run",
            "--dsn",
            "https://secret@ingest.wildedge.dev/key",
            "--app-version",
            "1.2.3",
            "--debug",
            "--",
            "gunicorn",
            "myapp.wsgi:app",
            "--workers",
            "4",
        ]
    )

    assert rc == 0
    assert captured["path"] == "/usr/bin/gunicorn"
    assert captured["argv"] == ["/usr/bin/gunicorn", "myapp.wsgi:app", "--workers", "4"]
    assert (
        captured["env"][constants.ENV_DSN] == "https://secret@ingest.wildedge.dev/key"
    )
    assert captured["env"][constants.WILDEDGE_AUTOLOAD] == "1"
    assert captured["env"][constants.ENV_APP_VERSION] == "1.2.3"
    assert captured["env"][constants.ENV_DEBUG] == "1"
    assert captured["env"][constants.ENV_PROPAGATE] == "1"
    assert captured["env"][constants.ENV_STRICT_INTEGRATIONS] == "0"
    assert captured["env"][constants.ENV_PRINT_STARTUP_REPORT] == "0"
    assert captured["env"][constants.ENV_FLUSH_TIMEOUT] == str(
        constants.DEFAULT_SHUTDOWN_FLUSH_TIMEOUT_SEC
    )


def test_cli_run_prepends_autoload_to_pythonpath(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(cli.os, "execle", _fake_execle(captured))
    monkeypatch.setattr(cli.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.delenv("PYTHONPATH", raising=False)

    cli.main(["run", "--", "gunicorn", "myapp.wsgi:app"])

    autoload_dir = str(cli.Path(__file__).parent.parent / "wildedge" / "autoload")
    assert captured["env"]["PYTHONPATH"] == autoload_dir


def test_cli_run_preserves_existing_pythonpath(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(cli.os, "execle", _fake_execle(captured))
    monkeypatch.setattr(cli.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.setenv("PYTHONPATH", "/existing/path")

    cli.main(["run", "--", "gunicorn", "myapp.wsgi:app"])

    pythonpath = captured["env"]["PYTHONPATH"]
    assert pythonpath.endswith(os.pathsep + "/existing/path")


def test_cli_run_sets_no_propagate_and_strict(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(cli.os, "execle", _fake_execle(captured))
    monkeypatch.setattr(cli.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")

    rc = cli.main(
        [
            "run",
            "--strict-integrations",
            "--no-propagate",
            "--",
            "gunicorn",
            "myapp.wsgi:app",
        ]
    )

    assert rc == 0
    assert captured["env"][constants.ENV_PROPAGATE] == "0"
    assert captured["env"][constants.ENV_STRICT_INTEGRATIONS] == "1"


def test_cli_run_sets_print_startup_report(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(cli.os, "execle", _fake_execle(captured))
    monkeypatch.setattr(cli.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")

    rc = cli.main(["run", "--print-startup-report", "--", "gunicorn", "myapp.wsgi:app"])

    assert rc == 0
    assert captured["env"][constants.ENV_PRINT_STARTUP_REPORT] == "1"


def test_cli_run_returns_127_for_missing_command(capsys, monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda cmd: None)
    monkeypatch.setattr(cli.Path, "is_file", lambda self: False)
    rc = cli.main(["run", "--", "nonexistent-command"])
    assert rc == 127
    assert "command not found" in capsys.readouterr().err


def test_cli_run_wraps_python_script_with_interpreter(monkeypatch, tmp_path):
    script = tmp_path / "app.py"
    script.write_text("pass")
    captured: dict = {}
    monkeypatch.setattr(cli.os, "execle", _fake_execle(captured))
    monkeypatch.setattr(
        cli.shutil,
        "which",
        lambda cmd: None if cmd.endswith(".py") else f"/usr/bin/{cmd}",
    )

    rc = cli.main(["run", "--", str(script)])

    assert rc == 0
    assert captured["argv"][1] == str(script)
    assert "python" in captured["path"].lower()


def test_install_runtime_requires_dsn(monkeypatch):
    monkeypatch.delenv(constants.ENV_DSN, raising=False)
    monkeypatch.delenv("WILDEDGE_DSN", raising=False)
    with pytest.raises(RuntimeError):
        bootstrap.install_runtime()


def test_install_runtime_default_flush_timeout_is_shutdown_budget(monkeypatch):
    class FakeWildEdge:
        SUPPORTED_INTEGRATIONS = {"onnx"}

        def __init__(self, *, dsn, app_version, debug):  # type: ignore[no-untyped-def]
            pass

        def instrument(self, name):  # type: ignore[no-untyped-def]
            pass

        def flush(self, timeout):  # type: ignore[no-untyped-def]
            pass

        def close(self):  # type: ignore[no-untyped-def]
            pass

    monkeypatch.setattr(bootstrap, "WildEdge", FakeWildEdge)
    monkeypatch.setenv(constants.ENV_DSN, "https://secret@ingest.wildedge.dev/key")
    monkeypatch.delenv(constants.ENV_FLUSH_TIMEOUT, raising=False)
    monkeypatch.setattr(bootstrap.importlib.util, "find_spec", lambda _: object())

    context = bootstrap.install_runtime()
    try:
        assert context.flush_timeout == constants.DEFAULT_SHUTDOWN_FLUSH_TIMEOUT_SEC
    finally:
        context.shutdown()


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
    monkeypatch.setenv(constants.ENV_DSN, "https://secret@ingest.wildedge.dev/key")
    monkeypatch.setenv(constants.ENV_APP_VERSION, "2.0.0")
    monkeypatch.setenv(constants.ENV_DEBUG, "1")
    monkeypatch.setenv(constants.ENV_INTEGRATIONS, "torch")
    monkeypatch.setenv(constants.ENV_FLUSH_TIMEOUT, "7.5")
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
    monkeypatch.setenv(constants.ENV_DSN, "https://secret@ingest.wildedge.dev/key")
    monkeypatch.setenv(constants.ENV_INTEGRATIONS, "onnx")
    monkeypatch.setenv(constants.ENV_STRICT_INTEGRATIONS, "1")

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
    rc = cli.main(["doctor", "--hubs", "huggingface"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "hub[huggingface]: OK" in out
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
        "hubs",
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


def test_doctor_reports_offline_and_dead_letter_checks(monkeypatch, capsys):
    monkeypatch.setenv("WILDEDGE_DSN", "https://secret@ingest.wildedge.dev/key")
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(cli, "check_writable_dir", lambda _: (True, "ok"))
    rc = cli.main(["doctor", "--integrations", "onnx"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "offline_queue_capacity: OK" in out
    assert "dead_letter_capacity: OK" in out
    assert "writable_offline_queue_dir: OK (ok)" in out
    assert "writable_dead_letter_dir: SKIP" in out


def test_doctor_reports_dead_letter_dir_when_enabled(monkeypatch, capsys):
    monkeypatch.setenv("WILDEDGE_DSN", "https://secret@ingest.wildedge.dev/key")
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(cli, "check_writable_dir", lambda _: (True, "ok"))
    rc = cli.main(["doctor", "--integrations", "onnx", "--dead-letter-persistence"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "writable_dead_letter_dir: OK (ok)" in out


def test_doctor_uses_project_key_for_default_namespace(monkeypatch, capsys):
    monkeypatch.setenv("WILDEDGE_DSN", "https://secret@ingest.wildedge.dev/test-prod")
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(cli, "check_writable_dir", lambda path: (True, str(path)))
    monkeypatch.delenv("WILDEDGE_APP_IDENTITY", raising=False)
    rc = cli.main(["doctor", "--integrations", "onnx"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "/test-prod/pending_queue" in out
    assert "/test-prod/dead_letters" in out


def test_doctor_uses_app_identity_override_for_namespace(monkeypatch, capsys):
    monkeypatch.setenv("WILDEDGE_DSN", "https://secret@ingest.wildedge.dev/test-prod")
    monkeypatch.setenv("WILDEDGE_APP_IDENTITY", "my-app")
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(cli, "check_writable_dir", lambda path: (True, str(path)))
    rc = cli.main(["doctor", "--integrations", "onnx"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "/my-app/pending_queue" in out
    assert "/my-app/dead_letters" in out


def test_runner_clears_runtime_env_when_no_propagate(monkeypatch):
    class FakeContext:
        debug = False
        print_startup_report = False

        def shutdown(self):  # type: ignore[no-untyped-def]
            return None

    monkeypatch.setenv(constants.ENV_PROPAGATE, "0")
    monkeypatch.setenv(constants.ENV_DSN, "https://secret@ingest.wildedge.dev/key")
    monkeypatch.setenv(constants.ENV_INTEGRATIONS, "all")
    monkeypatch.setattr(runtime_runner, "install_runtime", lambda: FakeContext())
    monkeypatch.setattr(runtime_runner.runpy, "run_path", lambda *a, **k: None)

    rc = runtime_runner.main(["--mode", "script", "--target", "app.py"])
    assert rc == 0
    assert constants.WILDEDGE_AUTOLOAD not in os.environ
    assert constants.ENV_INTEGRATIONS not in os.environ


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
    monkeypatch.setenv(constants.ENV_DSN, "https://secret@ingest.wildedge.dev/key")
    monkeypatch.setenv(constants.ENV_INTEGRATIONS, "x")

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


def test_parse_run_args_without_double_dash():
    """parse_run_args accepts tokens without a leading '--'."""
    cmd, args = cli.parse_run_args(["gunicorn", "myapp.wsgi:app", "--workers", "4"])
    assert cmd == "gunicorn"
    assert args == ["myapp.wsgi:app", "--workers", "4"]


def test_parse_run_args_strips_double_dash():
    """Leading '--' separator is stripped before parsing."""
    cmd, args = cli.parse_run_args(["--", "gunicorn", "myapp.wsgi:app"])
    assert cmd == "gunicorn"
    assert args == ["myapp.wsgi:app"]


def test_parse_run_args_empty_raises():
    """Empty token list raises ValueError."""
    with pytest.raises(ValueError, match="missing command"):
        cli.parse_run_args([])


def test_parse_run_args_only_double_dash_raises():
    """['--'] with nothing after raises ValueError."""
    with pytest.raises(ValueError, match="missing command"):
        cli.parse_run_args(["--"])
