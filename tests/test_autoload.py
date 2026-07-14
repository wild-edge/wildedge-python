"""Tests for wildedge/autoload/sitecustomize.py."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
import textwrap
from unittest.mock import patch

import wildedge.autoload.sitecustomize as _sc_mod

_MARKER = _sc_mod._INSTALLED_MARKER


def _reload_sitecustomize():
    """Reload sitecustomize module so module-level code re-executes."""
    sys.modules.pop("sitecustomize", None)
    importlib.reload(_sc_mod)
    return _sc_mod


def test_gunicorn_prefork_skips_double_init(monkeypatch):
    """Gunicorn pre-fork: sys.modules marker blocks re-init in the same interpreter.

    In gunicorn's fork-only model, workers inherit the parent's sys.modules so
    sitecustomize never re-executes. This covers the explicit reload edge case.
    """
    monkeypatch.setenv("WILDEDGE_AUTOLOAD", "1")
    monkeypatch.setitem(sys.modules, _MARKER, True)  # type: ignore[arg-type]

    calls = []
    with patch("wildedge.runtime.bootstrap.install_runtime", side_effect=calls.append):
        _reload_sitecustomize()

    assert calls == []


def test_skips_when_no_dsn(monkeypatch):
    """No activation env vars present: silent skip regardless of server."""
    monkeypatch.delitem(sys.modules, _MARKER, raising=False)
    monkeypatch.delenv("WILDEDGE_AUTOLOAD", raising=False)
    monkeypatch.delenv("WILDEDGE_DSN", raising=False)

    calls = []
    with patch("wildedge.runtime.bootstrap.install_runtime", side_effect=calls.append):
        _reload_sitecustomize()

    assert calls == []


def test_waitress_autoload_triggers_install(monkeypatch):
    """Waitress (single-process, thread-pool): WILDEDGE_AUTOLOAD=1 bootstraps the runtime.

    Waitress has no forking or reloader subprocess, so sitecustomize runs once
    in the server process and install_runtime is called.
    """
    monkeypatch.delitem(sys.modules, _MARKER, raising=False)
    monkeypatch.setenv("WILDEDGE_AUTOLOAD", "1")

    calls = []

    def fake_install(**kwargs):
        calls.append(kwargs)

    with patch("wildedge.runtime.bootstrap.install_runtime", side_effect=fake_install):
        _reload_sitecustomize()

    assert calls == [{"install_signal_handlers": False}]


def test_granian_dsn_triggers_install(monkeypatch):
    """Granian (direct DSN config): WILDEDGE_DSN alone is sufficient to bootstrap.

    Users running granian without `wildedge run` can set WILDEDGE_DSN and
    prepend wildedge/autoload/ to PYTHONPATH to get instrumentation.
    """
    monkeypatch.delitem(sys.modules, _MARKER, raising=False)
    monkeypatch.delenv("WILDEDGE_AUTOLOAD", raising=False)
    monkeypatch.setenv("WILDEDGE_DSN", "https://secret@ingest.wildedge.dev/key")

    calls = []

    def fake_install(**kwargs):
        calls.append(kwargs)

    with patch("wildedge.runtime.bootstrap.install_runtime", side_effect=fake_install):
        _reload_sitecustomize()

    assert calls == [{"install_signal_handlers": False}]


def test_bootstrap_exception_is_caught(monkeypatch, capsys):
    """Exception from install_runtime must not propagate; message written to stderr."""
    monkeypatch.delitem(sys.modules, _MARKER, raising=False)
    monkeypatch.setenv("WILDEDGE_AUTOLOAD", "1")

    with patch(
        "wildedge.runtime.bootstrap.install_runtime",
        side_effect=RuntimeError("boom"),
    ):
        _reload_sitecustomize()  # must not raise

    assert "boom" in capsys.readouterr().err


def test_chains_existing_sitecustomize(monkeypatch, tmp_path):
    """_load_existing_sitecustomize executes a pre-existing sitecustomize.py."""
    marker = tmp_path / "marker.txt"
    sc_file = tmp_path / "sitecustomize.py"
    sc_file.write_text(f"open({str(marker)!r}, 'w').close()\n")

    monkeypatch.delitem(sys.modules, _MARKER, raising=False)
    monkeypatch.delenv("WILDEDGE_AUTOLOAD", raising=False)
    monkeypatch.delenv("WILDEDGE_DSN", raising=False)
    sys.modules.pop("sitecustomize", None)

    real_find_spec = importlib.util.find_spec

    def patched_find_spec(name):
        if name == "sitecustomize":
            return importlib.util.spec_from_file_location("sitecustomize", str(sc_file))
        return real_find_spec(name)

    with patch.object(importlib.util, "find_spec", side_effect=patched_find_spec):
        _sc_mod._load_existing_sitecustomize()

    assert marker.exists(), "chained sitecustomize.py was not executed"


def test_uvicorn_reload_worker_bootstraps(tmp_path):
    """Uvicorn --reload: the server worker process bootstraps after the reloader already did.

    Uvicorn's reloader runs sitecustomize in the reloader process, then spawns
    the actual server worker via exec (fresh interpreter). The worker must be
    instrumented. The old os.environ guard (WILDEDGE_AUTOLOAD_ACTIVE) propagated
    across exec and blocked the worker's bootstrap. sys.modules is not inherited
    across exec, so the worker re-initialises correctly.
    """
    import os
    import pathlib

    autoload_dir = str(
        pathlib.Path(
            importlib.util.find_spec("wildedge.autoload.sitecustomize").origin
        ).parent
    )

    # Probe script: verify sitecustomize entered the bootstrap path in the child.
    # install_runtime will raise (no DSN) but the marker is set before the call,
    # so its presence in sys.modules means the worker reached the bootstrap code.
    script = textwrap.dedent("""\
        import sys
        import wildedge.autoload.sitecustomize as sc
        print("installed:", sc._INSTALLED_MARKER in sys.modules)
    """)
    script_file = tmp_path / "probe.py"
    script_file.write_text(script)

    env = os.environ.copy()
    env["WILDEDGE_AUTOLOAD"] = "1"
    # Simulate the env state the reloader leaves behind. This is what caused
    # the regression with the old os.environ guard.
    env["WILDEDGE_AUTOLOAD_ACTIVE"] = "1"
    env.pop("WILDEDGE_DSN", None)

    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = autoload_dir + (os.pathsep + pythonpath if pythonpath else "")

    result = subprocess.run(
        [sys.executable, str(script_file)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "installed: True" in result.stdout


def test_strict_missing_dsn_exits_120(monkeypatch):
    from wildedge.runtime.bootstrap import RuntimeConfigError

    monkeypatch.delitem(sys.modules, _MARKER, raising=False)
    monkeypatch.setenv("WILDEDGE_AUTOLOAD", "1")
    monkeypatch.setenv("WILDEDGE_STRICT", "1")
    monkeypatch.delenv("WILDEDGE_DSN", raising=False)

    with (
        patch(
            "wildedge.runtime.bootstrap.install_runtime",
            side_effect=RuntimeConfigError("WILDEDGE_DSN must be set"),
        ),
        patch("os._exit") as fake_exit,
    ):
        _reload_sitecustomize()

    fake_exit.assert_called_once_with(120)


def test_strict_integration_error_exits_121_without_strict_env(monkeypatch):
    """--strict-integrations is its own opt-in; it exits even without --strict."""
    from wildedge.runtime.bootstrap import RuntimeStrictIntegrationError

    monkeypatch.delitem(sys.modules, _MARKER, raising=False)
    monkeypatch.setenv("WILDEDGE_AUTOLOAD", "1")
    monkeypatch.delenv("WILDEDGE_STRICT", raising=False)

    with (
        patch(
            "wildedge.runtime.bootstrap.install_runtime",
            side_effect=RuntimeStrictIntegrationError("strict fail"),
        ),
        patch("os._exit") as fake_exit,
    ):
        _reload_sitecustomize()

    fake_exit.assert_called_once_with(121)


def test_strict_internal_error_exits_122(monkeypatch):
    monkeypatch.delitem(sys.modules, _MARKER, raising=False)
    monkeypatch.setenv("WILDEDGE_AUTOLOAD", "1")
    monkeypatch.setenv("WILDEDGE_STRICT", "1")

    with (
        patch(
            "wildedge.runtime.bootstrap.install_runtime",
            side_effect=RuntimeError("boom"),
        ),
        patch("os._exit") as fake_exit,
    ):
        _reload_sitecustomize()

    fake_exit.assert_called_once_with(122)


def test_non_strict_config_error_warns_and_continues(monkeypatch, capsys):
    from wildedge.runtime.bootstrap import RuntimeConfigError

    monkeypatch.delitem(sys.modules, _MARKER, raising=False)
    monkeypatch.setenv("WILDEDGE_AUTOLOAD", "1")
    monkeypatch.delenv("WILDEDGE_STRICT", raising=False)

    with patch(
        "wildedge.runtime.bootstrap.install_runtime",
        side_effect=RuntimeConfigError("no dsn"),
    ):
        _reload_sitecustomize()  # must not raise

    assert "running without telemetry" in capsys.readouterr().err


def test_startup_report_printed_from_env(monkeypatch, capsys):
    class FakeContext:
        debug = False
        print_startup_report = False
        integration_statuses: list = []

    monkeypatch.delitem(sys.modules, _MARKER, raising=False)
    monkeypatch.setenv("WILDEDGE_AUTOLOAD", "1")
    monkeypatch.setenv("WILDEDGE_PRINT_STARTUP_REPORT", "1")

    with (
        patch(
            "wildedge.runtime.bootstrap.install_runtime",
            return_value=FakeContext(),
        ),
        patch(
            "wildedge.runtime.bootstrap.format_startup_report",
            return_value="report-text",
        ),
    ):
        _reload_sitecustomize()

    assert "report-text" in capsys.readouterr().err


def test_no_propagate_clears_runtime_env(monkeypatch):
    import os

    class FakeContext:
        debug = False
        print_startup_report = False

    monkeypatch.delitem(sys.modules, _MARKER, raising=False)
    monkeypatch.setenv("WILDEDGE_AUTOLOAD", "1")
    monkeypatch.setenv("WILDEDGE_PROPAGATE", "0")
    monkeypatch.setenv("WILDEDGE_INTEGRATIONS", "all")

    with patch(
        "wildedge.runtime.bootstrap.install_runtime",
        return_value=FakeContext(),
    ):
        _reload_sitecustomize()

    assert "WILDEDGE_AUTOLOAD" not in os.environ
    assert "WILDEDGE_INTEGRATIONS" not in os.environ


def test_strict_missing_dsn_exit_code_in_real_interpreter(tmp_path):
    """sys.exit inside sitecustomize must terminate interpreter startup with our code."""
    import os
    import pathlib

    autoload_dir = str(
        pathlib.Path(
            importlib.util.find_spec("wildedge.autoload.sitecustomize").origin
        ).parent
    )

    env = os.environ.copy()
    env["WILDEDGE_AUTOLOAD"] = "1"
    env["WILDEDGE_STRICT"] = "1"
    env.pop("WILDEDGE_DSN", None)
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = autoload_dir + (os.pathsep + pythonpath if pythonpath else "")

    result = subprocess.run(
        [sys.executable, "-c", "print('should not run')"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 120, (result.returncode, result.stderr)
    assert "should not run" not in result.stdout
    assert "WILDEDGE_DSN must be set" in result.stderr
