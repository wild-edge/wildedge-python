"""Tests for wildedge/autoload/sitecustomize.py."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from unittest.mock import patch

import wildedge.autoload.sitecustomize as _sc_mod


def _reload_sitecustomize():
    """Reload sitecustomize module so module-level code re-executes."""
    sys.modules.pop("sitecustomize", None)
    importlib.reload(_sc_mod)
    return _sc_mod


def test_guard_prevents_double_init(monkeypatch):
    """WILDEDGE_AUTOLOAD_ACTIVE set: install_runtime not called."""
    monkeypatch.setenv("WILDEDGE_AUTOLOAD_ACTIVE", "1")
    monkeypatch.setenv("WILDEDGE_AUTOLOAD", "1")

    calls = []
    with patch("wildedge.runtime.bootstrap.install_runtime", side_effect=calls.append):
        _reload_sitecustomize()

    assert calls == []


def test_skips_when_no_dsn(monkeypatch):
    """No DSN env vars: silent skip."""
    monkeypatch.delenv("WILDEDGE_AUTOLOAD_ACTIVE", raising=False)
    monkeypatch.delenv("WILDEDGE_AUTOLOAD", raising=False)
    monkeypatch.delenv("WILDEDGE_DSN", raising=False)

    calls = []
    with patch("wildedge.runtime.bootstrap.install_runtime", side_effect=calls.append):
        _reload_sitecustomize()

    assert calls == []


def test_autoload_flag_triggers_install(monkeypatch):
    """WILDEDGE_AUTOLOAD=1 present: install_runtime called with install_signal_handlers=False."""
    monkeypatch.delenv("WILDEDGE_AUTOLOAD_ACTIVE", raising=False)
    monkeypatch.setenv("WILDEDGE_AUTOLOAD", "1")

    calls = []

    def fake_install(**kwargs):
        calls.append(kwargs)

    with patch("wildedge.runtime.bootstrap.install_runtime", side_effect=fake_install):
        _reload_sitecustomize()

    assert calls == [{"install_signal_handlers": False}]


def test_dsn_triggers_install(monkeypatch):
    """WILDEDGE_DSN present: install_runtime called."""
    monkeypatch.delenv("WILDEDGE_AUTOLOAD_ACTIVE", raising=False)
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
    monkeypatch.delenv("WILDEDGE_AUTOLOAD_ACTIVE", raising=False)
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

    monkeypatch.delenv("WILDEDGE_AUTOLOAD_ACTIVE", raising=False)
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
