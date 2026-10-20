"""Shared fixtures for the compatibility suite.

These tests run against real, pinned library versions (see
scripts/compat_matrix.py). Import the library directly: a missing or broken
install must fail the job, never skip it.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

import wildedge
from wildedge.platforms.device_info import DeviceInfo

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Fail any skipped test: a skip here means a dependency didn't install.

    Expected failures (xfail) are still allowed; they name a known bug.
    """
    outcome = yield
    report = outcome.get_result()
    if report.skipped and not hasattr(report, "wasxfail"):
        report.outcome = "failed"
        report.longrepr = f"compat tests must not skip: {report.longrepr}"


def sdk_http_module(package: str) -> ModuleType:
    """The HTTP library an API SDK is built on: httpx, or httpx2 in newer releases.

    Tests pass a client with a MockTransport from that same library, so no
    request leaves the machine.
    """
    base_client = importlib.import_module(f"{package}._base_client")
    module = getattr(base_client, "httpx2", None) or base_client.httpx
    assert hasattr(module, "MockTransport"), module
    return module


class _DummyConsumer:
    def __init__(self, *args, **kwargs):
        pass

    def flush(self, timeout: float = 5.0) -> None:
        pass

    def close(self, timeout: float | None = None) -> None:
        pass

    def _pause(self) -> None:
        pass

    def _resume(self) -> None:
        pass


class _DummyTransmitter:
    def __init__(self, *args, **kwargs):
        pass


class CompatClient:
    """A WildEdge client that records published events instead of sending them."""

    def __init__(self, client: wildedge.WildEdge) -> None:
        self.client = client
        self.events: list[dict] = []
        # Handles keep a reference to client.publish, so swap it before any
        # model is registered.
        client.publish = self.events.append  # type: ignore[method-assign]

    def __getattr__(self, name: str):
        return getattr(self.client, name)

    def of_type(self, event_type: str) -> list[dict]:
        return [e for e in self.events if e["event_type"] == event_type]

    def one(self, event_type: str) -> dict:
        found = self.of_type(event_type)
        types = [e["event_type"] for e in self.events]
        assert len(found) == 1, f"expected one {event_type} event, got {types}"
        return found[0]


@pytest.fixture(scope="session")
def _session_client():
    # One client per process, as in a real app: integration patches are
    # process-wide and stay bound to the client that installed them (#74).
    with (
        patch(
            "wildedge.client.detect_device",
            return_value=DeviceInfo(device_id="compat-device", device_type="linux"),
        ),
        patch("wildedge.client.Consumer", _DummyConsumer),
        patch("wildedge.client.Transmitter", _DummyTransmitter),
    ):
        client = wildedge.WildEdge(
            dsn="https://compat-secret@ingest.wildedge.dev/compat-project",
            app_version="compat",
            enable_offline_persistence=False,
            sampling_interval_s=0,
        )
    wrapped = CompatClient(client)
    yield wrapped
    client.close()


@pytest.fixture
def compat_client(_session_client):
    """The shared client, with the events of earlier tests cleared."""
    _session_client.events.clear()
    return _session_client
