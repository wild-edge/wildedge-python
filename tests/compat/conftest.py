from __future__ import annotations

from unittest.mock import patch

import pytest

import wildedge
from wildedge.device import DeviceInfo


class _DummyConsumer:
    def __init__(self, *args, **kwargs):
        pass

    def flush(self, timeout: float = 5.0) -> None:
        pass

    def close(self) -> None:
        pass


class _DummyTransmitter:
    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture
def compat_client():
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
        )
    yield client
    client.close()
