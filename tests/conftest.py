"""Shared fixtures for the WildEdge SDK test suite."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from wildedge.client import WildEdge
from wildedge.device import DeviceInfo
from wildedge.model import ModelInfo


@pytest.fixture
def device_info():
    return DeviceInfo(
        app_version="1.0.0",
        device_id="test-device-id",
        device_type="linux",
        os_version="test-os",
        locale="en_US",
        timezone="UTC",
        cpu_arch="x86_64",
        cpu_cores=4,
        ram_total_bytes=8 * 1024**3,
        disk_total_bytes=256 * 1024**3,
        accelerators=["cpu"],
        gpu_name=None,
    )


@pytest.fixture
def model_info():
    return ModelInfo(
        model_name="test-model",
        model_version="1.0.0",
        model_source="local",
        model_format="onnx",
        model_family="test",
        quantization="int8",
    )


@pytest.fixture
def publish_spy():
    events = []

    def publish(event):
        events.append(event)

    publish.events = events
    return publish


@pytest.fixture
def dummy_handle():
    handle = SimpleNamespace(
        model_id="model-1",
        track_download=MagicMock(),
        track_load=MagicMock(),
        track_unload=MagicMock(),
        track_inference=MagicMock(),
        track_error=MagicMock(),
    )
    return handle


@pytest.fixture
def client_with_stubbed_runtime():
    with (
        patch("wildedge.client.detect_device", return_value=DeviceInfo("id", "linux")),
        patch("wildedge.client.Transmitter"),
        patch("wildedge.client.Consumer"),
    ):
        client = WildEdge(dsn="https://secret@ingest.wildedge.dev/key")
    return client
