"""Tests for device detection."""

from pathlib import Path
from unittest.mock import patch

from wildedge.platforms import detect_device, get_device_id_path
from wildedge.platforms.base import Platform, hmac_device_id
from wildedge.platforms.device_info import DeviceInfo
from wildedge.platforms.linux import LinuxPlatform
from wildedge.platforms.macos import MacOSPlatform


class TestHmacDeviceId:
    def test_output_is_hex_string(self):
        result = hmac_device_id("my-api-key", "my-uuid")
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex digest

    def test_same_inputs_produce_same_output(self):
        r1 = hmac_device_id("key", "uuid")
        r2 = hmac_device_id("key", "uuid")
        assert r1 == r2

    def test_different_api_keys_produce_different_ids(self):
        r1 = hmac_device_id("key1", "same-uuid")
        r2 = hmac_device_id("key2", "same-uuid")
        assert r1 != r2


class TestDetectDevice:
    def test_returns_device_info(self, monkeypatch):
        monkeypatch.setattr(
            Platform, "load_or_create_device_uuid", lambda self: "test-uuid"
        )
        info = detect_device(api_key="test-key", app_version="1.0.0")
        assert isinstance(info, DeviceInfo)
        assert info.app_version == "1.0.0"
        assert info.sdk_version == "wildedge-python-0.1.0"

    def test_device_type_is_normalised(self, monkeypatch):
        monkeypatch.setattr(
            Platform, "load_or_create_device_uuid", lambda self: "test-uuid"
        )
        with patch("wildedge.platforms.CURRENT_PLATFORM", MacOSPlatform()):
            info = detect_device(api_key="k", app_version="1.0")
        assert info.device_type == "macos"

    def test_linux_device_type(self, monkeypatch):
        monkeypatch.setattr(
            Platform, "load_or_create_device_uuid", lambda self: "test-uuid"
        )
        with patch("wildedge.platforms.CURRENT_PLATFORM", LinuxPlatform()):
            info = detect_device(api_key="k", app_version="1.0")
        assert info.device_type == "linux"

    def test_device_id_is_hmac_of_uuid(self, monkeypatch):
        monkeypatch.setattr(
            Platform, "load_or_create_device_uuid", lambda self: "fixed-uuid"
        )
        info = detect_device(api_key="fixed-key", app_version="1.0")
        assert info.device_id == hmac_device_id("fixed-key", "fixed-uuid")

    def test_overrides_applied(self, monkeypatch):
        monkeypatch.setattr(
            Platform, "load_or_create_device_uuid", lambda self: "test-uuid"
        )
        info = detect_device(
            api_key="k",
            app_version="1.0",
            overrides={"device_model": "Jetson AGX"},
        )
        assert info.device_model == "Jetson AGX"

    def test_to_dict_returns_protocol_fields(self):
        info = DeviceInfo(
            app_version="2.0.0",
            device_id="hashed-id",
            device_type="linux",
            os_version="Ubuntu 22.04",
            locale="en_US",
            timezone="UTC",
        )
        d = info.to_dict()
        assert d["device_id"] == "hashed-id"
        assert d["device_type"] == "linux"
        assert d["app_version"] == "2.0.0"
        assert "sdk_version" in d

    def test_get_device_id_path_uses_config_constants(self):
        class FakePlatform(Platform):
            wire_type = "test"

            def config_base(self):
                return Path("/tmp/test-config")

            def state_base(self):
                return Path("/tmp")

            def cache_base(self):
                return Path("/tmp")

            def device_model(self):
                return None

            def os_version(self):
                return None

            def disk_bytes(self):
                return None

            def gpu_accelerators(self):
                return [], None

            def gpu_accelerator_for_offload(self):
                return "cpu"

        with patch("wildedge.platforms.CURRENT_PLATFORM", FakePlatform()):
            path = get_device_id_path()
        assert path == Path("/tmp/test-config/wildedge/device_id")
