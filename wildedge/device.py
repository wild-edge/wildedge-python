"""Device info auto-detection (stdlib-only)."""

from __future__ import annotations

import hashlib
import hmac
import locale
import os
import platform
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from wildedge import constants
from wildedge.platforms import CURRENT_PLATFORM
from wildedge.platforms import PLATFORMS as _PLATFORMS
from wildedge.platforms.base import debug_detection_failure
from wildedge.platforms.linux import LinuxPlatform
from wildedge.platforms.macos import MacOSPlatform
from wildedge.platforms.unknown import UnknownPlatform
from wildedge.platforms.windows import WindowsPlatform

PLATFORMS = _PLATFORMS

__all__ = [
    "CURRENT_PLATFORM",
    "PLATFORMS",
    "LinuxPlatform",
    "MacOSPlatform",
    "WindowsPlatform",
    "UnknownPlatform",
    "DeviceInfo",
    "get_device_id_path",
    "load_or_create_device_uuid",
    "hmac_device_id",
    "detect_gpu_info",
    "detect_locale",
    "detect_timezone",
    "detect_device",
]


def get_device_id_path() -> Path:
    """
    Returns the path to the device ID file for the current platform.
    """
    return (
        CURRENT_PLATFORM.config_base()
        / constants.DEVICE_ID_DIR
        / constants.DEVICE_ID_FILE
    )


def load_or_create_device_uuid() -> str:
    """
    Loads a persistent UUID for this device from disk, or creates one if not found.
    """
    path = get_device_id_path()
    try:
        if path.exists():
            stored = path.read_text().strip()
            if stored:
                return stored
    except OSError as exc:
        debug_detection_failure("device_uuid read", exc)

    new_id = str(uuid.uuid4())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_id)
    except OSError as exc:
        debug_detection_failure("device_uuid write", exc)
    return new_id


def hmac_device_id(api_key: str, raw_id: str) -> str:
    return hmac.new(
        key=api_key.encode(),
        msg=raw_id.encode(),
        digestmod=hashlib.sha256,
    ).hexdigest()


def detect_gpu_info() -> tuple[list[str], str | None]:
    """Returns (accelerators, gpu_name). Always includes 'cpu'. Stdlib + ctypes only."""
    gpu_accs, gpu_name = CURRENT_PLATFORM.gpu_accelerators()
    return ["cpu", *gpu_accs], gpu_name


def detect_locale() -> str | None:
    try:
        loc = locale.getlocale()
        return loc[0] if loc else None
    except Exception as exc:
        debug_detection_failure("locale", exc)
        return None


def detect_timezone() -> str | None:
    try:
        return datetime.now().astimezone().tzname()
    except Exception as exc:
        debug_detection_failure("timezone", exc)
        return None


@dataclass
class DeviceInfo:
    device_id: str
    device_type: str
    sdk_version: str = constants.SDK_VERSION
    device_model: str | None = None
    os_version: str | None = None
    locale: str | None = None
    timezone: str | None = None
    cpu_arch: str | None = None
    cpu_cores: int | None = None
    ram_total_bytes: int | None = None
    disk_total_bytes: int | None = None
    accelerators: list[str] = field(default_factory=list)
    gpu_name: str | None = None
    app_version: str | None = None

    def to_dict(self) -> dict:
        d = {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "device_model": self.device_model,
            "os_version": self.os_version,
            "sdk_version": self.sdk_version,
            "locale": self.locale,
            "timezone": self.timezone,
            "cpu_arch": self.cpu_arch,
            "cpu_cores": self.cpu_cores,
            "ram_total_bytes": self.ram_total_bytes,
            "disk_total_bytes": self.disk_total_bytes,
            "accelerators": self.accelerators,
            "gpu_name": self.gpu_name,
        }
        if self.app_version is not None:
            d["app_version"] = self.app_version
        return d


def detect_device(
    api_key: str, app_version: str | None, overrides: dict | None = None
) -> DeviceInfo:
    """Auto-detect device info and HMAC the stored UUID with api_key."""
    raw_uuid = load_or_create_device_uuid()
    device_id = hmac_device_id(api_key, raw_uuid)
    accelerators, gpu_name = detect_gpu_info()

    info = DeviceInfo(
        app_version=app_version,
        device_id=device_id,
        device_type=CURRENT_PLATFORM.wire_type,
        device_model=CURRENT_PLATFORM.device_model(),
        os_version=platform.version(),
        locale=detect_locale(),
        timezone=detect_timezone(),
        cpu_arch=platform.machine() or None,
        cpu_cores=os.cpu_count(),
        ram_total_bytes=CURRENT_PLATFORM.ram_bytes(),
        disk_total_bytes=CURRENT_PLATFORM.disk_bytes(),
        accelerators=accelerators,
        gpu_name=gpu_name,
    )

    if overrides:
        for k, v in overrides.items():
            if hasattr(info, k):
                setattr(info, k, v)

    return info
