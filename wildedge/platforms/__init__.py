from __future__ import annotations

import sys
from pathlib import Path

from wildedge.platforms.base import Platform
from wildedge.platforms.device_info import DeviceInfo
from wildedge.platforms.hardware import HardwareContext
from wildedge.platforms.linux import LinuxPlatform
from wildedge.platforms.macos import MacOSPlatform
from wildedge.platforms.unknown import UnknownPlatform
from wildedge.platforms.windows import WindowsPlatform

PLATFORMS: dict[str, Platform] = {
    "linux": LinuxPlatform(),
    "darwin": MacOSPlatform(),
    "win32": WindowsPlatform(),
}


def get_current_platform() -> Platform:
    return PLATFORMS.get(sys.platform, UnknownPlatform())


CURRENT_PLATFORM = get_current_platform()


def get_device_id_path() -> Path:
    return CURRENT_PLATFORM.get_device_id_path()


def detect_device(
    api_key: str, app_version: str | None, overrides: dict | None = None
) -> DeviceInfo:
    return CURRENT_PLATFORM.detect_device(api_key, app_version, overrides)


def capture_hardware(accelerator_actual: str | None = None) -> HardwareContext:
    ctx = CURRENT_PLATFORM.hardware_context()
    if accelerator_actual is not None:
        ctx.accelerator_actual = accelerator_actual
    return ctx
