from __future__ import annotations

import sys

from wildedge.platforms.base import Platform
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


def capture(accelerator_actual: str | None = None) -> HardwareContext:
    ctx = CURRENT_PLATFORM.hardware_context()
    if accelerator_actual is not None:
        ctx.accelerator_actual = accelerator_actual
    return ctx
