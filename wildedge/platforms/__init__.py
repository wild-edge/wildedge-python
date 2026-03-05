from __future__ import annotations

import sys

from wildedge.platforms.base import PlatformAdapter
from wildedge.platforms.linux import LinuxPlatform
from wildedge.platforms.macos import MacOSPlatform
from wildedge.platforms.unknown import UnknownPlatform
from wildedge.platforms.windows import WindowsPlatform

PLATFORMS: dict[str, PlatformAdapter] = {
    "linux": LinuxPlatform(),
    "darwin": MacOSPlatform(),
    "win32": WindowsPlatform(),
}


def get_current_platform() -> PlatformAdapter:
    return PLATFORMS.get(sys.platform, UnknownPlatform())


CURRENT_PLATFORM = get_current_platform()
