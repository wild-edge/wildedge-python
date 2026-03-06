from __future__ import annotations

import shutil
import sys
from pathlib import Path

from wildedge.platforms.base import debug_detection_failure


class UnknownPlatform:
    wire_type = sys.platform

    def config_base(self) -> Path:
        return Path.home() / ".config"

    def state_base(self) -> Path:
        return Path.home() / ".local" / "state"

    def cache_base(self) -> Path:
        return Path.home() / ".cache"

    def device_model(self) -> str | None:
        return None

    def ram_bytes(self) -> int | None:
        return None

    def disk_bytes(self) -> int | None:
        try:
            return shutil.disk_usage("/").total
        except Exception as exc:
            debug_detection_failure("unknown disk_bytes", exc)
            return None

    def gpu_accelerators(self) -> tuple[list[str], str | None]:
        return [], None

    def gpu_accelerator_for_offload(self) -> str:
        return "cpu"
