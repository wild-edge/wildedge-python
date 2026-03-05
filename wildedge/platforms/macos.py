from __future__ import annotations

import ctypes
import platform
import shutil
from pathlib import Path

from wildedge.platforms.base import debug_detection_failure


class MacOSPlatform:
    wire_type = "macos"

    def config_base(self) -> Path:
        return Path.home() / ".config"

    def device_model(self) -> str | None:
        try:
            buf = ctypes.create_string_buffer(128)
            size = ctypes.c_size_t(ctypes.sizeof(buf))
            ret = ctypes.CDLL(None).sysctlbyname(
                b"hw.model", buf, ctypes.byref(size), None, 0
            )
            if ret == 0:
                val = buf.value.decode().strip()
                return val or None
        except Exception as exc:
            debug_detection_failure("macos device_model", exc)
        return None

    def ram_bytes(self) -> int | None:
        try:
            buf = ctypes.c_uint64(0)
            size = ctypes.c_size_t(ctypes.sizeof(buf))
            ret = ctypes.CDLL(None).sysctlbyname(
                b"hw.memsize", ctypes.byref(buf), ctypes.byref(size), None, 0
            )
            return buf.value if ret == 0 else None
        except Exception as exc:
            debug_detection_failure("macos ram_bytes", exc)
            return None

    def disk_bytes(self) -> int | None:
        try:
            return shutil.disk_usage("/").total
        except Exception as exc:
            debug_detection_failure("macos disk_bytes", exc)
            return None

    def gpu_accelerators(self) -> tuple[list[str], str | None]:
        if platform.machine() == "arm64":
            return ["mps"], None
        return [], None

    def gpu_accelerator_for_offload(self) -> str:
        return "mps" if platform.machine() == "arm64" else "cpu"
