from __future__ import annotations

import ctypes
import os
import platform
import shutil
from pathlib import Path

from wildedge.platforms.base import (
    cuda_device_count,
    debug_detection_failure,
    nvml_gpu_name,
)

try:
    import winreg as _winreg  # type: ignore[import]  # Windows only
except ImportError:
    _winreg = None  # type: ignore[assignment]


class WindowsPlatform:
    wire_type = "windows"

    def config_base(self) -> Path:
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))

    def state_base(self) -> Path:
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))

    def cache_base(self) -> Path:
        return self.state_base()

    def device_model(self) -> str | None:
        if _winreg is None:
            return None
        try:
            key = _winreg.OpenKey(
                _winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\BIOS",
            )
            model, _ = _winreg.QueryValueEx(key, "SystemProductName")
            _winreg.CloseKey(key)
            return str(model).strip() or None
        except Exception as exc:
            debug_detection_failure("windows device_model", exc)
            return None

    def os_version(self) -> str | None:
        try:
            return platform.version() or None
        except Exception as exc:
            debug_detection_failure("windows os_version", exc)
            return None

    def ram_bytes(self) -> int | None:
        try:

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return stat.ullTotalPhys
        except Exception as exc:
            debug_detection_failure("windows ram_bytes", exc)
            return None

    def disk_bytes(self) -> int | None:
        try:
            drive = os.environ.get("SystemDrive", "C:\\")
            return shutil.disk_usage(drive).total
        except Exception as exc:
            debug_detection_failure("windows disk_bytes", exc)
            return None

    def gpu_accelerators(self) -> tuple[list[str], str | None]:
        if cuda_device_count("nvcuda.dll") > 0:
            return ["cuda"], nvml_gpu_name("nvml.dll")
        return [], None

    def gpu_accelerator_for_offload(self) -> str:
        accs, _ = self.gpu_accelerators()
        return accs[0] if accs else "cpu"
