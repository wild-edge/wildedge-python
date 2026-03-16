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
from wildedge.platforms.hardware import HardwareContext

try:
    import winreg as _winreg  # type: ignore[import]  # Windows only
except ImportError:
    _winreg = None  # type: ignore[assignment]


class _MEMORYSTATUSEX(ctypes.Structure):
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


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


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

    def meminfo(self) -> tuple[int | None, int | None]:
        """Return (total_bytes, available_bytes) from a single GlobalMemoryStatusEx call."""
        try:
            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return stat.ullTotalPhys, stat.ullAvailPhys
        except Exception as exc:
            debug_detection_failure("windows meminfo", exc)
            return None, None

    def ram_bytes(self) -> int | None:
        return self.meminfo()[0]

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

    def battery(self) -> tuple[float | None, bool | None]:
        """Read battery level and charging state via GetSystemPowerStatus."""
        try:
            status = _SystemPowerStatus()
            if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):  # type: ignore[attr-defined]
                return None, None
            # BatteryLifePercent == 255 means unknown
            level = (
                status.BatteryLifePercent / 100.0
                if status.BatteryLifePercent != 255
                else None
            )
            # ACLineStatus: 0=offline, 1=online, 255=unknown
            charging = (
                bool(status.ACLineStatus == 1) if status.ACLineStatus != 255 else None
            )
            return level, charging
        except Exception as exc:
            debug_detection_failure("windows battery", exc)
            return None, None

    def hardware_context(self) -> HardwareContext:
        _, mem_available = self.meminfo()
        bat_level, bat_charging = self.battery()
        return HardwareContext(
            memory_available_bytes=mem_available,
            battery_level=bat_level,
            battery_charging=bat_charging,
        )
