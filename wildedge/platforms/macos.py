from __future__ import annotations

import ctypes
import platform
import shutil
from pathlib import Path

from wildedge.platforms.base import debug_detection_failure
from wildedge.platforms.hardware import HardwareContext

_libc: ctypes.CDLL | None = None


def _get_libc() -> ctypes.CDLL:
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(None)
    return _libc


_CF_STRING_ENCODING_UTF8 = 0x08000100
_CF_NUMBER_SINT32_TYPE = 3


def _sysctl_uint32(name: bytes) -> int | None:
    try:
        buf = ctypes.c_uint32(0)
        size = ctypes.c_size_t(ctypes.sizeof(buf))
        ret = _get_libc().sysctlbyname(
            name, ctypes.byref(buf), ctypes.byref(size), None, 0
        )
        return int(buf.value) if ret == 0 else None
    except Exception as exc:
        debug_detection_failure(f"macos sysctl_uint32({name!r})", exc)
        return None


def _sysctl_uint64(name: bytes) -> int | None:
    try:
        buf = ctypes.c_uint64(0)
        size = ctypes.c_size_t(ctypes.sizeof(buf))
        ret = _get_libc().sysctlbyname(
            name, ctypes.byref(buf), ctypes.byref(size), None, 0
        )
        return int(buf.value) if ret == 0 else None
    except Exception as exc:
        debug_detection_failure(f"macos sysctl_uint64({name!r})", exc)
        return None


class MacOSPlatform:
    wire_type = "macos"

    def config_base(self) -> Path:
        return Path.home() / ".config"

    def state_base(self) -> Path:
        return Path.home() / "Library" / "Application Support"

    def cache_base(self) -> Path:
        return Path.home() / "Library" / "Caches"

    def device_model(self) -> str | None:
        try:
            buf = ctypes.create_string_buffer(128)
            size = ctypes.c_size_t(ctypes.sizeof(buf))
            ret = _get_libc().sysctlbyname(
                b"hw.model", buf, ctypes.byref(size), None, 0
            )
            if ret == 0:
                val = buf.value.decode().strip()
                return val or None
        except Exception as exc:
            debug_detection_failure("macos device_model", exc)
        return None

    def meminfo(self) -> tuple[int | None, int | None]:
        """Return (total_bytes, available_bytes) from sysctl."""
        total = _sysctl_uint64(b"hw.memsize")
        page_size = _sysctl_uint32(b"hw.pagesize")
        free_count = _sysctl_uint32(b"vm.page_free_count")
        available = (
            free_count * page_size
            if free_count is not None and page_size is not None
            else None
        )
        return total, available

    def ram_bytes(self) -> int | None:
        return self.meminfo()[0]

    def disk_bytes(self) -> int | None:
        try:
            return shutil.disk_usage("/").total
        except Exception as exc:
            debug_detection_failure("macos disk_bytes", exc)
            return None

    def os_version(self) -> str | None:
        try:
            ver = platform.mac_ver()[0]
            return ver or None
        except Exception as exc:
            debug_detection_failure("macos os_version", exc)
            return None

    def gpu_accelerators(self) -> tuple[list[str], str | None]:
        if platform.machine() == "arm64":
            return ["mps"], None
        return [], None

    def gpu_accelerator_for_offload(self) -> str:
        return "mps" if platform.machine() == "arm64" else "cpu"

    def battery(self) -> tuple[float | None, bool | None]:
        """Read battery level and charging state via IOKit AppleSmartBattery."""
        try:
            iokit = ctypes.cdll.LoadLibrary(
                "/System/Library/Frameworks/IOKit.framework/IOKit"
            )
            cf = ctypes.cdll.LoadLibrary(
                "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
            )

            iokit.IOServiceMatching.restype = ctypes.c_void_p
            iokit.IOIteratorNext.restype = ctypes.c_uint32
            iokit.IORegistryEntryCreateCFProperties.restype = ctypes.c_int32
            iokit.IOObjectRelease.argtypes = [ctypes.c_uint32]

            cf.CFDictionaryGetValue.restype = ctypes.c_void_p
            cf.CFNumberGetValue.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int32,
                ctypes.c_void_p,
            ]
            cf.CFBooleanGetValue.restype = ctypes.c_bool
            cf.CFStringCreateWithCString.restype = ctypes.c_void_p
            cf.CFRelease.argtypes = [ctypes.c_void_p]

            matching = iokit.IOServiceMatching(b"AppleSmartBattery")
            if not matching:
                return None, None

            iterator = ctypes.c_uint32(0)
            if (
                iokit.IOServiceGetMatchingServices(
                    ctypes.c_uint32(0),
                    ctypes.c_void_p(matching),
                    ctypes.byref(iterator),
                )
                != 0
            ):
                return None, None

            service = iokit.IOIteratorNext(iterator)
            iokit.IOObjectRelease(iterator)
            if not service:
                return None, None

            props = ctypes.c_void_p(0)
            ret = iokit.IORegistryEntryCreateCFProperties(
                service, ctypes.byref(props), None, 0
            )
            iokit.IOObjectRelease(service)
            if ret != 0:
                return None, None

            def cfstr(s: bytes) -> ctypes.c_void_p:
                return ctypes.c_void_p(
                    cf.CFStringCreateWithCString(None, s, _CF_STRING_ENCODING_UTF8)
                )

            cap_key = cfstr(b"CurrentCapacity")
            max_key = cfstr(b"MaxCapacity")
            chg_key = cfstr(b"IsCharging")

            level: float | None = None
            charging: bool | None = None

            cap_ref = cf.CFDictionaryGetValue(props, cap_key)
            max_ref = cf.CFDictionaryGetValue(props, max_key)
            if cap_ref and max_ref:
                cur = ctypes.c_int32(0)
                mx = ctypes.c_int32(0)
                cf.CFNumberGetValue(
                    ctypes.c_void_p(cap_ref), _CF_NUMBER_SINT32_TYPE, ctypes.byref(cur)
                )
                cf.CFNumberGetValue(
                    ctypes.c_void_p(max_ref), _CF_NUMBER_SINT32_TYPE, ctypes.byref(mx)
                )
                if mx.value > 0:
                    level = cur.value / mx.value

            chg_ref = cf.CFDictionaryGetValue(props, chg_key)
            if chg_ref:
                charging = bool(cf.CFBooleanGetValue(ctypes.c_void_p(chg_ref)))

            cf.CFRelease(props)
            for k in (cap_key, max_key, chg_key):
                if k:
                    cf.CFRelease(k)

            return level, charging
        except Exception as exc:
            debug_detection_failure("macos battery", exc)
            return None, None

    def hardware_context(self) -> HardwareContext:
        _, mem_available = self.meminfo()
        bat_level, bat_charging = self.battery()
        return HardwareContext(
            memory_available_bytes=mem_available,
            battery_level=bat_level,
            battery_charging=bat_charging,
            # cpu_freq not available on macOS (no public API on Apple Silicon)
            # thermal not available via public Python API
        )
