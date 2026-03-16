from __future__ import annotations

import ctypes
import hashlib
import hmac
import locale
import os
import platform
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from wildedge import constants
from wildedge.logging import logger
from wildedge.platforms.device_info import DeviceInfo
from wildedge.platforms.hardware import HardwareContext, ThermalContext


class Platform(ABC):
    wire_type: str

    @abstractmethod
    def config_base(self) -> Path: ...

    @abstractmethod
    def state_base(self) -> Path: ...

    @abstractmethod
    def cache_base(self) -> Path: ...

    @abstractmethod
    def device_model(self) -> str | None: ...

    @abstractmethod
    def os_version(self) -> str | None: ...

    @abstractmethod
    def disk_bytes(self) -> int | None: ...

    @abstractmethod
    def gpu_accelerators(self) -> tuple[list[str], str | None]: ...

    @abstractmethod
    def gpu_accelerator_for_offload(self) -> str: ...

    # Optional hardware introspection — subclasses override what they support.

    def meminfo(self) -> tuple[int | None, int | None]:
        return None, None

    def ram_bytes(self) -> int | None:
        return self.meminfo()[0]

    def battery(self) -> tuple[float | None, bool | None]:
        return None, None

    def cpu_freq(self) -> tuple[int | None, int | None]:
        return None, None

    def thermal(self) -> ThermalContext | None:
        return None

    def hardware_context(self) -> HardwareContext:
        _, mem_available = self.meminfo()
        bat_level, bat_charging = self.battery()
        cpu_cur, cpu_max = self.cpu_freq()
        return HardwareContext(
            thermal=self.thermal(),
            battery_level=bat_level,
            battery_charging=bat_charging,
            memory_available_bytes=mem_available,
            cpu_freq_mhz=cpu_cur,
            cpu_freq_max_mhz=cpu_max,
        )

    def get_device_id_path(self) -> Path:
        return self.config_base() / constants.DEVICE_ID_DIR / constants.DEVICE_ID_FILE

    def load_or_create_device_uuid(self) -> str:
        path = self.get_device_id_path()
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

    def detect_device(
        self, api_key: str, app_version: str | None, overrides: dict | None = None
    ) -> DeviceInfo:
        raw_uuid = self.load_or_create_device_uuid()
        device_id = hmac_device_id(api_key, raw_uuid)
        gpu_accs, gpu_name = self.gpu_accelerators()
        info = DeviceInfo(
            app_version=app_version,
            device_id=device_id,
            device_type=self.wire_type,
            device_model=self.device_model(),
            os_version=self.os_version(),
            locale=detect_locale(),
            timezone=detect_timezone(),
            cpu_arch=platform.machine() or None,
            cpu_cores=os.cpu_count(),
            ram_total_bytes=self.ram_bytes(),
            disk_total_bytes=self.disk_bytes(),
            accelerators=["cpu", *gpu_accs],
            gpu_name=gpu_name,
        )
        if overrides:
            for k, v in overrides.items():
                if hasattr(info, k):
                    setattr(info, k, v)
        return info


def hmac_device_id(api_key: str, raw_id: str) -> str:
    return hmac.new(
        key=api_key.encode(),
        msg=raw_id.encode(),
        digestmod=hashlib.sha256,
    ).hexdigest()


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


def debug_detection_failure(context: str, exc: BaseException) -> None:
    logger.debug("wildedge: device detection failed for %s: %s", context, exc)


def nvml_gpu_name(lib_name: str) -> str | None:
    try:
        nvml = ctypes.CDLL(lib_name)
        nvml.nvmlInit()
        handle = ctypes.c_void_p()
        nvml.nvmlDeviceGetHandleByIndex(0, ctypes.byref(handle))
        buf = ctypes.create_string_buffer(96)
        nvml.nvmlDeviceGetName(handle, buf, ctypes.c_uint(96))
        nvml.nvmlShutdown()
        return buf.value.decode()
    except Exception as exc:
        debug_detection_failure(f"nvml gpu name ({lib_name})", exc)
        return None


def cuda_device_count(lib_name: str) -> int:
    try:
        cuda = ctypes.CDLL(lib_name)
        init_rc = cuda.cuInit(0)
        if init_rc != 0:
            debug_detection_failure(f"cuda init ({lib_name})", RuntimeError(init_rc))
            return 0
        count = ctypes.c_int(0)
        count_rc = cuda.cuDeviceGetCount(ctypes.byref(count))
        if count_rc != 0:
            debug_detection_failure(
                f"cuda device count ({lib_name})", RuntimeError(count_rc)
            )
            return 0
        return max(0, int(count.value))
    except OSError as exc:
        debug_detection_failure(f"cuda library load ({lib_name})", exc)
        return 0
    except Exception as exc:
        debug_detection_failure(f"cuda device detection ({lib_name})", exc)
        return 0


def hip_device_count(lib_name: str) -> int:
    try:
        hip = ctypes.CDLL(lib_name)
        count = ctypes.c_int(0)
        rc = hip.hipGetDeviceCount(ctypes.byref(count))
        if rc != 0:
            debug_detection_failure(f"hip device count ({lib_name})", RuntimeError(rc))
            return 0
        return max(0, int(count.value))
    except OSError as exc:
        debug_detection_failure(f"hip library load ({lib_name})", exc)
        return 0
    except Exception as exc:
        debug_detection_failure(f"hip device detection ({lib_name})", exc)
        return 0
