from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

from wildedge.platforms.base import (
    cuda_device_count,
    debug_detection_failure,
    hip_device_count,
    nvml_gpu_name,
)
from wildedge.platforms.hardware import HardwareContext, ThermalContext

# Thermal zone type substrings that identify a CPU zone.
CPU_THERMAL_ZONE_TYPES = ("cpu", "x86_pkg", "acpi", "soc", "pkg")

# Maps sysfs trip-point type -> (normalized state, state_raw).
# Evaluated in priority order (highest severity first).
TRIP_POINT_STATES: tuple[tuple[str, str, str], ...] = (
    ("critical", "critical", "critical"),
    ("hot", "serious", "hot"),
    ("passive", "fair", "passive"),
)
TRIP_POINT_DEFAULT = ("nominal", "active")


class LinuxPlatform:
    wire_type = "linux"

    def config_base(self) -> Path:
        return Path.home() / ".config"

    def state_base(self) -> Path:
        return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))

    def cache_base(self) -> Path:
        return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

    def device_model(self) -> str | None:
        for path in (
            "/sys/class/dmi/id/product_name",
            "/sys/firmware/devicetree/base/model",
        ):
            try:
                val = Path(path).read_text().strip().rstrip("\x00")
                if val:
                    return val
            except Exception as exc:
                debug_detection_failure(f"linux device_model ({path})", exc)
        return None

    def os_version(self) -> str | None:
        try:
            path = Path("/etc/os-release")
            if path.exists():
                for line in path.read_text().splitlines():
                    if line.startswith("PRETTY_NAME="):
                        return line.split("=", 1)[1].strip().strip('"') or None
        except Exception as exc:
            debug_detection_failure("linux os_version", exc)
        try:
            return platform.version() or None
        except Exception:
            return None

    def meminfo(self) -> tuple[int | None, int | None]:
        """Return (total_bytes, available_bytes) from a single /proc/meminfo read."""
        total: int | None = None
        available: int | None = None
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) * 1024
                    elif line.startswith("MemAvailable:"):
                        available = int(line.split()[1]) * 1024
                    if total is not None and available is not None:
                        break
        except Exception as exc:
            debug_detection_failure("linux meminfo", exc)
        return total, available

    def ram_bytes(self) -> int | None:
        return self.meminfo()[0]

    def disk_bytes(self) -> int | None:
        try:
            return shutil.disk_usage("/").total
        except Exception as exc:
            debug_detection_failure("linux disk_bytes", exc)
            return None

    def gpu_accelerators(self) -> tuple[list[str], str | None]:
        accelerators = []
        gpu_name = None
        if cuda_device_count("libcuda.so.1") > 0:
            accelerators.append("cuda")
            gpu_name = nvml_gpu_name("libnvidia-ml.so.1")
        if hip_device_count("libamdhip64.so") > 0:
            accelerators.append("rocm")
        return accelerators, gpu_name

    def gpu_accelerator_for_offload(self) -> str:
        accs, _ = self.gpu_accelerators()
        return accs[0] if accs else "cpu"

    def hardware_context(self) -> HardwareContext:
        bat_level, bat_charging = self.battery()
        cpu_cur, cpu_max = self.cpu_freq()
        _, mem_available = self.meminfo()
        return HardwareContext(
            thermal=self.thermal(),
            battery_level=bat_level,
            battery_charging=bat_charging,
            memory_available_bytes=mem_available,
            cpu_freq_mhz=cpu_cur,
            cpu_freq_max_mhz=cpu_max,
        )

    def cpu_freq(self) -> tuple[int | None, int | None]:
        """Read current and max CPU frequency (MHz) from cpufreq for cpu0."""
        base = Path("/sys/devices/system/cpu/cpu0/cpufreq")
        try:
            cur_path = base / "scaling_cur_freq"
            max_path = base / "cpuinfo_max_freq"
            cur = (
                int(cur_path.read_text().strip()) // 1000 if cur_path.exists() else None
            )
            max_f = (
                int(max_path.read_text().strip()) // 1000 if max_path.exists() else None
            )
            return cur, max_f
        except Exception as exc:
            debug_detection_failure("linux cpu_freq", exc)
            return None, None

    def battery(self) -> tuple[float | None, bool | None]:
        """Read level (0.0-1.0) and charging state from the first BAT* supply."""
        try:
            for bat in sorted(Path("/sys/class/power_supply").glob("BAT*")):
                cap_path = bat / "capacity"
                status_path = bat / "status"
                level = (
                    int(cap_path.read_text().strip()) / 100.0
                    if cap_path.exists()
                    else None
                )
                status_raw = (
                    status_path.read_text().strip().lower()
                    if status_path.exists()
                    else None
                )
                charging = status_raw in ("charging", "full") if status_raw else None
                return level, charging
        except Exception as exc:
            debug_detection_failure("linux battery", exc)
        return None, None

    def thermal(self) -> ThermalContext | None:
        """Read the first CPU thermal zone and map trip points to a normalized state."""
        try:
            thermal_dir = Path("/sys/class/thermal")
            if not thermal_dir.exists():
                return None

            for zone in sorted(thermal_dir.glob("thermal_zone*")):
                try:
                    zone_type = (zone / "type").read_text().strip().lower()
                except Exception:
                    continue
                if not any(k in zone_type for k in CPU_THERMAL_ZONE_TYPES):
                    continue

                try:
                    temp_c = int((zone / "temp").read_text().strip()) / 1000.0
                except Exception:
                    continue

                trip_temps: dict[str, float] = {}
                for trip_path in sorted(zone.glob("trip_point_*_type")):
                    try:
                        ttype = trip_path.read_text().strip().lower()
                        idx = trip_path.name[len("trip_point_") : -len("_type")]
                        ttemp = (
                            int((zone / f"trip_point_{idx}_temp").read_text().strip())
                            / 1000.0
                        )
                        trip_temps[ttype] = ttemp
                    except Exception:
                        continue

                state, state_raw = TRIP_POINT_DEFAULT
                for trip_type, norm_state, norm_raw in TRIP_POINT_STATES:
                    if trip_type in trip_temps and temp_c >= trip_temps[trip_type]:
                        state, state_raw = norm_state, norm_raw
                        break

                return ThermalContext(
                    state=state, state_raw=state_raw, cpu_temp_celsius=temp_c
                )
        except Exception as exc:
            debug_detection_failure("linux thermal", exc)
        return None
