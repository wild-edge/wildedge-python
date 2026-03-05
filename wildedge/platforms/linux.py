from __future__ import annotations

import shutil
from pathlib import Path

from wildedge.platforms.base import (
    cuda_device_count,
    debug_detection_failure,
    hip_device_count,
    nvml_gpu_name,
)


class LinuxPlatform:
    wire_type = "linux"

    def config_base(self) -> Path:
        return Path.home() / ".config"

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

    def ram_bytes(self) -> int | None:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except Exception as exc:
            debug_detection_failure("linux ram_bytes", exc)
        return None

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
