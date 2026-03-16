from __future__ import annotations

from dataclasses import dataclass, field

from wildedge import constants


@dataclass
class DeviceInfo:
    device_id: str
    device_type: str
    sdk_version: str = constants.SDK_VERSION
    device_model: str | None = None
    os_version: str | None = None
    locale: str | None = None
    timezone: str | None = None
    cpu_arch: str | None = None
    cpu_cores: int | None = None
    ram_total_bytes: int | None = None
    disk_total_bytes: int | None = None
    accelerators: list[str] = field(default_factory=list)
    gpu_name: str | None = None
    app_version: str | None = None

    def to_dict(self) -> dict:
        d = {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "device_model": self.device_model,
            "os_version": self.os_version,
            "sdk_version": self.sdk_version,
            "locale": self.locale,
            "timezone": self.timezone,
            "cpu_arch": self.cpu_arch,
            "cpu_cores": self.cpu_cores,
            "ram_total_bytes": self.ram_total_bytes,
            "disk_total_bytes": self.disk_total_bytes,
            "accelerators": self.accelerators,
            "gpu_name": self.gpu_name,
        }
        if self.app_version is not None:
            d["app_version"] = self.app_version
        return d
