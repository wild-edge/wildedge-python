"""Hardware context dataclasses and inference-time capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ThermalContext:
    state: str | None = None  # nominal|fair|serious|critical
    state_raw: str | None = None  # platform-native value
    cpu_temp_celsius: float | None = None

    def to_dict(self) -> dict:
        return {
            k: v
            for k, v in {
                "state": self.state,
                "state_raw": self.state_raw,
                "cpu_temp_celsius": self.cpu_temp_celsius,
            }.items()
            if v is not None
        }


@dataclass
class HardwareContext:
    """Device hardware state at the time of inference."""

    thermal: ThermalContext | None = None
    battery_level: float | None = None
    battery_charging: bool | None = None
    memory_available_bytes: int | None = None
    cpu_freq_mhz: int | None = None
    cpu_freq_max_mhz: int | None = None
    accelerator_actual: str | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {}
        if self.thermal is not None:
            thermal_dict = self.thermal.to_dict()
            if thermal_dict:
                d["thermal"] = thermal_dict
        for k, v in [
            ("battery_level", self.battery_level),
            ("battery_charging", self.battery_charging),
            ("memory_available_bytes", self.memory_available_bytes),
            ("cpu_freq_mhz", self.cpu_freq_mhz),
            ("cpu_freq_max_mhz", self.cpu_freq_max_mhz),
            ("accelerator_actual", self.accelerator_actual),
        ]:
            if v is not None:
                d[k] = v
        return d


def capture(accelerator_actual: str | None = None) -> HardwareContext:
    """Read current hardware state. Never raises.

    Pass ``accelerator_actual`` if the runtime reports which accelerator was
    used; it may differ from the accelerator requested at load time.
    """
    # Lazy import avoids a circular dependency: __init__.py imports platform
    # modules which import this file.
    from wildedge.platforms import CURRENT_PLATFORM  # noqa: PLC0415

    ctx = CURRENT_PLATFORM.hardware_context()
    if accelerator_actual is not None:
        ctx.accelerator_actual = accelerator_actual
    return ctx
