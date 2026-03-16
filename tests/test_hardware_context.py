from __future__ import annotations

from wildedge.events.inference import InferenceEvent
from wildedge.platforms.hardware import HardwareContext, ThermalContext, capture
from wildedge.platforms.linux import LinuxPlatform


def test_thermal_context_to_dict_omits_none():
    t = ThermalContext(state="fair", state_raw="passive")
    assert t.to_dict() == {"state": "fair", "state_raw": "passive"}


def test_thermal_context_to_dict_includes_temp():
    t = ThermalContext(state="nominal", state_raw="active", cpu_temp_celsius=42.0)
    assert t.to_dict()["cpu_temp_celsius"] == 42.0


def test_hardware_context_to_dict_omits_none_fields():
    hw = HardwareContext(memory_available_bytes=1_000_000)
    assert hw.to_dict() == {"memory_available_bytes": 1_000_000}


def test_hardware_context_to_dict_includes_thermal():
    hw = HardwareContext(
        thermal=ThermalContext(
            state="nominal", state_raw="active", cpu_temp_celsius=42.0
        ),
        cpu_freq_mhz=2400,
        cpu_freq_max_mhz=3200,
    )
    d = hw.to_dict()
    assert d["thermal"] == {
        "state": "nominal",
        "state_raw": "active",
        "cpu_temp_celsius": 42.0,
    }
    assert d["cpu_freq_mhz"] == 2400


def test_hardware_context_empty_thermal_not_serialised():
    assert "thermal" not in HardwareContext(thermal=ThermalContext()).to_dict()


def test_hardware_context_accelerator_actual_serialised():
    assert (
        HardwareContext(accelerator_actual="cpu").to_dict()["accelerator_actual"]
        == "cpu"
    )


def test_inference_event_hardware_included_in_dict():
    hw = HardwareContext(memory_available_bytes=512_000_000, accelerator_actual="mps")
    inference = InferenceEvent(model_id="m1", duration_ms=10, hardware=hw).to_dict()[
        "inference"
    ]
    assert inference["hardware"]["memory_available_bytes"] == 512_000_000
    assert inference["hardware"]["accelerator_actual"] == "mps"


def test_inference_event_no_hardware_key_when_none():
    assert (
        "hardware"
        not in InferenceEvent(model_id="m1", duration_ms=10).to_dict()["inference"]
    )


def test_capture_sets_accelerator_actual(monkeypatch):
    monkeypatch.setattr("wildedge.platforms.CURRENT_PLATFORM", LinuxPlatform())
    monkeypatch.setattr(
        LinuxPlatform, "hardware_context", lambda self: HardwareContext()
    )
    assert capture(accelerator_actual="gpu").accelerator_actual == "gpu"


def test_capture_preserves_existing_accelerator_when_not_passed(monkeypatch):
    monkeypatch.setattr("wildedge.platforms.CURRENT_PLATFORM", LinuxPlatform())
    monkeypatch.setattr(
        LinuxPlatform,
        "hardware_context",
        lambda self: HardwareContext(accelerator_actual="mps"),
    )
    assert capture().accelerator_actual == "mps"
