from __future__ import annotations

from pathlib import Path

import pytest

from wildedge.platforms.hardware import ThermalContext
from wildedge.platforms.linux import (
    CPU_THERMAL_ZONE_TYPES,
    TRIP_POINT_STATES,
    LinuxPlatform,
)


def test_meminfo_parses_total_and_available(monkeypatch):
    content = "MemTotal:       16384000 kB\nMemFree:         1000000 kB\nMemAvailable:    8192000 kB\n"
    monkeypatch.setattr(
        "builtins.open",
        lambda p, **kw: (
            __import__("io").StringIO(content)
            if p == "/proc/meminfo"
            else open(p, **kw)
        ),
    )
    total, available = LinuxPlatform().meminfo()
    assert total == 16384000 * 1024
    assert available == 8192000 * 1024


def test_ram_bytes_delegates_to_meminfo(monkeypatch):
    monkeypatch.setattr(
        LinuxPlatform, "meminfo", lambda self: (8_000_000_000, 4_000_000_000)
    )
    assert LinuxPlatform().ram_bytes() == 8_000_000_000


def test_cpu_freq_converts_khz_to_mhz(tmp_path, monkeypatch):
    cpufreq = tmp_path / "cpufreq"
    cpufreq.mkdir()
    (cpufreq / "scaling_cur_freq").write_text("2400000\n")
    (cpufreq / "cpuinfo_max_freq").write_text("3200000\n")

    monkeypatch.setattr(
        "wildedge.platforms.linux.Path",
        lambda p: cpufreq if str(p).endswith("cpu0/cpufreq") else Path(p),
    )
    cur, max_f = LinuxPlatform().cpu_freq()
    assert cur == 2400
    assert max_f == 3200


def test_battery_reads_capacity_and_status(tmp_path, monkeypatch):
    ps = tmp_path / "power_supply"
    bat = ps / "BAT0"
    bat.mkdir(parents=True)
    (bat / "capacity").write_text("78\n")
    (bat / "status").write_text("Discharging\n")
    monkeypatch.setattr(
        "wildedge.platforms.linux.Path",
        lambda p: ps if p == "/sys/class/power_supply" else Path(p),
    )
    level, charging = LinuxPlatform().battery()
    assert level == pytest.approx(0.78)
    assert charging is False


def test_battery_charging_status(tmp_path, monkeypatch):
    ps = tmp_path / "power_supply"
    bat = ps / "BAT0"
    bat.mkdir(parents=True)
    (bat / "capacity").write_text("95\n")
    (bat / "status").write_text("Charging\n")
    monkeypatch.setattr(
        "wildedge.platforms.linux.Path",
        lambda p: ps if p == "/sys/class/power_supply" else Path(p),
    )
    _, charging = LinuxPlatform().battery()
    assert charging is True


def test_battery_no_supply_returns_none(tmp_path, monkeypatch):
    ps = tmp_path / "empty_ps"
    ps.mkdir()
    monkeypatch.setattr(
        "wildedge.platforms.linux.Path",
        lambda p: ps if p == "/sys/class/power_supply" else Path(p),
    )
    level, charging = LinuxPlatform().battery()
    assert level is None and charging is None


def test_thermal_nominal(tmp_path, monkeypatch):
    zone = tmp_path / "thermal_zone0"
    zone.mkdir()
    (zone / "type").write_text("x86_pkg_temp\n")
    (zone / "temp").write_text("45000\n")
    (zone / "trip_point_0_type").write_text("passive\n")
    (zone / "trip_point_0_temp").write_text("80000\n")
    (zone / "trip_point_1_type").write_text("critical\n")
    (zone / "trip_point_1_temp").write_text("100000\n")
    monkeypatch.setattr(
        "wildedge.platforms.linux.Path",
        lambda p: tmp_path if p == "/sys/class/thermal" else Path(p),
    )
    ctx = LinuxPlatform().thermal()
    assert ctx is not None
    assert ctx.state == "nominal"
    assert ctx.state_raw == "active"
    assert ctx.cpu_temp_celsius == pytest.approx(45.0)


def test_thermal_serious(tmp_path, monkeypatch):
    zone = tmp_path / "thermal_zone0"
    zone.mkdir()
    (zone / "type").write_text("cpu\n")
    (zone / "temp").write_text("90000\n")
    (zone / "trip_point_0_type").write_text("passive\n")
    (zone / "trip_point_0_temp").write_text("80000\n")
    (zone / "trip_point_1_type").write_text("hot\n")
    (zone / "trip_point_1_temp").write_text("85000\n")
    (zone / "trip_point_2_type").write_text("critical\n")
    (zone / "trip_point_2_temp").write_text("100000\n")
    monkeypatch.setattr(
        "wildedge.platforms.linux.Path",
        lambda p: tmp_path if p == "/sys/class/thermal" else Path(p),
    )
    ctx = LinuxPlatform().thermal()
    assert ctx is not None
    assert ctx.state == "serious"
    assert ctx.state_raw == "hot"


def test_thermal_skips_non_cpu_zone(tmp_path, monkeypatch):
    zone = tmp_path / "thermal_zone0"
    zone.mkdir()
    (zone / "type").write_text("iwlwifi_1\n")
    (zone / "temp").write_text("50000\n")
    monkeypatch.setattr(
        "wildedge.platforms.linux.Path",
        lambda p: tmp_path if p == "/sys/class/thermal" else Path(p),
    )
    assert LinuxPlatform().thermal() is None


def test_thermal_missing_sysfs_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "wildedge.platforms.linux.Path",
        lambda p: tmp_path / "no_thermal" if p == "/sys/class/thermal" else Path(p),
    )
    assert LinuxPlatform().thermal() is None


def test_hardware_context_assembles_fields(monkeypatch):
    monkeypatch.setattr(LinuxPlatform, "meminfo", lambda self: (None, 2_000_000_000))
    monkeypatch.setattr(LinuxPlatform, "battery", lambda self: (0.5, True))
    monkeypatch.setattr(LinuxPlatform, "cpu_freq", lambda self: (1800, 3200))
    monkeypatch.setattr(
        LinuxPlatform,
        "thermal",
        lambda self: ThermalContext(state="fair", state_raw="passive"),
    )
    ctx = LinuxPlatform().hardware_context()
    assert ctx.memory_available_bytes == 2_000_000_000
    assert ctx.battery_level == pytest.approx(0.5)
    assert ctx.battery_charging is True
    assert ctx.cpu_freq_mhz == 1800
    assert ctx.cpu_freq_max_mhz == 3200
    assert ctx.thermal.state == "fair"


def test_cpu_thermal_zone_types_is_tuple_of_strings():
    assert isinstance(CPU_THERMAL_ZONE_TYPES, tuple)
    assert all(isinstance(k, str) for k in CPU_THERMAL_ZONE_TYPES)


def test_trip_point_states_covers_all_severity_levels():
    trip_types = {row[0] for row in TRIP_POINT_STATES}
    assert {"critical", "hot", "passive"} <= trip_types


@pytest.mark.requires_linux
def test_real_meminfo():
    total, available = LinuxPlatform().meminfo()
    assert isinstance(total, int) and total > 0
    assert isinstance(available, int) and 0 < available <= total


@pytest.mark.requires_linux
def test_real_cpu_freq():
    cur, max_f = LinuxPlatform().cpu_freq()
    if cur is not None:
        assert 0 < cur <= 10_000
    if max_f is not None:
        assert 0 < max_f <= 10_000
    if cur is not None and max_f is not None:
        assert cur <= max_f


@pytest.mark.requires_linux
def test_real_battery():
    level, charging = LinuxPlatform().battery()
    if level is not None:
        assert 0.0 <= level <= 1.0
    if charging is not None:
        assert isinstance(charging, bool)


@pytest.mark.requires_linux
def test_real_thermal():
    ctx = LinuxPlatform().thermal()
    if ctx is not None:
        assert ctx.state in ("nominal", "fair", "serious", "critical")
        assert isinstance(ctx.state_raw, str)
        if ctx.cpu_temp_celsius is not None:
            assert -10 <= ctx.cpu_temp_celsius <= 150
