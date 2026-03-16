from __future__ import annotations

import pytest

from wildedge.platforms.macos import MacOSPlatform


def test_meminfo_computes_available(monkeypatch):
    from wildedge.platforms import macos as macos_mod

    monkeypatch.setattr(
        macos_mod,
        "_sysctl_uint64",
        lambda name: 16_000_000_000 if name == b"hw.memsize" else None,
    )
    monkeypatch.setattr(
        macos_mod,
        "_sysctl_uint32",
        lambda name: {b"hw.pagesize": 4096, b"vm.page_free_count": 100_000}.get(name),
    )
    total, available = MacOSPlatform().meminfo()
    assert total == 16_000_000_000
    assert available == 4096 * 100_000


def test_meminfo_available_none_when_page_size_missing(monkeypatch):
    from wildedge.platforms import macos as macos_mod

    monkeypatch.setattr(macos_mod, "_sysctl_uint64", lambda name: 16_000_000_000)
    monkeypatch.setattr(macos_mod, "_sysctl_uint32", lambda name: None)
    _, available = MacOSPlatform().meminfo()
    assert available is None


def test_ram_bytes_delegates_to_meminfo(monkeypatch):
    monkeypatch.setattr(
        MacOSPlatform, "meminfo", lambda self: (8_000_000_000, 1_000_000_000)
    )
    assert MacOSPlatform().ram_bytes() == 8_000_000_000


def test_hardware_context_fields(monkeypatch):
    monkeypatch.setattr(MacOSPlatform, "meminfo", lambda self: (None, 3_000_000_000))
    monkeypatch.setattr(MacOSPlatform, "battery", lambda self: (0.8, False))
    ctx = MacOSPlatform().hardware_context()
    assert ctx.memory_available_bytes == 3_000_000_000
    assert ctx.battery_level == pytest.approx(0.8)
    assert ctx.battery_charging is False
    assert ctx.cpu_freq_mhz is None
    assert ctx.thermal is None


@pytest.mark.requires_macos
def test_real_meminfo():
    total, available = MacOSPlatform().meminfo()
    assert isinstance(total, int) and total > 0
    if available is not None:
        assert 0 < available <= total


@pytest.mark.requires_macos
def test_real_battery():
    level, charging = MacOSPlatform().battery()
    if level is not None:
        assert 0.0 <= level <= 1.0
    if charging is not None:
        assert isinstance(charging, bool)
