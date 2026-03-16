from __future__ import annotations

import pytest

from wildedge.platforms.windows import WindowsPlatform


def test_ram_bytes_delegates_to_meminfo(monkeypatch):
    monkeypatch.setattr(
        WindowsPlatform, "meminfo", lambda self: (16_000_000_000, 4_000_000_000)
    )
    assert WindowsPlatform().ram_bytes() == 16_000_000_000


def test_hardware_context_fields(monkeypatch):
    monkeypatch.setattr(WindowsPlatform, "meminfo", lambda self: (None, 6_000_000_000))
    monkeypatch.setattr(WindowsPlatform, "battery", lambda self: (0.6, True))
    ctx = WindowsPlatform().hardware_context()
    assert ctx.memory_available_bytes == 6_000_000_000
    assert ctx.battery_level == pytest.approx(0.6)
    assert ctx.battery_charging is True


@pytest.mark.requires_windows
def test_real_meminfo():
    total, available = WindowsPlatform().meminfo()
    assert isinstance(total, int) and total > 0
    assert isinstance(available, int) and 0 < available <= total


@pytest.mark.requires_windows
def test_real_battery():
    level, charging = WindowsPlatform().battery()
    if level is not None:
        assert 0.0 <= level <= 1.0
    if charging is not None:
        assert isinstance(charging, bool)
