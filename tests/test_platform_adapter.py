from __future__ import annotations

import pytest

from wildedge.platforms.hardware import HardwareContext
from wildedge.platforms.linux import LinuxPlatform
from wildedge.platforms.macos import MacOSPlatform
from wildedge.platforms.windows import WindowsPlatform

PLATFORMS = [
    pytest.param(LinuxPlatform, id="linux", marks=pytest.mark.requires_linux),
    pytest.param(MacOSPlatform, id="macos", marks=pytest.mark.requires_macos),
    pytest.param(WindowsPlatform, id="windows", marks=pytest.mark.requires_windows),
]


@pytest.mark.parametrize("platform_cls", PLATFORMS)
def test_real_device_model(platform_cls):
    model = platform_cls().device_model()
    if model is not None:
        assert isinstance(model, str) and model


@pytest.mark.parametrize("platform_cls", PLATFORMS)
def test_real_os_version(platform_cls):
    ver = platform_cls().os_version()
    if ver is not None:
        assert isinstance(ver, str) and ver


@pytest.mark.parametrize("platform_cls", PLATFORMS)
def test_real_ram_bytes(platform_cls):
    total = platform_cls().ram_bytes()
    assert isinstance(total, int) and total > 0


@pytest.mark.parametrize("platform_cls", PLATFORMS)
def test_real_disk_bytes(platform_cls):
    total = platform_cls().disk_bytes()
    assert isinstance(total, int) and total > 0


@pytest.mark.parametrize("platform_cls", PLATFORMS)
def test_real_gpu_accelerators(platform_cls):
    accs, gpu_name = platform_cls().gpu_accelerators()
    assert isinstance(accs, list)
    assert all(isinstance(a, str) for a in accs)
    if gpu_name is not None:
        assert isinstance(gpu_name, str) and gpu_name


@pytest.mark.parametrize("platform_cls", PLATFORMS)
def test_real_gpu_accelerator_for_offload(platform_cls):
    offload = platform_cls().gpu_accelerator_for_offload()
    assert isinstance(offload, str) and offload


@pytest.mark.parametrize("platform_cls", PLATFORMS)
def test_real_hardware_context(platform_cls):
    ctx = platform_cls().hardware_context()
    assert isinstance(ctx, HardwareContext)
