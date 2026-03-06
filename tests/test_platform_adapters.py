from __future__ import annotations

import pytest

from wildedge.platforms.base import cuda_device_count
from wildedge.platforms.linux import LinuxPlatform
from wildedge.platforms.macos import MacOSPlatform
from wildedge.platforms.unknown import UnknownPlatform
from wildedge.platforms.windows import WindowsPlatform


@pytest.mark.requires_linux
def test_linux_gpu_accelerators_cuda_and_rocm_counts(monkeypatch):
    platform = LinuxPlatform()
    monkeypatch.setattr("wildedge.platforms.linux.cuda_device_count", lambda _: 1)
    monkeypatch.setattr("wildedge.platforms.linux.hip_device_count", lambda _: 1)
    monkeypatch.setattr("wildedge.platforms.linux.nvml_gpu_name", lambda _: "A100")
    accelerators, gpu_name = platform.gpu_accelerators()
    assert accelerators == ["cuda", "rocm"]
    assert gpu_name == "A100"


@pytest.mark.requires_windows
def test_windows_gpu_accelerators_uses_device_count(monkeypatch):
    platform = WindowsPlatform()
    monkeypatch.setattr("wildedge.platforms.windows.cuda_device_count", lambda _: 0)
    accelerators, gpu_name = platform.gpu_accelerators()
    assert accelerators == []
    assert gpu_name is None


@pytest.mark.requires_macos
def test_macos_gpu_accelerators_for_arm64(monkeypatch):
    platform = MacOSPlatform()
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    accelerators, gpu_name = platform.gpu_accelerators()
    assert accelerators == ["mps"]
    assert gpu_name is None


def test_unknown_platform_disk_bytes_failure_returns_none(monkeypatch):
    platform = UnknownPlatform()

    def raise_disk_error(_path):
        raise OSError("disk failure")

    monkeypatch.setattr("shutil.disk_usage", raise_disk_error)
    assert platform.disk_bytes() is None


def test_base_cuda_device_count_nonzero(monkeypatch):
    class FakeCudaLib:
        def cuInit(self, _flags):  # noqa: N802
            return 0

        def cuDeviceGetCount(self, ptr):  # noqa: N802
            ptr._obj.value = 2  # noqa: SLF001
            return 0

    monkeypatch.setattr("ctypes.CDLL", lambda _name: FakeCudaLib())
    assert cuda_device_count("libcuda.so.1") == 2


def test_base_cuda_device_count_init_failure_returns_zero(monkeypatch):
    class FakeCudaLib:
        def cuInit(self, _flags):  # noqa: N802
            return 1

    monkeypatch.setattr("ctypes.CDLL", lambda _name: FakeCudaLib())
    assert cuda_device_count("libcuda.so.1") == 0


def test_platform_adapters_expose_state_and_cache_paths():
    for adapter in (
        LinuxPlatform(),
        MacOSPlatform(),
        WindowsPlatform(),
        UnknownPlatform(),
    ):
        assert adapter.state_base()
        assert adapter.cache_base()
