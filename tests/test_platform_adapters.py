from __future__ import annotations

from wildedge.platforms.base import cuda_device_count
from wildedge.platforms.linux import LinuxPlatform
from wildedge.platforms.macos import MacOSPlatform
from wildedge.platforms.unknown import UnknownPlatform
from wildedge.platforms.windows import WindowsPlatform


def test_linux_gpu_accelerators_cuda_and_rocm_counts(monkeypatch):
    platform = LinuxPlatform()
    monkeypatch.setattr("wildedge.platforms.linux.cuda_device_count", lambda _: 1)
    monkeypatch.setattr("wildedge.platforms.linux.hip_device_count", lambda _: 1)
    monkeypatch.setattr("wildedge.platforms.linux.nvml_gpu_name", lambda _: "A100")
    accelerators, gpu_name = platform.gpu_accelerators()
    assert accelerators == ["cuda", "rocm"]
    assert gpu_name == "A100"


def test_windows_gpu_accelerators_uses_device_count(monkeypatch):
    platform = WindowsPlatform()
    monkeypatch.setattr("wildedge.platforms.windows.cuda_device_count", lambda _: 0)
    accelerators, gpu_name = platform.gpu_accelerators()
    assert accelerators == []
    assert gpu_name is None


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


def test_macos_os_version_returns_mac_ver(monkeypatch):
    monkeypatch.setattr("platform.mac_ver", lambda: ("15.3.0", ("", "", ""), ""))
    assert MacOSPlatform().os_version() == "15.3.0"


def test_macos_os_version_returns_none_on_empty(monkeypatch):
    monkeypatch.setattr("platform.mac_ver", lambda: ("", ("", "", ""), ""))
    assert MacOSPlatform().os_version() is None


def test_linux_os_version_reads_os_release(monkeypatch, tmp_path):
    os_release = tmp_path / "os-release"
    os_release.write_text(
        'ID=ubuntu\nPRETTY_NAME="Ubuntu 22.04.3 LTS"\nVERSION_ID="22.04"\n'
    )
    monkeypatch.setattr(
        "wildedge.platforms.linux.Path",
        lambda p: (
            os_release if p == "/etc/os-release" else __import__("pathlib").Path(p)
        ),
    )
    assert LinuxPlatform().os_version() == "Ubuntu 22.04.3 LTS"


def test_linux_os_version_falls_back_to_platform_version(monkeypatch, tmp_path):
    missing = tmp_path / "no-os-release"
    monkeypatch.setattr(
        "wildedge.platforms.linux.Path",
        lambda p: missing if p == "/etc/os-release" else __import__("pathlib").Path(p),
    )
    monkeypatch.setattr(
        "wildedge.platforms.linux.platform.version",
        lambda: "#1 SMP Fri Jan 1 00:00:00 UTC 2021",
    )
    assert LinuxPlatform().os_version() == "#1 SMP Fri Jan 1 00:00:00 UTC 2021"


def test_windows_os_version_returns_platform_version(monkeypatch):
    monkeypatch.setattr(
        "wildedge.platforms.windows.platform.version", lambda: "10.0.22621"
    )
    assert WindowsPlatform().os_version() == "10.0.22621"


def test_unknown_os_version_returns_platform_version(monkeypatch):
    monkeypatch.setattr(
        "wildedge.platforms.unknown.platform.version", lambda: "some-kernel-version"
    )
    assert UnknownPlatform().os_version() == "some-kernel-version"


def test_platform_adapters_expose_state_and_cache_paths():
    for adapter in (
        LinuxPlatform(),
        MacOSPlatform(),
        WindowsPlatform(),
        UnknownPlatform(),
    ):
        assert adapter.state_base()
        assert adapter.cache_base()
