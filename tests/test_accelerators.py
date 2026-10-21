"""Accelerator names are shared across integrations and reported consistently."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from wildedge import accelerators
from wildedge.integrations.common import (
    mlx_accelerator,
    tf_weights_accelerator,
    torch_module_accelerator,
)
from wildedge.integrations.mlx import MlxExtractor


def torch_like(*device_types):
    params = [SimpleNamespace(device=SimpleNamespace(type=t)) for t in device_types]
    return SimpleNamespace(parameters=lambda: iter(params))


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps", "xpu"])
def test_torch_module_reports_its_parameter_device(device_type):
    assert torch_module_accelerator(torch_like(device_type)) == device_type


def test_torch_module_without_parameters_has_no_device():
    assert torch_module_accelerator(torch_like()) is None


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("/job:localhost/replica:0/task:0/device:GPU:0", accelerators.CUDA),
        ("/job:localhost/replica:0/task:0/device:CPU:0", accelerators.CPU),
    ],
)
def test_tf_weights_report_the_platform_gpu(device, expected):
    model = SimpleNamespace(weights=[SimpleNamespace(device=device)])
    with patch(
        "wildedge.platforms.CURRENT_PLATFORM.gpu_accelerator_for_offload",
        return_value=accelerators.CUDA,
    ):
        assert tf_weights_accelerator(model) == expected


def test_tf_model_without_weights_is_cpu():
    assert tf_weights_accelerator(SimpleNamespace(weights=[])) == accelerators.CPU


@pytest.fixture
def fake_mlx(monkeypatch):
    mlx, core = ModuleType("mlx"), ModuleType("mlx.core")
    core.cpu, core.gpu = "cpu-device", "gpu-device"
    core.default = core.gpu
    core.default_device = lambda: core.default
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return core


def test_mlx_reports_the_apple_gpu_by_default(fake_mlx):
    assert mlx_accelerator() == accelerators.MPS


def test_mlx_follows_a_cpu_default_device(fake_mlx):
    fake_mlx.default = fake_mlx.cpu
    assert mlx_accelerator() == accelerators.CPU


def test_mlx_lm_models_report_their_device(fake_mlx):
    class Model:
        def __call__(self, *args):
            return None

    handle = SimpleNamespace(detected_accelerator=accelerators.CPU)
    MlxExtractor().install_hooks(Model(), handle)
    assert handle.detected_accelerator == accelerators.MPS
