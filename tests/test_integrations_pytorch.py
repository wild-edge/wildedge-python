"""Tests for PyTorch integration extractor."""

from __future__ import annotations

from unittest.mock import MagicMock

from wildedge.integrations.pytorch import (
    PytorchExtractor,
)
from wildedge.integrations.pytorch import (
    _detect_accelerator as torch_detect_accelerator,
)
from wildedge.integrations.pytorch import (
    _detect_quantization as torch_detect_quantization,
)
from wildedge.model import ModelHandle, ModelInfo


def make_handle(publish_spy) -> ModelHandle:
    info = ModelInfo(
        model_name="test",
        model_version="1.0",
        model_source="local",
        model_format="test",
    )
    return ModelHandle(model_id="m", info=info, publish=publish_spy)


class _TorchBase:
    """Looks like torch.nn.Module to the MRO check."""


_TorchBase.__name__ = "Module"
_TorchBase.__module__ = "torch.nn.modules.module"


class _FakeParam:
    class _Device:
        type = "cpu"

    device = _Device()
    dtype = "torch.float32"


class _FakeTorchModel(_TorchBase):
    def parameters(self):
        yield _FakeParam()

    def modules(self):
        return iter([self])

    def register_forward_pre_hook(self, hook):
        self._pre_hook = hook
        return MagicMock()

    def register_forward_hook(self, hook):
        self._post_hook = hook
        return MagicMock()


class TestPytorchExtractor:
    extractor = PytorchExtractor()

    def test_can_handle_torch_module(self):
        assert self.extractor.can_handle(_FakeTorchModel()) is True

    def test_can_handle_rejects_plain_object(self):
        assert self.extractor.can_handle(object()) is False

    def test_detect_accelerator_reads_parameter_device(self):
        model = _FakeTorchModel()
        assert torch_detect_accelerator(model) == "cpu"

    def test_detect_accelerator_cuda(self):
        model = _FakeTorchModel()
        model.parameters = lambda: iter([MagicMock(device=MagicMock(type="cuda"))])
        assert torch_detect_accelerator(model) == "cuda"

    def test_detect_accelerator_no_parameters_falls_back(self):
        model = _FakeTorchModel()
        model.parameters = lambda: iter([])
        assert isinstance(torch_detect_accelerator(model), str)

    def test_detect_quantization_by_module_name(self):
        model = _FakeTorchModel()

        class QuantizedLinear:
            pass

        QuantizedLinear.__module__ = "torch.nn.quantized"
        model.modules = lambda: iter([QuantizedLinear()])
        model.parameters = lambda: iter([])
        assert torch_detect_quantization(model) == "int8"

    def test_detect_quantization_by_param_dtype(self):
        model = _FakeTorchModel()
        model.modules = lambda: iter([])
        model.parameters = lambda: iter([MagicMock(dtype="torch.float16")])
        assert torch_detect_quantization(model) == "f16"

    def test_detect_quantization_by_param_dtype_bf16(self):
        model = _FakeTorchModel()
        model.modules = lambda: iter([])
        model.parameters = lambda: iter([MagicMock(dtype="torch.bfloat16")])
        assert torch_detect_quantization(model) == "bf16"

    def test_detect_quantization_by_param_dtype_qint(self):
        model = _FakeTorchModel()
        model.modules = lambda: iter([])
        model.parameters = lambda: iter([MagicMock(dtype="torch.qint8")])
        assert torch_detect_quantization(model) == "int8"

    def test_detect_quantization_by_param_dtype_quint(self):
        model = _FakeTorchModel()
        model.modules = lambda: iter([])
        model.parameters = lambda: iter([MagicMock(dtype="torch.quint8")])
        assert torch_detect_quantization(model) == "int8"

    def test_detect_quantization_by_param_dtype_int8(self):
        model = _FakeTorchModel()
        model.modules = lambda: iter([])
        model.parameters = lambda: iter([MagicMock(dtype="torch.int8")])
        assert torch_detect_quantization(model) == "int8"

    def test_detect_quantization_returns_none_when_unknown(self):
        model = _FakeTorchModel()
        model.modules = lambda: iter([])
        model.parameters = lambda: iter([MagicMock(dtype="torch.float32")])
        assert torch_detect_quantization(model) is None

    def test_detect_quantization_returns_none_on_exception(self):
        model = _FakeTorchModel()
        model.modules = lambda: iter([])

        def broken_parameters():
            raise RuntimeError("broken params")

        model.parameters = broken_parameters
        assert torch_detect_quantization(model) is None

    def test_extract_info_uses_class_name_as_model_id(self):
        model = _FakeTorchModel()
        model_id, info = self.extractor.extract_info(model, {})
        assert model_id == "_FakeTorchModel"
        assert info.model_format == "pytorch"

    def test_extract_info_override_model_id(self):
        model = _FakeTorchModel()
        model_id, _ = self.extractor.extract_info(model, {"id": "my-resnet"})
        assert model_id == "my-resnet"

    def test_install_hooks_publishes_inference(self, publish_spy):
        model = _FakeTorchModel()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(model, handle)
        model._pre_hook(model, (None,))
        model._post_hook(model, (None,), None)
        assert len(publish_spy.events) == 1
        assert publish_spy.events[0]["event_type"] == "inference"

    def test_install_hooks_sets_detected_accelerator(self, publish_spy):
        model = _FakeTorchModel()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(model, handle)
        assert handle.detected_accelerator == "cpu"
