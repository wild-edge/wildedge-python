"""Tests for ONNX, GGUF, and Keras integration extractors."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from wildedge.integrations.gguf import (
    GgufExtractor,
    _parse_quantization,
)
from wildedge.integrations.gguf import (
    _detect_accelerator as gguf_detect_accelerator,
)
from wildedge.integrations.keras import KerasExtractor
from wildedge.integrations.keras import _detect_accelerator as keras_detect_accelerator
from wildedge.integrations.onnx import (
    OnnxExtractor,
)
from wildedge.integrations.onnx import (
    _detect_accelerator as onnx_detect_accelerator,
)
from wildedge.integrations.onnx import (
    _detect_quantization as onnx_detect_quantization,
)
from wildedge.model import ModelHandle, ModelInfo

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_handle(publish_spy) -> ModelHandle:
    info = ModelInfo(
        model_name="test",
        model_version="1.0",
        model_source="local",
        model_format="test",
    )
    return ModelHandle(model_id="m", info=info, publish=publish_spy)


# ---------------------------------------------------------------------------
# Fake objects — no ML libraries required
# ---------------------------------------------------------------------------

# ONNX


class _FakeMeta:
    graph_name = "resnet50"
    version = 1


class _FakeInput:
    type = "tensor(float32)"


class _FakeOrtSession:
    def get_modelmeta(self):
        return _FakeMeta()

    def get_inputs(self):
        return [_FakeInput()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, input_feed, run_options=None):
        return [[1.0]]


# GGUF


class Llama:  # must match the name check in GgufExtractor.can_handle
    model_path = "/models/llama-3-Q4_K_M.gguf"
    n_gpu_layers = 0
    metadata = {"general.architecture": "llama", "general.version": "3.0"}

    def __call__(self, *args, **kwargs):
        return {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}


class _FailingLlama(Llama):
    def __call__(self, *args, **kwargs):
        raise RuntimeError("cuda oom")


# Keras


class _KerasBase:
    """Looks like keras.Model to the MRO check."""


_KerasBase.__name__ = "Model"
_KerasBase.__module__ = "keras.src.trainers.trainer"


class _FakeKerasModel(_KerasBase):
    name = "my_seq"
    weights = []

    def __call__(self, *args, **kwargs):
        return [1.0, 2.0]


class _FailingKerasModel(_KerasBase):
    name = "failing"
    weights = []

    def __call__(self, *args, **kwargs):
        raise RuntimeError("gpu oom")


# ---------------------------------------------------------------------------
# ONNX
# ---------------------------------------------------------------------------


class TestOnnxExtractor:
    extractor = OnnxExtractor()

    def test_can_handle_true_when_ort_available(self):
        with patch("wildedge.integrations.onnx.ort") as mock_ort:
            mock_ort.InferenceSession = _FakeOrtSession
            assert self.extractor.can_handle(_FakeOrtSession()) is True

    def test_can_handle_false_when_ort_not_installed(self):
        with patch("wildedge.integrations.onnx.ort", None):
            assert self.extractor.can_handle(_FakeOrtSession()) is False

    def test_can_handle_false_for_wrong_type(self):
        with patch("wildedge.integrations.onnx.ort") as mock_ort:
            mock_ort.InferenceSession = _FakeOrtSession
            assert self.extractor.can_handle(object()) is False

    def test_detect_accelerator_cuda_provider(self):
        session = MagicMock()
        session.get_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        assert onnx_detect_accelerator(session) == "cuda"

    def test_detect_accelerator_defaults_to_cpu(self):
        session = MagicMock()
        session.get_providers.return_value = ["CPUExecutionProvider"]
        assert onnx_detect_accelerator(session) == "cpu"

    def test_detect_quantization_int8(self):
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(type="tensor(int8)")]
        assert onnx_detect_quantization(session) == "int8"

    def test_detect_quantization_f32(self):
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(type="tensor(float32)")]
        assert onnx_detect_quantization(session) == "f32"

    def test_extract_info_uses_graph_name_as_model_id(self):
        model_id, info = self.extractor.extract_info(_FakeOrtSession(), {})
        assert model_id == "resnet50"
        assert info.model_format == "onnx"

    def test_extract_info_none_model_id_when_graph_name_empty(self):
        session = _FakeOrtSession()
        session.get_modelmeta = lambda: MagicMock(graph_name="", version=0)
        model_id, _ = self.extractor.extract_info(session, {})
        assert model_id is None

    def test_install_hooks_publishes_inference_on_run(self, publish_spy):
        session = _FakeOrtSession()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(session, handle)
        session.run(None, {})
        assert len(publish_spy.events) == 1
        assert publish_spy.events[0]["event_type"] == "inference"

    def test_install_hooks_tracks_error_on_exception(self, publish_spy):
        def _fail(*a, **kw):
            raise RuntimeError("inference failed")

        session = _FakeOrtSession()
        session.run = _fail
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(session, handle)
        with pytest.raises(RuntimeError):
            session.run(None, {})
        assert publish_spy.events[0]["event_type"] == "error"


# ---------------------------------------------------------------------------
# GGUF
# ---------------------------------------------------------------------------


class TestGgufExtractor:
    extractor = GgufExtractor()

    def test_can_handle_matches_llama_class(self):
        assert self.extractor.can_handle(Llama()) is True

    def test_can_handle_rejects_other_types(self):
        assert self.extractor.can_handle(object()) is False

    def test_parse_quantization_q4_k_m(self):
        assert _parse_quantization("llama-3-Q4_K_M.gguf") == "q4_k_m"

    def test_parse_quantization_f16(self):
        assert _parse_quantization("model-F16.gguf") == "f16"

    def test_parse_quantization_none_for_unknown(self):
        assert _parse_quantization("model.gguf") is None

    def test_detect_accelerator_cpu_when_no_gpu_layers(self):
        llm = Llama()
        llm.n_gpu_layers = 0
        assert gguf_detect_accelerator(llm) == "cpu"

    def test_detect_accelerator_non_cpu_when_gpu_layers_offloaded(self):
        llm = Llama()
        llm.n_gpu_layers = 32
        with patch(
            "wildedge.integrations.gguf.CURRENT_PLATFORM.gpu_accelerator_for_offload",
            return_value="cuda",
        ):
            assert gguf_detect_accelerator(llm) == "cuda"

    def test_extract_info_uses_filename_stem_as_model_id(self):
        model_id, info = self.extractor.extract_info(Llama(), {})
        assert model_id == "llama-3-Q4_K_M"
        assert info.model_format == "gguf"
        assert info.quantization == "q4_k_m"

    def test_extract_info_reads_family_from_metadata(self):
        _, info = self.extractor.extract_info(Llama(), {})
        assert info.model_family == "llama"

    def test_install_hooks_publishes_inference_on_call(self, publish_spy):
        llm = Llama()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(llm, handle)
        llm()
        assert len(publish_spy.events) == 1
        assert publish_spy.events[0]["event_type"] == "inference"

    def test_install_hooks_includes_token_counts(self, publish_spy):
        llm = Llama()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(llm, handle)
        llm()
        event = publish_spy.events[0]
        assert event["inference"]["output_meta"]["tokens_in"] == 10
        assert event["inference"]["output_meta"]["tokens_out"] == 20

    def test_install_hooks_tracks_error_on_exception(self, publish_spy):
        llm = _FailingLlama()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(llm, handle)
        with pytest.raises(RuntimeError):
            llm()
        assert publish_spy.events[0]["event_type"] == "error"


# ---------------------------------------------------------------------------
# Keras
# ---------------------------------------------------------------------------


class TestKerasExtractor:
    extractor = KerasExtractor()

    def test_can_handle_keras_model(self):
        assert self.extractor.can_handle(_FakeKerasModel()) is True

    def test_can_handle_rejects_plain_object(self):
        assert self.extractor.can_handle(object()) is False

    def test_extract_info_uses_model_name_as_model_id(self):
        model_id, info = self.extractor.extract_info(_FakeKerasModel(), {})
        assert model_id == "my_seq"
        assert info.model_format == "keras"

    def test_extract_info_override_model_id(self):
        model_id, _ = self.extractor.extract_info(_FakeKerasModel(), {"id": "my-bert"})
        assert model_id == "my-bert"

    def test_detect_accelerator_gpu_from_weight_device(self):
        model = _FakeKerasModel()
        model.weights = [MagicMock(device="/GPU:0")]
        with patch(
            "wildedge.integrations.keras.CURRENT_PLATFORM.gpu_accelerator_for_offload",
            return_value="cuda",
        ):
            assert keras_detect_accelerator(model) == "cuda"

    def test_install_hooks_publishes_inference_on_call(self, publish_spy):
        model = _FakeKerasModel()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(model, handle)
        model()
        assert len(publish_spy.events) == 1
        assert publish_spy.events[0]["event_type"] == "inference"

    def test_install_hooks_tracks_error_and_reraises(self, publish_spy):
        model = _FailingKerasModel()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(model, handle)
        with pytest.raises(RuntimeError, match="gpu oom"):
            model()
        assert publish_spy.events[0]["event_type"] == "error"

    def test_install_hooks_does_not_affect_other_instances(self, publish_spy):
        model_a = _FakeKerasModel()
        model_b = _FakeKerasModel()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(model_a, handle)
        model_b()  # should NOT publish
        assert len(publish_spy.events) == 0
