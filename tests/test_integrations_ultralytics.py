"""Tests for the Ultralytics/YOLO integration extractor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from wildedge.integrations.ultralytics import (
    YOLO_FAMILY_RE,
    UltralyticsExtractor,
    build_download_record,
    classify_output_meta,
    detect_accelerator,
    detect_quantization,
    detection_output_meta,
    image_input_meta,
    is_yolo,
    weights_file_exists,
)
from wildedge.model import ModelHandle, ModelInfo

# ---------------------------------------------------------------------------
# Fake objects — no ultralytics required
# ---------------------------------------------------------------------------


class FakeParam:
    class Device:
        type = "cpu"

    device = Device()
    dtype = type("dtype", (), {"__str__": lambda self: "torch.float32"})()


class FakeInnerModel:
    def parameters(self):
        yield FakeParam()


class FakeBoxes:
    class _Tensor:
        def __init__(self, data):
            self._data = data

        def tolist(self):
            return self._data

    def __init__(self, preds):
        # preds: list of (x1, y1, x2, y2, conf, cls)
        self.xyxy = self._Tensor([[p[0], p[1], p[2], p[3]] for p in preds])
        self.conf = self._Tensor([p[4] for p in preds])
        self.cls = self._Tensor([p[5] for p in preds])


class FakeResult:
    def __init__(self, preds=None):
        self.boxes = FakeBoxes(preds or [])
        self.probs = None


class FakeClassifyProbs:
    top5 = [2, 0, 1, 3, 4]

    class _Tensor:
        def tolist(self):
            return [0.9, 0.05, 0.02, 0.02, 0.01]

    top5conf = _Tensor()


class FakeClassifyResult:
    boxes = None
    probs = FakeClassifyProbs()


# YOLO must have this exact class name for is_yolo() to match.
class YOLO:
    task = "detect"
    names = {0: "person", 1: "bicycle"}
    ckpt_path = "/models/yolov8n.pt"
    model = FakeInnerModel()

    def __call__(self, source=None, stream=False, **kwargs):
        return [FakeResult([(10, 20, 100, 200, 0.9, 0)])]


class FailingYOLO(YOLO):
    def __call__(self, source=None, stream=False, **kwargs):
        raise RuntimeError("cuda oom")


class ClassifyYOLO(YOLO):
    task = "classify"

    def __call__(self, source=None, stream=False, **kwargs):
        return [FakeClassifyResult()]


def make_handle(publish_spy) -> ModelHandle:
    info = ModelInfo(
        model_name="yolov8n",
        model_version="unknown",
        model_source="local",
        model_format="pytorch",
    )
    return ModelHandle(model_id="yolov8n", info=info, publish=publish_spy)


# ---------------------------------------------------------------------------
# is_yolo
# ---------------------------------------------------------------------------


def test_is_yolo_true_for_yolo_instance():
    assert is_yolo(YOLO()) is True


def test_is_yolo_false_for_plain_object():
    assert is_yolo(object()) is False


def test_is_yolo_false_for_string():
    assert is_yolo("yolo") is False


# ---------------------------------------------------------------------------
# YOLO_FAMILY_RE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("yolov8n", "yolov8"),
        ("yolov8s-seg", "yolov8"),
        ("yolov9e", "yolov9"),
        ("yolov10n", "yolov10"),
        ("yolo11n", "yolo11"),
    ],
)
def test_yolo_family_re_extracts_prefix(name, expected):
    m = YOLO_FAMILY_RE.match(name)
    assert m is not None
    assert m.group(1).lower() == expected


def test_yolo_family_re_no_match_for_unknown_format():
    assert YOLO_FAMILY_RE.match("yolo_nas_s") is None


# ---------------------------------------------------------------------------
# detect_accelerator
# ---------------------------------------------------------------------------


def test_detect_accelerator_returns_cpu_by_default():
    assert detect_accelerator(YOLO()) == "cpu"


def test_detect_accelerator_returns_cuda_when_param_on_cuda():
    model = YOLO()
    cuda_param = MagicMock()
    cuda_param.device.type = "cuda"
    model.model = MagicMock()
    model.model.parameters = MagicMock(return_value=iter([cuda_param]))
    assert detect_accelerator(model) == "cuda"


def test_detect_accelerator_returns_cpu_when_no_inner_model():
    model = YOLO()
    model.model = None
    assert detect_accelerator(model) == "cpu"


# ---------------------------------------------------------------------------
# detect_quantization
# ---------------------------------------------------------------------------


def test_detect_quantization_f32_from_float32_dtype():
    assert detect_quantization(YOLO()) == "f32"


def test_detect_quantization_f16_from_float16_param():
    model = YOLO()
    p = MagicMock()
    p.dtype.__str__ = lambda self: "torch.float16"
    model.model = MagicMock()
    model.model.parameters = MagicMock(return_value=iter([p]))
    assert detect_quantization(model) == "f16"


def test_detect_quantization_none_when_no_inner_model():
    model = YOLO()
    model.model = None
    assert detect_quantization(model) is None


# ---------------------------------------------------------------------------
# image_input_meta
# ---------------------------------------------------------------------------


def test_image_input_meta_returns_none_when_numpy_absent():
    with patch("wildedge.integrations.ultralytics.np", None):
        assert image_input_meta(object()) is None


def test_image_input_meta_extracts_hwc_array():
    pytest.importorskip("numpy")
    import numpy as np

    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    meta = image_input_meta(arr)
    assert meta is not None
    assert meta.width == 640
    assert meta.height == 480
    assert meta.channels == 3


def test_image_input_meta_returns_none_for_non_array():
    assert image_input_meta("not_an_array") is None


# ---------------------------------------------------------------------------
# detection_output_meta
# ---------------------------------------------------------------------------


def test_detection_output_meta_builds_from_results():
    results = [FakeResult([(10, 20, 100, 200, 0.9, 0), (5, 5, 50, 50, 0.6, 1)])]
    names = {0: "person", 1: "bicycle"}
    meta = detection_output_meta(results, names)
    assert meta is not None
    assert meta.num_predictions == 2
    assert meta.top_k is not None
    assert meta.top_k[0].label == "person"
    assert meta.top_k[0].confidence == 0.9
    assert meta.avg_confidence == pytest.approx(0.75, abs=0.01)
    assert meta.num_classes == 2


def test_detection_output_meta_empty_results_gives_zero_preds():
    meta = detection_output_meta([FakeResult([])], {})
    assert meta is not None
    assert meta.num_predictions == 0
    assert meta.top_k is None


def test_detection_output_meta_top_k_capped_at_five():
    preds = [(i, i, i + 10, i + 10, 0.5, i) for i in range(8)]
    names = {i: str(i) for i in range(8)}
    meta = detection_output_meta([FakeResult(preds)], names)
    assert meta is not None
    assert len(meta.top_k) == 5


# ---------------------------------------------------------------------------
# classify_output_meta
# ---------------------------------------------------------------------------


def test_classify_output_meta_builds_from_results():
    names = {0: "cat", 1: "dog", 2: "bird", 3: "fish", 4: "hamster"}
    meta = classify_output_meta([FakeClassifyResult()], names)
    assert meta is not None
    assert meta.num_predictions == 5
    assert meta.top_k is not None
    assert meta.top_k[0].label == "bird"  # top5[0] = 2 → "bird"
    assert meta.avg_confidence == 0.9


def test_classify_output_meta_missing_probs_gives_empty():
    result = MagicMock()
    result.probs = None
    meta = classify_output_meta([result], {})
    assert meta is not None
    assert meta.num_predictions == 0


# ---------------------------------------------------------------------------
# weights_file_exists
# ---------------------------------------------------------------------------


def test_weights_file_exists_true_when_file_on_disk(tmp_path):
    f = tmp_path / "yolov8n.pt"
    f.write_bytes(b"fake")
    assert weights_file_exists(str(f)) is True


def test_weights_file_exists_false_when_file_missing(tmp_path):
    with patch("wildedge.integrations.ultralytics._ULTRALYTICS_WEIGHTS_DIR", None):
        assert weights_file_exists(str(tmp_path / "missing.pt")) is False


def test_weights_file_exists_true_for_non_string_arg():
    assert weights_file_exists(42) is True


def test_weights_file_exists_checks_ultralytics_weights_dir(tmp_path):
    f = tmp_path / "yolov8n.pt"
    f.write_bytes(b"fake")
    with patch("wildedge.integrations.ultralytics._ULTRALYTICS_WEIGHTS_DIR", tmp_path):
        assert weights_file_exists("yolov8n.pt") is True


# ---------------------------------------------------------------------------
# build_download_record
# ---------------------------------------------------------------------------


def test_build_download_record_builds_for_existing_file(tmp_path):
    f = tmp_path / "yolov8n.pt"
    f.write_bytes(b"x" * 1000)
    model = MagicMock()
    model.ckpt_path = str(f)
    rec = build_download_record(model, load_ms=500)
    assert rec is not None
    assert rec["repo_id"] == "yolov8n"
    assert rec["size"] == 1000
    assert rec["cache_hit"] is False
    assert rec["source_type"] == "ultralytics"
    assert "yolov8n.pt" in rec["source_url"]


def test_build_download_record_returns_none_when_no_ckpt_path():
    model = MagicMock()
    model.ckpt_path = None
    assert build_download_record(model, load_ms=100) is None


def test_build_download_record_returns_none_when_file_missing(tmp_path):
    model = MagicMock()
    model.ckpt_path = str(tmp_path / "missing.pt")
    assert build_download_record(model, load_ms=100) is None


# ---------------------------------------------------------------------------
# UltralyticsExtractor
# ---------------------------------------------------------------------------

_extractor = UltralyticsExtractor()


def test_extractor_can_handle_yolo_instance():
    assert _extractor.can_handle(YOLO()) is True


def test_extractor_can_handle_rejects_other_types():
    assert _extractor.can_handle(object()) is False


def test_extractor_extract_info_uses_ckpt_stem_as_model_id():
    model_id, info = _extractor.extract_info(YOLO(), {})
    assert model_id == "yolov8n"
    assert info.model_name == "yolov8n"
    assert info.model_format == "pytorch"


def test_extractor_extract_info_derives_family():
    _, info = _extractor.extract_info(YOLO(), {})
    assert info.model_family == "yolov8"


def test_extractor_extract_info_override_model_id():
    model_id, _ = _extractor.extract_info(YOLO(), {"id": "my-detector"})
    assert model_id == "my-detector"


def test_extractor_extract_info_fallback_model_name_when_no_ckpt():
    model = YOLO()
    model.ckpt_path = ""
    model_id, info = _extractor.extract_info(model, {})
    assert model_id == "yolo"
    assert info.model_name == "yolo"


def test_extractor_memory_bytes_returns_file_size(tmp_path):
    f = tmp_path / "yolov8n.pt"
    f.write_bytes(b"x" * 2048)
    model = YOLO()
    model.ckpt_path = str(f)
    assert _extractor.memory_bytes(model) == 2048


def test_extractor_memory_bytes_returns_none_when_file_missing():
    model = YOLO()
    model.ckpt_path = "/nonexistent/path.pt"
    assert _extractor.memory_bytes(model) is None


def test_extractor_install_hooks_publishes_inference_on_call(publish_spy):
    model = YOLO()
    handle = make_handle(publish_spy)
    _extractor.install_hooks(model, handle)
    model()
    assert len(publish_spy.events) == 1
    assert publish_spy.events[0]["event_type"] == "inference"


def test_extractor_install_hooks_detect_output_meta(publish_spy):
    model = YOLO()
    handle = make_handle(publish_spy)
    _extractor.install_hooks(model, handle)
    model()
    event = publish_spy.events[0]
    assert event["inference"]["output_meta"]["task"] == "detection"
    assert event["inference"]["output_meta"]["num_predictions"] == 1
    assert event["inference"]["input_modality"] == "image"
    assert event["inference"]["output_modality"] == "detection"


def test_extractor_install_hooks_classify_output_meta(publish_spy):
    model = ClassifyYOLO()
    handle = make_handle(publish_spy)
    _extractor.install_hooks(model, handle)
    model()
    assert publish_spy.events[0]["inference"]["output_modality"] == "classification"


def test_extractor_install_hooks_tracks_error_on_exception(publish_spy):
    model = FailingYOLO()
    handle = make_handle(publish_spy)
    _extractor.install_hooks(model, handle)
    with pytest.raises(RuntimeError, match="cuda oom"):
        model()
    assert publish_spy.events[0]["event_type"] == "error"


def test_extractor_install_hooks_does_not_affect_other_instances(publish_spy):
    model_a = YOLO()
    model_b = YOLO()
    handle = make_handle(publish_spy)
    _extractor.install_hooks(model_a, handle)
    model_b()
    assert len(publish_spy.events) == 0


def test_extractor_install_hooks_batch_size_from_list_source(publish_spy):
    np = pytest.importorskip("numpy")
    model = YOLO()
    handle = make_handle(publish_spy)
    _extractor.install_hooks(model, handle)
    frames = [np.zeros((480, 640, 3), dtype=np.uint8)] * 3
    model(frames)
    assert publish_spy.events[0]["inference"]["batch_size"] == 3
