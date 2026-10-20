from __future__ import annotations

import importlib.util

import pytest
import ultralytics
from PIL import Image

# Early 8.0.x releases keep their helpers in ultralytics.yolo.utils. The
# integration imports ultralytics.utils, so on those releases it is off.
LEGACY_LAYOUT = importlib.util.find_spec("ultralytics.utils") is None
legacy_xfail = pytest.mark.xfail(
    LEGACY_LAYOUT,
    strict=True,
    reason="integration is disabled on ultralytics without ultralytics.utils (#71)",
)


@pytest.fixture
def yolo(compat_client):
    compat_client.instrument("ultralytics")
    # Built from the architecture config, so no weights are downloaded.
    return ultralytics.YOLO("yolov8n.yaml")


@pytest.fixture
def image(tmp_path) -> str:
    # A file path: ultralytics 8.0.0 can't take a numpy array as the source.
    path = tmp_path / "blank.png"
    Image.new("RGB", (64, 64)).save(path)
    return str(path)


@legacy_xfail
def test_yolo_load_is_tracked(compat_client, yolo):
    compat_client.one("model_load")
    assert compat_client.of_type("model_download") == []


@legacy_xfail
def test_yolo_call_is_tracked(compat_client, yolo, image):
    results = yolo(image, imgsz=64, verbose=False)
    assert len(results) == 1

    inference = compat_client.one("inference")
    assert inference["model_id"] in compat_client.registry.handles
    assert inference["inference"]["input_modality"] == "image"
    assert inference["inference"]["success"] is True


@pytest.mark.xfail(
    strict=True,
    reason="model.predict() is not tracked, only model() (#72)",
)
def test_yolo_predict_is_tracked(compat_client, yolo, image):
    yolo.predict(image, imgsz=64, verbose=False)
    compat_client.one("inference")
