from __future__ import annotations

import pytest
import ultralytics
from PIL import Image


@pytest.fixture
def yolo(compat_client):
    compat_client.instrument("ultralytics")
    # Built from the architecture config, so no weights are downloaded.
    return ultralytics.YOLO("yolov8n.yaml")


@pytest.fixture
def image(tmp_path) -> str:
    # A file path is a source every supported ultralytics release accepts.
    path = tmp_path / "blank.png"
    Image.new("RGB", (64, 64)).save(path)
    return str(path)


def test_yolo_load_is_tracked(compat_client, yolo):
    compat_client.one("model_load")
    assert compat_client.of_type("model_download") == []


def test_yolo_call_is_tracked(compat_client, yolo, image):
    results = yolo(image, imgsz=64, verbose=False)
    assert len(results) == 1

    inference = compat_client.one("inference")
    assert inference["model_id"] in compat_client.registry.handles
    assert inference["inference"]["input_modality"] == "image"
    assert inference["inference"]["success"] is True


def test_yolo_predict_is_tracked(compat_client, yolo, image):
    yolo.predict(image, imgsz=64, verbose=False)
    compat_client.one("inference")


def test_predict_and_call_each_track_once(compat_client, yolo, image):
    yolo.predict(image, imgsz=64, verbose=False)
    assert len(compat_client.of_type("inference")) == 1
    yolo(image, imgsz=64, verbose=False)
    assert len(compat_client.of_type("inference")) == 2
