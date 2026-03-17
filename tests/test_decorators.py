"""Tests for the @wildedge.track decorator and context manager."""

import time

import pytest

from wildedge.decorators import track
from wildedge.model import ModelHandle, ModelInfo


def make_handle(publish_spy):
    info = ModelInfo(
        model_name="test",
        model_version="1.0",
        model_source="local",
        model_format="onnx",
    )
    return ModelHandle(model_id="test-model", info=info, publish=publish_spy)


class TestTrackDecorator:
    def test_decorator_tracks_successful_inference(self, publish_spy):
        handle = make_handle(publish_spy)

        @track(handle, input_type="image", output_type="classification")
        def my_inference(x):
            return x * 2

        result = my_inference(5)
        assert result == 10
        assert len(publish_spy.events) == 1
        event = publish_spy.events[0]
        assert event["event_type"] == "inference"
        assert event["inference"]["input_modality"] == "image"
        assert event["inference"]["output_modality"] == "classification"
        assert event["inference"]["success"] is True

    def test_decorator_tracks_error_and_reraises(self, publish_spy):
        handle = make_handle(publish_spy)

        @track(handle, capture_errors=True)
        def failing_func():
            raise RuntimeError("inference crashed")

        with pytest.raises(RuntimeError, match="inference crashed"):
            failing_func()

        assert len(publish_spy.events) == 1
        event = publish_spy.events[0]
        assert event["event_type"] == "error"

    def test_decorator_without_error_capture_does_not_track_error(self, publish_spy):
        handle = make_handle(publish_spy)

        @track(handle, capture_errors=False)
        def failing_func():
            raise ValueError("oops")

        with pytest.raises(ValueError):
            failing_func()

        assert len(publish_spy.events) == 0

    def test_decorator_preserves_function_name(self, publish_spy):
        handle = make_handle(publish_spy)

        @track(handle)
        def my_named_function():
            pass

        assert my_named_function.__name__ == "my_named_function"

    def test_decorator_passes_args(self, publish_spy):
        handle = make_handle(publish_spy)

        @track(handle)
        def add(a, b):
            return a + b

        assert add(3, 4) == 7


class TestTrackContextManager:
    def test_context_manager_tracks_successful_inference(self, publish_spy):
        handle = make_handle(publish_spy)

        with track(handle, input_type="text", output_type="text"):
            pass

        assert len(publish_spy.events) == 1
        event = publish_spy.events[0]
        assert event["event_type"] == "inference"
        assert event["inference"]["success"] is True

    def test_context_manager_tracks_error_and_does_not_suppress(self, publish_spy):
        handle = make_handle(publish_spy)

        with pytest.raises(ValueError, match="boom"):
            with track(handle, capture_errors=True):
                raise ValueError("boom")

        assert len(publish_spy.events) == 1
        event = publish_spy.events[0]
        assert event["event_type"] == "error"
        assert "boom" in event["error"]["error_message"]

    def test_context_manager_measures_duration(self, publish_spy):
        handle = make_handle(publish_spy)

        with track(handle):
            time.sleep(0.01)

        event = publish_spy.events[0]
        assert event["inference"]["duration_ms"] > 0

    def test_context_manager_default_modality(self, publish_spy):
        handle = make_handle(publish_spy)

        with track(handle):
            pass

        event = publish_spy.events[0]
        assert event["inference"]["input_modality"] == "structured"
        assert event["inference"]["output_modality"] == "structured"
