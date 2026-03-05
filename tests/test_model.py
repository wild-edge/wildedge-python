"""Tests for ModelRegistry and ModelHandle."""

from wildedge.model import ModelHandle, ModelInfo, ModelRegistry


def make_info(**kwargs) -> ModelInfo:
    defaults = dict(
        model_name="test",
        model_version="1.0",
        model_source="local",
        model_format="onnx",
    )
    return ModelInfo(**{**defaults, **kwargs})


def noop_publish(event: dict) -> None:
    pass


class TestModelRegistry:
    def test_register_returns_handle(self):
        registry = ModelRegistry()
        handle, is_new = registry.register("model-1", make_info(), noop_publish)
        assert isinstance(handle, ModelHandle)
        assert handle.model_id == "model-1"
        assert is_new is True

    def test_register_same_id_returns_existing_handle(self):
        registry = ModelRegistry()
        handle1, _ = registry.register("model-1", make_info(), noop_publish)
        handle2, is_new = registry.register("model-1", make_info(), noop_publish)
        assert handle1 is handle2
        assert is_new is False

    def test_register_same_id_does_not_overwrite_info(self):
        registry = ModelRegistry()
        registry.register("model-1", make_info(model_name="original"), noop_publish)
        registry.register("model-1", make_info(model_name="replacement"), noop_publish)
        assert registry.models["model-1"].model_name == "original"

    def test_register_different_ids_are_independent(self):
        registry = ModelRegistry()
        h1, _ = registry.register("model-1", make_info(), noop_publish)
        h2, _ = registry.register("model-2", make_info(), noop_publish)
        assert h1 is not h2
        assert len(registry.models) == 2

    def test_snapshot_reflects_registered_models(self):
        registry = ModelRegistry()
        registry.register("model-1", make_info(model_name="yolo"), noop_publish)
        snap = registry.snapshot()
        assert "model-1" in snap
        assert snap["model-1"]["model_name"] == "yolo"

    def test_snapshot_excludes_duplicate_registration(self):
        registry = ModelRegistry()
        registry.register("model-1", make_info(model_name="original"), noop_publish)
        registry.register("model-1", make_info(model_name="replacement"), noop_publish)
        snap = registry.snapshot()
        assert len(snap) == 1
        assert snap["model-1"]["model_name"] == "original"
