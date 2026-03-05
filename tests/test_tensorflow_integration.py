from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

from wildedge.integrations.tensorflow import TensorflowExtractor
from wildedge.model import ModelHandle, ModelInfo


def make_handle(publish_spy) -> ModelHandle:
    info = ModelInfo(
        model_name="tf-model",
        model_version="1.0",
        model_source="local",
        model_format="tensorflow",
    )
    return ModelHandle(model_id="tf-model", info=info, publish=publish_spy)


class _KerasBase:
    pass


_KerasBase.__name__ = "Model"
_KerasBase.__module__ = "keras.src.models.model"


class _FakeTfModel(_KerasBase):
    name = "fake-tf-model"
    weights = []

    def __call__(self, *args, **kwargs):
        return [1.0]

    def predict(self, *args, **kwargs):
        return [1.0]


class TestTensorflowExtractor:
    extractor = TensorflowExtractor()

    def test_can_handle_tensorflow_model(self):
        assert self.extractor.can_handle(_FakeTfModel()) is True

    def test_can_handle_rejects_plain_object(self):
        assert self.extractor.can_handle(object()) is False

    def test_install_hooks_tracks_call(self, publish_spy):
        model = _FakeTfModel()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(model, handle)
        model()
        assert publish_spy.events[0]["event_type"] == "inference"

    def test_install_hooks_tracks_predict(self, publish_spy):
        model = _FakeTfModel()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(model, handle)
        model.predict([1.0])
        assert publish_spy.events[0]["event_type"] == "inference"

    def test_install_auto_load_patch_is_idempotent(self, monkeypatch):
        import wildedge.integrations.tensorflow as tf_mod

        def load_model(*args, **kwargs):
            return object()

        def saved_model_load(*args, **kwargs):
            return object()

        fake_tf = types.SimpleNamespace(
            keras=types.SimpleNamespace(
                models=types.SimpleNamespace(load_model=load_model)
            ),
            saved_model=types.SimpleNamespace(load=saved_model_load),
        )
        monkeypatch.setattr(tf_mod, "_tf", fake_tf)
        monkeypatch.setattr(tf_mod, "_tf_auto_patched", False)

        def client_ref():
            return None

        TensorflowExtractor.install_auto_load_patch(client_ref)
        first_load_model = fake_tf.keras.models.load_model
        first_saved_model = fake_tf.saved_model.load
        TensorflowExtractor.install_auto_load_patch(client_ref)
        second_load_model = fake_tf.keras.models.load_model
        second_saved_model = fake_tf.saved_model.load

        assert first_load_model is second_load_model
        assert first_saved_model is second_saved_model


def test_client_instrument_tensorflow_calls_auto_patch(client_with_stubbed_runtime):
    client = client_with_stubbed_runtime
    patcher = MagicMock()
    with patch.dict(client.PATCH_INSTALLERS, {"tensorflow": patcher}):
        client.instrument("tensorflow")
    patcher.assert_called_once()
