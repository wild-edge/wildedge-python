"""TensorFlow integration."""

from __future__ import annotations

import functools
import threading
import time
from typing import TYPE_CHECKING

from wildedge import constants
from wildedge.device import CURRENT_PLATFORM
from wildedge.integrations.base import BaseExtractor, patch_instance_call_once
from wildedge.integrations.common import (
    debug_failure,
    infer_input_modality_from_layer_types,
    num_classes_from_output_shape,
)
from wildedge.model import ModelInfo
from wildedge.timing import elapsed_ms

if TYPE_CHECKING:
    from wildedge.model import ModelHandle

try:
    import tensorflow as _tf  # type: ignore[import-untyped]
except ImportError:
    _tf = None  # type: ignore[assignment]

TENSORFLOW_CALL_PATCH_NAME = "tensorflow_call"
TENSORFLOW_HANDLE_ATTR = "__wildedge_tensorflow_handle__"
TENSORFLOW_PREDICT_PATCHED_ATTR = "__wildedge_tensorflow_predict_patched__"
_TF_AUTO_PATCH_LOCK = threading.Lock()
_tf_auto_patched = False
TF_AUTO_LOAD_PATCH_NAME = "tensorflow_auto_load"


debug_tensorflow_failure = functools.partial(debug_failure, "tensorflow")


def is_tensorflow_model(obj: object) -> bool:
    return any(
        c.__name__ == "Model"
        and ("keras" in c.__module__ or c.__module__.startswith("tensorflow"))
        for c in type(obj).__mro__
    )


def detect_accelerator(obj: object) -> str:
    try:
        weights = getattr(obj, "weights", None) or []
        if weights:
            device = str(getattr(weights[0], "device", "") or "")
            if any(g in device.upper() for g in ("GPU", "CUDA")):
                return CURRENT_PLATFORM.gpu_accelerator_for_offload()
    except Exception as exc:
        debug_tensorflow_failure("accelerator detection", exc)
    return "cpu"


def build_patched_call(
    original_call,
    *,
    num_classes: int = 0,
    static_input_modality: str | None = None,
):
    input_modality = static_input_modality or "tensor"
    output_modality = "classification" if num_classes > 0 else "tensor"

    def patched_call(self_inner, *args, **kwargs):
        handle = getattr(self_inner, TENSORFLOW_HANDLE_ATTR, None)
        if handle is None:
            return original_call(self_inner, *args, **kwargs)

        t0 = time.perf_counter()
        try:
            result = original_call(self_inner, *args, **kwargs)
            handle.track_inference(
                duration_ms=elapsed_ms(t0),
                input_modality=input_modality,
                output_modality=output_modality,
                success=True,
            )
            return result
        except Exception as exc:
            handle.track_error(
                error_code="UNKNOWN",
                error_message=str(exc)[: constants.ERROR_MSG_MAX_LEN],
            )
            raise

    return patched_call


def patch_predict_once(
    obj: object,
    *,
    num_classes: int = 0,
    static_input_modality: str | None = None,
) -> None:
    if getattr(obj, TENSORFLOW_PREDICT_PATCHED_ATTR, False):
        return

    predict = getattr(obj, "predict", None)
    if not callable(predict):
        return

    input_modality = static_input_modality or "tensor"
    output_modality = "classification" if num_classes > 0 else "tensor"

    def patched_predict(*args, **kwargs):
        handle = getattr(obj, TENSORFLOW_HANDLE_ATTR, None)
        if handle is None:
            return predict(*args, **kwargs)

        t0 = time.perf_counter()
        try:
            result = predict(*args, **kwargs)
            handle.track_inference(
                duration_ms=elapsed_ms(t0),
                input_modality=input_modality,
                output_modality=output_modality,
                success=True,
            )
            return result
        except Exception as exc:
            handle.track_error(
                error_code="UNKNOWN",
                error_message=str(exc)[: constants.ERROR_MSG_MAX_LEN],
            )
            raise

    setattr(obj, "predict", patched_predict)
    setattr(obj, TENSORFLOW_PREDICT_PATCHED_ATTR, True)


def build_load_patch(client_ref: object, original_load):
    def patched_load(*args, **kwargs):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        model = original_load(*args, **kwargs)
        load_ms = elapsed_ms(t0)
        client = client_ref()  # type: ignore[call-arg]
        if client is not None and not client.closed:
            client._on_model_auto_loaded(model, load_ms=load_ms)
        return model

    patched_load.__wildedge_patch_name__ = TF_AUTO_LOAD_PATCH_NAME  # type: ignore[attr-defined]
    patched_load.__wildedge_original_call__ = original_load  # type: ignore[attr-defined]
    return patched_load


class TensorflowExtractor(BaseExtractor):
    def can_handle(self, obj: object) -> bool:
        return is_tensorflow_model(obj)

    def extract_info(
        self, obj: object, overrides: dict
    ) -> tuple[str | None, ModelInfo]:
        model_name = getattr(obj, "name", None) or type(obj).__name__
        model_id = overrides.pop("id", None) or model_name
        version = overrides.pop("version", "unknown")
        family = overrides.pop("family", None)
        source = overrides.pop("source", "local")
        quantization = overrides.pop("quantization", None)

        info = ModelInfo(
            model_name=model_name,
            model_version=version,
            model_source=source,
            model_format="tensorflow",
            model_family=family,
            quantization=quantization,
        )
        for key, value in overrides.items():
            if hasattr(info, key):
                setattr(info, key, value)
        return model_id, info

    def install_hooks(self, obj: object, handle: ModelHandle) -> None:
        handle.detected_accelerator = detect_accelerator(obj)

        static_input_modality: str | None = None
        static_num_classes: int = 0
        try:
            layer_types = [type(layer).__name__ for layer in obj.layers]  # type: ignore[union-attr]
            static_input_modality = infer_input_modality_from_layer_types(layer_types)
        except Exception as exc:
            debug_tensorflow_failure("layer type scan", exc)
        try:
            out_shape = obj.output_shape  # type: ignore[union-attr]
            if isinstance(out_shape, tuple):
                static_num_classes = num_classes_from_output_shape(out_shape)
        except Exception as exc:
            debug_tensorflow_failure("output shape inspection", exc)

        setattr(obj, TENSORFLOW_HANDLE_ATTR, handle)
        patch_instance_call_once(
            obj,
            patch_name=TENSORFLOW_CALL_PATCH_NAME,
            make_patched_call=functools.partial(
                build_patched_call,
                num_classes=static_num_classes,
                static_input_modality=static_input_modality,
            ),
        )
        patch_predict_once(
            obj,
            num_classes=static_num_classes,
            static_input_modality=static_input_modality,
        )

    @classmethod
    def install_auto_load_patch(cls, client_ref: object) -> None:
        global _tf_auto_patched
        if _tf_auto_patched or _tf is None:
            return

        with _TF_AUTO_PATCH_LOCK:
            if _tf_auto_patched:
                return

            try:
                keras_models = _tf.keras.models
                saved_model = _tf.saved_model
            except Exception as exc:
                debug_tensorflow_failure("auto-load patch setup", exc)
                return

            load_model = getattr(keras_models, "load_model", None)
            if (
                callable(load_model)
                and getattr(load_model, "__wildedge_patch_name__", None)
                != TF_AUTO_LOAD_PATCH_NAME
            ):
                keras_models.load_model = build_load_patch(client_ref, load_model)

            saved_model_load = getattr(saved_model, "load", None)
            if (
                callable(saved_model_load)
                and getattr(saved_model_load, "__wildedge_patch_name__", None)
                != TF_AUTO_LOAD_PATCH_NAME
            ):
                saved_model.load = build_load_patch(client_ref, saved_model_load)

            _tf_auto_patched = True
