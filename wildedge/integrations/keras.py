"""Keras integration."""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING

try:
    import numpy as _np
except ImportError:
    _np = None  # type: ignore[assignment]

from wildedge import constants
from wildedge.device import CURRENT_PLATFORM
from wildedge.events.inference import ClassificationOutputMeta, TopKPrediction
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

KERAS_CALL_PATCH_NAME = "keras_call"
KERAS_HANDLE_ATTR = "__wildedge_keras_handle__"


def is_keras_model(obj: object) -> bool:
    return any(
        c.__name__ == "Model" and "keras" in c.__module__ for c in type(obj).__mro__
    )


debug_keras_failure = functools.partial(debug_failure, "keras")


def classification_output_meta(
    result: object, num_classes: int
) -> ClassificationOutputMeta | None:
    """Build ClassificationOutputMeta from a Keras output tensor. Returns None on any error."""
    if _np is None:
        return None
    try:
        arr = _np.array(result)
        if arr.ndim != 2 or arr.shape[1] != num_classes:
            return None
        exp = _np.exp(arr - arr.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        avg_probs = probs.mean(axis=0)
        top_idx = avg_probs.argsort()[::-1][: min(5, num_classes)]
        return ClassificationOutputMeta(
            num_predictions=num_classes,
            avg_confidence=round(float(probs.max(axis=-1).mean()), 4),
            top_k=[
                TopKPrediction(label=str(i), confidence=round(float(avg_probs[i]), 4))
                for i in top_idx
            ],
        )
    except Exception as exc:
        debug_keras_failure("classification output metadata", exc)
        return None


def build_patched_call(
    original_call,
    *,
    num_classes: int = 0,
    static_input_modality: str | None = None,
):
    input_modality = static_input_modality or "tensor"
    output_modality = "classification" if num_classes > 0 else "tensor"

    def patched_call(self_inner, *args, **kwargs):
        handle = getattr(self_inner, KERAS_HANDLE_ATTR, None)
        if handle is None:
            return original_call(self_inner, *args, **kwargs)

        t0 = time.perf_counter()
        try:
            result = original_call(self_inner, *args, **kwargs)
            out_meta = (
                classification_output_meta(result, num_classes)
                if num_classes > 0
                else None
            )
            handle.track_inference(
                duration_ms=elapsed_ms(t0),
                input_modality=input_modality,
                output_modality=output_modality,
                output_meta=out_meta,
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


def detect_accelerator(obj: object) -> str:
    try:
        weights = getattr(obj, "weights", None) or []
        if weights:
            device = str(getattr(weights[0], "device", "") or "")
            if any(g in device.upper() for g in ("GPU", "CUDA")):
                return CURRENT_PLATFORM.gpu_accelerator_for_offload()
    except Exception as exc:
        debug_keras_failure("accelerator detection", exc)
    return "cpu"


class KerasExtractor(BaseExtractor):
    def can_handle(self, obj: object) -> bool:
        return is_keras_model(obj)

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
            model_format="keras",
            model_family=family,
            quantization=quantization,
        )
        for k, v in overrides.items():
            if hasattr(info, k):
                setattr(info, k, v)

        return model_id, info

    def install_hooks(self, obj: object, handle: ModelHandle) -> None:
        handle.detected_accelerator = detect_accelerator(obj)

        static_input_modality: str | None = None
        static_num_classes: int = 0
        try:
            layer_types = [type(layer).__name__ for layer in obj.layers]  # type: ignore[union-attr]
            static_input_modality = infer_input_modality_from_layer_types(layer_types)
        except Exception as exc:
            debug_keras_failure("layer type scan", exc)
        try:
            out_shape = obj.output_shape  # type: ignore[union-attr]
            if isinstance(out_shape, tuple):
                static_num_classes = num_classes_from_output_shape(out_shape)
        except Exception as exc:
            debug_keras_failure("output shape inspection", exc)

        setattr(obj, KERAS_HANDLE_ATTR, handle)
        patch_instance_call_once(
            obj,
            patch_name=KERAS_CALL_PATCH_NAME,
            make_patched_call=functools.partial(
                build_patched_call,
                num_classes=static_num_classes,
                static_input_modality=static_input_modality,
            ),
        )
