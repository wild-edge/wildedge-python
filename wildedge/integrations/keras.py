"""Keras integration."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from wildedge import constants
from wildedge.device import CURRENT_PLATFORM
from wildedge.integrations.base import BaseExtractor, patch_instance_call_once
from wildedge.logging import logger
from wildedge.model import ModelInfo
from wildedge.timing import elapsed_ms

if TYPE_CHECKING:
    from wildedge.model import ModelHandle

KERAS_CALL_PATCH_NAME = "keras_call"
KERAS_HANDLE_ATTR = "__wildedge_keras_handle__"


def _is_keras_model(obj: object) -> bool:
    return any(
        c.__name__ == "Model" and "keras" in c.__module__ for c in type(obj).__mro__
    )


def _debug_keras_failure(context: str, exc: BaseException) -> None:
    logger.debug("wildedge: keras %s failed: %s", context, exc)


def _build_patched_call(original_call):
    def patched_call(self_inner, *args, **kwargs):
        handle = getattr(self_inner, KERAS_HANDLE_ATTR, None)
        if handle is None:
            return original_call(self_inner, *args, **kwargs)

        t0 = time.perf_counter()
        try:
            result = original_call(self_inner, *args, **kwargs)
            handle.track_inference(
                duration_ms=elapsed_ms(t0),
                input_modality="tensor",
                output_modality="tensor",
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


def _detect_accelerator(obj: object) -> str:
    try:
        weights = getattr(obj, "weights", None) or []
        if weights:
            device = str(getattr(weights[0], "device", "") or "")
            if any(g in device.upper() for g in ("GPU", "CUDA")):
                return CURRENT_PLATFORM.gpu_accelerator_for_offload()
    except Exception as exc:
        _debug_keras_failure("accelerator detection", exc)
    return "cpu"


class KerasExtractor(BaseExtractor):
    def can_handle(self, obj: object) -> bool:
        return _is_keras_model(obj)

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
        handle.detected_accelerator = _detect_accelerator(obj)
        setattr(obj, KERAS_HANDLE_ATTR, handle)
        patch_instance_call_once(
            obj,
            patch_name=KERAS_CALL_PATCH_NAME,
            make_patched_call=_build_patched_call,
        )
