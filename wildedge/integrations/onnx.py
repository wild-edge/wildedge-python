"""ONNX Runtime integration."""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING, Any

from wildedge import constants
from wildedge.events.inference import (
    ClassificationOutputMeta,
    HistogramSummary,
    ImageInputMeta,
    TopKPrediction,
)
from wildedge.integrations.base import BaseExtractor
from wildedge.logging import logger
from wildedge.model import ModelInfo
from wildedge.timing import elapsed_ms

if TYPE_CHECKING:
    from wildedge.model import ModelHandle

try:
    import numpy as np  # type: ignore[import-untyped]
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import onnxruntime as ort  # type: ignore[import-untyped]
except ImportError:
    ort = None  # type: ignore[assignment]


_ort_patched = False
_ORT_PATCH_LOCK = threading.Lock()
_GENERIC_GRAPH_NAMES = {"torch_jit", "main", "model", "network", "graph", "onnx_model"}
ONNX_AUTO_LOAD_PATCH_NAME = "onnx_auto_load"


def _debug_onnx_failure(context: str, exc: BaseException) -> None:
    logger.debug("wildedge: onnx %s failed: %s", context, exc)


def _model_id_from_path(path: str) -> str | None:
    """Derive a useful model_id from an ONNX file path.

    Handles two cases:
    - HF Hub cache path (``models--org--name/...``): returns ``org/name``
    - Plain file path: returns the stem if it's not a generic name
    """
    for part in path.replace("\\", "/").split("/"):
        if part.startswith("models--"):
            return part[len("models--") :].replace("--", "/", 1)
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem if stem and stem not in _GENERIC_GRAPH_NAMES else None


def _is_ort_session(obj: object) -> bool:
    return ort is not None and isinstance(obj, ort.InferenceSession)


_PROVIDER_TO_ACCELERATOR: dict[str, str] = {
    "CUDAExecutionProvider": "cuda",
    "TensorrtExecutionProvider": "cuda",
    "ROCMExecutionProvider": "rocm",
    "MIGraphXExecutionProvider": "rocm",
    "CoreMLExecutionProvider": "coreml",
    "DmlExecutionProvider": "directml",
    "OpenVINOExecutionProvider": "openvino",
}


def _detect_accelerator(session: Any) -> str:
    try:
        for provider in session.get_providers():
            acc = _PROVIDER_TO_ACCELERATOR.get(provider)
            if acc:
                return acc
    except Exception as exc:
        _debug_onnx_failure("accelerator detection", exc)
    return "cpu"


def _detect_quantization(session: Any) -> str | None:
    """Inspect graph nodes for quantization markers."""
    # Best effort: infer quantization from input dtype
    try:
        inputs = session.get_inputs()
        if inputs:
            dtype = inputs[0].type
            if "int8" in dtype or "uint8" in dtype:
                return "int8"
            if "float16" in dtype:
                return "f16"
            if "float32" in dtype:
                return "f32"
    except Exception as exc:
        _debug_onnx_failure("quantization detection", exc)

    return None


def _image_input_meta(arr: Any) -> ImageInputMeta | None:
    """Extract ImageInputMeta from a (N, C, H, W) numpy array. Best-effort, never raises."""
    try:
        if np is None:
            return None
        shape = arr.shape
        if len(shape) != 4:
            return None
        _n, c, h, w = shape

        floats = arr.astype(np.float32)
        t_min = float(floats.min())
        t_max = float(floats.max())
        span = t_max - t_min
        norm = (floats - t_min) / span if span > 0 else floats

        brightness_mean = round(float(norm.mean()), 4)
        brightness_stddev = round(float(norm.std()), 4)

        flat = norm.flatten()
        edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        buckets = [
            int(((flat >= lo) & (flat < hi)).sum()) for lo, hi in zip(edges, edges[1:])
        ]
        buckets[-1] += int((flat == 1.0).sum())

        return ImageInputMeta(
            width=int(w),
            height=int(h),
            channels=int(c),
            histogram_summary=HistogramSummary(
                brightness_mean=brightness_mean,
                brightness_stddev=brightness_stddev,
                brightness_buckets=buckets,
                contrast=brightness_stddev,
            ),
        )
    except Exception as exc:
        _debug_onnx_failure("image input meta extraction", exc)
        return None


class OnnxExtractor(BaseExtractor):
    def can_handle(self, obj: object) -> bool:
        return _is_ort_session(obj)

    def extract_info(
        self, obj: object, overrides: dict
    ) -> tuple[str | None, ModelInfo]:
        meta = obj.get_modelmeta()  # type: ignore[union-attr]

        raw_name = (meta.graph_name or "").strip()
        graph_name = (
            raw_name if raw_name and raw_name not in _GENERIC_GRAPH_NAMES else None
        )
        model_id = overrides.pop("id", None) or graph_name

        raw_version = str(meta.version) if meta.version else None
        version = overrides.pop("version", None) or (
            raw_version if raw_version and raw_version != "0" else None
        )
        if version is None:
            logger.warning(
                "wildedge: ONNX model version could not be detected - sending as null"
            )
            version = "unknown"

        family = overrides.pop("family", None)
        if family is None:
            logger.warning(
                "wildedge: ONNX model family could not be detected - sending as null"
            )

        quantization = overrides.pop("quantization", None)
        if quantization is None:
            quantization = _detect_quantization(obj)
            if quantization is None:
                logger.warning(
                    "wildedge: ONNX model quantization could not be detected - sending as null"
                )

        source = overrides.pop("source", "local")

        info = ModelInfo(
            model_name=graph_name or model_id or "onnx-model",
            model_version=version,
            model_source=source,
            model_format="onnx",
            model_family=family,
            quantization=quantization,
        )
        for k, v in overrides.items():
            if hasattr(info, k):
                setattr(info, k, v)

        return model_id, info

    def install_hooks(self, obj: object, handle: ModelHandle) -> None:
        handle.detected_accelerator = _detect_accelerator(obj)

        # Detect classification output at hook-install time from session metadata.
        num_classes: int = 0
        try:
            outputs = obj.get_outputs()  # type: ignore[union-attr]
            if outputs:
                out_shape = outputs[0].shape
                if (
                    len(out_shape) == 2
                    and isinstance(out_shape[1], int)
                    and out_shape[1] > 1
                ):
                    num_classes = out_shape[1]
        except Exception as exc:
            _debug_onnx_failure("output shape inspection", exc)

        original_run = obj.run  # type: ignore[union-attr]

        def patched_run(output_names, input_feed, run_options=None):
            first = next(iter(input_feed.values()), None) if input_feed else None

            batch_size: int | None = None
            input_modality = "structured"
            input_meta = None

            if first is not None and hasattr(first, "shape"):
                try:
                    batch_size = int(first.shape[0])
                except Exception as exc:
                    _debug_onnx_failure("batch size extraction", exc)
                if len(getattr(first, "shape", ())) == 4:
                    input_modality = "image"
                    input_meta = _image_input_meta(first)

            t0 = time.perf_counter()
            try:
                result = original_run(output_names, input_feed, run_options)
                duration_ms = elapsed_ms(t0)

                output_modality = "structured"
                output_meta = None
                if num_classes > 0 and result and np is not None:
                    try:
                        logits = result[0]  # (N, C)
                        # softmax over class axis
                        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
                        probs = exp / exp.sum(axis=-1, keepdims=True)
                        avg_probs = probs.mean(axis=0)
                        top_k = min(5, num_classes)
                        top_idx = avg_probs.argsort()[::-1][:top_k]
                        output_meta = ClassificationOutputMeta(
                            num_predictions=num_classes,
                            avg_confidence=round(float(probs.max(axis=-1).mean()), 4),
                            top_k=[
                                TopKPrediction(
                                    label=str(i),
                                    confidence=round(float(avg_probs[i]), 4),
                                )
                                for i in top_idx
                            ],
                        )
                        output_modality = "classification"
                    except Exception as exc:
                        _debug_onnx_failure("classification output metadata", exc)

                handle.track_inference(
                    duration_ms=duration_ms,
                    batch_size=batch_size,
                    input_modality=input_modality,
                    output_modality=output_modality,
                    input_meta=input_meta,
                    output_meta=output_meta,
                    success=True,
                )
                return result
            except Exception as exc:
                handle.track_error(
                    error_code="UNKNOWN",
                    error_message=str(exc)[: constants.ERROR_MSG_MAX_LEN],
                )
                raise

        obj.run = patched_run  # type: ignore[union-attr]

    @classmethod
    def install_auto_load_patch(cls, client_ref: object) -> None:
        """Replace ort.InferenceSession with a Python subclass that times __init__.

        ``ort.InferenceSession`` is a C extension type whose ``__init__`` cannot
        be replaced directly, so we swap the name in the ``onnxruntime`` module
        namespace with a thin Python subclass. Existing ``isinstance`` checks
        continue to work because the subclass inherits from the original.
        """
        global _ort_patched
        if _ort_patched or ort is None:
            return

        with _ORT_PATCH_LOCK:
            if _ort_patched:
                return

            current_cls = ort.InferenceSession
            if (
                getattr(current_cls, "__wildedge_patch_name__", None)
                == ONNX_AUTO_LOAD_PATCH_NAME
            ):
                _ort_patched = True
                return

            original_cls = current_cls
            original_init = original_cls.__init__

            def patched_init(self_inner, *args, **kwargs):  # type: ignore[no-untyped-def]
                path_arg = args[0] if args else kwargs.get("path_or_bytes")
                t0 = time.perf_counter()
                original_init(self_inner, *args, **kwargs)
                load_ms = elapsed_ms(t0)
                c = client_ref()  # type: ignore[call-arg]
                if c is not None and not c.closed:
                    model_id = (
                        _model_id_from_path(path_arg)
                        if isinstance(path_arg, str)
                        else None
                    )
                    c._on_model_auto_loaded(
                        self_inner, load_ms=load_ms, model_id=model_id
                    )

            patched_cls = type(  # type: ignore[misc]
                "InferenceSession",
                (original_cls,),
                {
                    "__init__": patched_init,
                    "__wildedge_patch_name__": ONNX_AUTO_LOAD_PATCH_NAME,
                    "__wildedge_original_class__": original_cls,
                },
            )
            ort.InferenceSession = patched_cls
            _ort_patched = True
