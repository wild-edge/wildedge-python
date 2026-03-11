"""Ultralytics integration."""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from wildedge import constants
from wildedge.events.inference import (
    ClassificationOutputMeta,
    DetectionOutputMeta,
    HistogramSummary,
    ImageInputMeta,
    TopKPrediction,
)
from wildedge.integrations.base import BaseExtractor, patch_instance_call_once
from wildedge.logging import logger
from wildedge.model import ModelInfo
from wildedge.timing import elapsed_ms

try:
    import ultralytics as _ultralytics
    from ultralytics.utils import (
        WEIGHTS_DIR as _ULTRALYTICS_WEIGHTS_DIR,  # type: ignore[import-untyped]
    )
except ImportError:
    _ultralytics = None  # type: ignore[assignment]
    _ULTRALYTICS_WEIGHTS_DIR = None  # type: ignore[assignment]

try:
    import numpy as np  # type: ignore[import-untyped]
except ImportError:
    np = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from wildedge.model import ModelHandle

# --- Patch state (mutable, module-level) ---
_ultralytics_patched = False
_ULTRALYTICS_PATCH_LOCK = threading.Lock()

# --- Marker names written onto patched objects and classes ---
YOLO_CALL_PATCH_NAME = "ultralytics_call"
YOLO_HANDLE_ATTR = "__wildedge_yolo_handle__"
YOLO_AUTO_LOAD_PATCH_NAME = "ultralytics_auto_load"

# --- Extracts family prefix from names like "yolov8n", "yolov9e", "yolo11n" ---
YOLO_FAMILY_RE = re.compile(r"^(yolo(?:v\d+|\d+))", re.IGNORECASE)


def debug_failure(context: str, exc: BaseException) -> None:
    logger.debug("wildedge: ultralytics %s failed: %s", context, exc)


def is_yolo(obj: object) -> bool:
    # String check avoids importing ultralytics when it is not installed.
    return type(obj).__name__ == "YOLO"


def detect_accelerator(obj: object) -> str:
    try:
        inner = getattr(obj, "model", None)
        if inner is not None:
            try:
                first_param = next(inner.parameters())
                device_type = str(getattr(first_param.device, "type", ""))
                if device_type:
                    return device_type
            except StopIteration:
                pass
    except Exception as exc:
        debug_failure("accelerator detection", exc)
    return "cpu"


def detect_quantization(obj: object) -> str | None:
    try:
        inner = getattr(obj, "model", None)
        if inner is None:
            return None
        for p in inner.parameters():
            dtype = str(p.dtype)
            if "bfloat16" in dtype:
                return "bf16"
            if "float16" in dtype:
                return "f16"
            if "int8" in dtype or "qint" in dtype:
                return "int8"
            if "float32" in dtype:
                return "f32"
            break  # only need the first parameter's dtype
    except Exception as exc:
        debug_failure("quantization detection", exc)
    return None


def image_input_meta(arr: object) -> ImageInputMeta | None:
    """Extract ImageInputMeta from a numpy array (H, W, C). Best-effort, never raises."""
    try:
        if np is None or not isinstance(arr, np.ndarray):
            return None
        shape = arr.shape
        if len(shape) == 3:
            h, w, c = shape
        elif len(shape) == 2:
            h, w = shape
            c = 1
        else:
            return None

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
        debug_failure("image input meta extraction", exc)
        return None


def detection_output_meta(
    results: list, names: dict | None
) -> DetectionOutputMeta | None:
    """Build DetectionOutputMeta from a list of ultralytics Results. Best-effort, never raises."""
    try:
        total_preds = 0
        all_confs: list[float] = []
        top_preds: list[TopKPrediction] = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            try:
                xyxy = boxes.xyxy.tolist() if hasattr(boxes.xyxy, "tolist") else []
                confs = boxes.conf.tolist() if hasattr(boxes.conf, "tolist") else []
                classes = boxes.cls.tolist() if hasattr(boxes.cls, "tolist") else []
            except Exception:
                continue

            total_preds += len(confs)
            all_confs.extend(confs)

            for i, (conf, cls_id) in enumerate(zip(confs, classes)):
                if len(top_preds) >= 5:
                    break
                label = (names or {}).get(int(cls_id), str(int(cls_id)))
                bbox = [int(v) for v in xyxy[i]] if i < len(xyxy) else None
                top_preds.append(
                    TopKPrediction(
                        label=label,
                        confidence=round(float(conf), 4),
                        bbox=bbox,
                    )
                )

        avg_conf = round(sum(all_confs) / len(all_confs), 4) if all_confs else None
        return DetectionOutputMeta(
            task="detection",
            num_predictions=total_preds,
            top_k=top_preds if top_preds else None,
            avg_confidence=avg_conf,
            num_classes=len(names) if names else None,
        )
    except Exception as exc:
        debug_failure("detection output meta extraction", exc)
        return None


def classify_output_meta(
    results: list, names: dict | None
) -> ClassificationOutputMeta | None:
    """Build ClassificationOutputMeta from a list of ultralytics Results for classify task.
    Best-effort, never raises."""
    try:
        top_preds: list[TopKPrediction] = []
        top1_conf: float | None = None

        for result in results:
            probs = getattr(result, "probs", None)
            if probs is None:
                continue
            try:
                top5 = probs.top5
                top5conf = (
                    probs.top5conf.tolist() if hasattr(probs.top5conf, "tolist") else []
                )
                for cls_id, conf in zip(top5, top5conf):
                    label = (names or {}).get(int(cls_id), str(int(cls_id)))
                    top_preds.append(
                        TopKPrediction(label=label, confidence=round(float(conf), 4))
                    )
                if top5conf:
                    top1_conf = round(float(top5conf[0]), 4)
            except Exception:
                continue
            break  # one result per sample for classify

        return ClassificationOutputMeta(
            num_predictions=len(top_preds),
            top_k=top_preds if top_preds else None,
            avg_confidence=top1_conf,
        )
    except Exception as exc:
        debug_failure("classify output meta extraction", exc)
        return None


def weights_file_exists(model_arg: object) -> bool:
    """Return True if the weights file appears to already be on disk."""
    if not isinstance(model_arg, str):
        return True  # not a path string — weights already in memory or a loaded object
    p = Path(model_arg)
    if p.is_file():
        return True
    # ultralytics resolves short names (e.g. "yolov8n.pt") against its assets dir
    if (
        _ULTRALYTICS_WEIGHTS_DIR is not None
        and (_ULTRALYTICS_WEIGHTS_DIR / p.name).is_file()
    ):
        return True
    return False


def build_download_record(obj: object, load_ms: int) -> dict | None:
    """Build a single download record for the model weights file. Best-effort."""
    try:
        ckpt_path = getattr(obj, "ckpt_path", None)
        if not ckpt_path:
            return None
        p = Path(ckpt_path)
        if not p.is_file():
            return None
        file_size = p.stat().st_size
        bandwidth_bps = int(file_size / load_ms * 1000) if load_ms > 0 else None
        return {
            "repo_id": p.stem,
            "source_type": "ultralytics",
            "source_url": f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{p.name}",
            "size": file_size,
            "duration_ms": load_ms,
            "cache_hit": False,
            "bandwidth_bps": bandwidth_bps,
        }
    except Exception as exc:
        debug_failure("download record build", exc)
    return None


def build_patched_call(original_call):  # type: ignore[no-untyped-def]
    def patched_call(self_inner, *args, **kwargs):  # type: ignore[no-untyped-def]
        handle = getattr(self_inner, YOLO_HANDLE_ATTR, None)
        if handle is None:
            return original_call(self_inner, *args, **kwargs)

        source = args[0] if args else kwargs.get("source")

        batch_size: int | None = None
        input_meta = None
        if source is not None and np is not None:
            try:
                if isinstance(source, np.ndarray):
                    input_meta = image_input_meta(source)
                    batch_size = 1
                elif isinstance(source, list) and source:
                    batch_size = len(source)
                    if isinstance(source[0], np.ndarray):
                        input_meta = image_input_meta(source[0])
            except Exception as exc:
                debug_failure("input meta extraction", exc)

        t0 = time.perf_counter()
        try:
            results = original_call(self_inner, *args, **kwargs)
            duration_ms = elapsed_ms(t0)

            task = getattr(self_inner, "task", "detect") or "detect"
            names = getattr(self_inner, "names", None)

            output_meta = None
            output_modality = "detection"
            if isinstance(results, list) and results:
                try:
                    if task == "classify":
                        output_meta = classify_output_meta(results, names)
                        output_modality = "classification"
                    else:
                        output_meta = detection_output_meta(results, names)
                except Exception as exc:
                    debug_failure("output meta extraction", exc)

            handle.track_inference(
                duration_ms=duration_ms,
                batch_size=batch_size,
                input_modality="image",
                output_modality=output_modality,
                input_meta=input_meta,
                output_meta=output_meta,
                success=True,
            )
            return results
        except Exception as exc:
            handle.track_error(
                error_code="UNKNOWN",
                error_message=str(exc)[: constants.ERROR_MSG_MAX_LEN],
            )
            raise

    return patched_call


class UltralyticsExtractor(BaseExtractor):
    def can_handle(self, obj: object) -> bool:
        return is_yolo(obj)

    def extract_info(
        self, obj: object, overrides: dict
    ) -> tuple[str | None, ModelInfo]:
        ckpt_path = getattr(obj, "ckpt_path", None) or ""
        stem = Path(ckpt_path).stem if ckpt_path else None
        model_name = stem or "yolo"
        model_id = overrides.pop("id", None) or model_name

        family = overrides.pop("family", None)
        if family is None and model_name:
            m = YOLO_FAMILY_RE.match(model_name)
            family = m.group(1).lower() if m else None
        if family is None:
            logger.warning(
                "wildedge: ultralytics model family could not be detected - sending as null"
            )

        quantization = overrides.pop("quantization", None) or detect_quantization(obj)
        if quantization is None:
            logger.warning(
                "wildedge: ultralytics model quantization could not be detected - sending as null"
            )

        version = overrides.pop("version", "unknown")
        source = overrides.pop("source", "local")

        info = ModelInfo(
            model_name=model_name,
            model_version=version,
            model_source=source,
            model_format="pytorch",
            model_family=family,
            quantization=quantization,
        )
        for k, v in overrides.items():
            if hasattr(info, k):
                setattr(info, k, v)

        return model_id, info

    def memory_bytes(self, obj: object) -> int | None:
        try:
            ckpt_path = getattr(obj, "ckpt_path", None)
            if ckpt_path:
                return Path(ckpt_path).stat().st_size
        except Exception as exc:
            debug_failure("model size detection", exc)
        return None

    def install_hooks(self, obj: object, handle: ModelHandle) -> None:
        handle.detected_accelerator = detect_accelerator(obj)
        setattr(obj, YOLO_HANDLE_ATTR, handle)
        patch_instance_call_once(
            obj,
            patch_name=YOLO_CALL_PATCH_NAME,
            make_patched_call=build_patched_call,
        )

    @classmethod
    def install_auto_load_patch(cls, client_ref: object) -> None:
        """Patch ultralytics.YOLO.__init__ for automatic load, unload, and download tracking.

        Called once at WildEdge client initialisation. Any subsequent
        ``YOLO(...)`` construction is timed and registered automatically.
        If the weights file does not exist before the call, a download event
        is emitted with the file size and the total load duration.
        """
        global _ultralytics_patched
        if _ultralytics_patched or _ultralytics is None:
            return

        with _ULTRALYTICS_PATCH_LOCK:
            if _ultralytics_patched:
                return

            original_init = _ultralytics.YOLO.__init__
            if (
                getattr(original_init, "__wildedge_patch_name__", None)
                == YOLO_AUTO_LOAD_PATCH_NAME
            ):
                _ultralytics_patched = True
                return

            def patched_init(self_inner, *args, **kwargs):  # type: ignore[no-untyped-def]
                model_arg = args[0] if args else kwargs.get("model", "yolov8n.pt")
                weights_existed = weights_file_exists(model_arg)

                t0 = time.perf_counter()
                original_init(self_inner, *args, **kwargs)
                load_ms = elapsed_ms(t0)

                c = client_ref()  # type: ignore[call-arg]
                if c is not None and not c.closed:
                    downloads = None
                    if not weights_existed:
                        record = build_download_record(self_inner, load_ms)
                        if record is not None:
                            downloads = [record]
                    c._on_model_auto_loaded(
                        self_inner, load_ms=load_ms, downloads=downloads
                    )

            patched_init.__wildedge_patch_name__ = YOLO_AUTO_LOAD_PATCH_NAME  # type: ignore[attr-defined]
            patched_init.__wildedge_original_call__ = original_init  # type: ignore[attr-defined]
            _ultralytics.YOLO.__init__ = patched_init
            _ultralytics_patched = True
