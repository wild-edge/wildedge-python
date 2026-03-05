"""PyTorch integration."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from wildedge.events.inference import (
    ClassificationOutputMeta,
    HistogramSummary,
    ImageInputMeta,
    TopKPrediction,
)
from wildedge.hf_cache import downloads_from_cache_diff, scan_model_caches
from wildedge.integrations.base import BaseExtractor
from wildedge.logging import logger
from wildedge.model import ModelInfo
from wildedge.timing import elapsed_ms

try:
    import timm as _timm
    from timm.data import ImageNetInfo as _ImageNetInfo
except ImportError:
    _timm = None  # type: ignore[assignment]
    _ImageNetInfo = None  # type: ignore[assignment,misc]

try:
    import torch as _torch
except ImportError:
    _torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from wildedge.model import ModelHandle

_GENERIC_MODULE_NAMES = {"Module", "Sequential", "ModuleList", "ModuleDict"}
_timm_patched = False
_TIMM_PATCH_LOCK = threading.Lock()
TIMM_AUTO_LOAD_PATCH_NAME = "timm_auto_load"


def _debug_pytorch_failure(context: str, exc: BaseException) -> None:
    logger.debug("wildedge: pytorch %s failed: %s", context, exc)


def _is_torch_module(obj: object) -> bool:
    return any(
        c.__name__ == "Module" and c.__module__.startswith("torch")
        for c in type(obj).__mro__
    )


def _parameter_device_type(obj: object) -> str | None:
    try:
        first_param = next(obj.parameters())  # type: ignore[union-attr]
    except StopIteration:
        return None
    except Exception as exc:
        _debug_pytorch_failure("parameter device discovery", exc)
        return None
    return str(getattr(first_param.device, "type", "") or None)


def _detect_accelerator(obj: object) -> str:
    device_type = _parameter_device_type(obj)
    if device_type:
        return device_type  # 'cpu', 'cuda', 'mps', 'xpu', etc.
    return "cpu"


def _detect_quantization(obj: object) -> str | None:
    try:
        for module in obj.modules():  # type: ignore[union-attr]
            cls = type(module)
            if "quantized" in cls.__module__ or "quant" in cls.__name__.lower():
                return "int8"
        for p in obj.parameters():  # type: ignore[union-attr]
            dtype = str(p.dtype)
            if "int8" in dtype or "qint" in dtype or "quint" in dtype:
                return "int8"
            if "bfloat16" in dtype:
                return "bf16"
            if "float16" in dtype:
                return "f16"
    except Exception as exc:
        _debug_pytorch_failure("quantization detection", exc)
    return None


def _image_input_meta(tensor: object) -> ImageInputMeta | None:
    """Extract ImageInputMeta from a (N, C, H, W) tensor. Best-effort, never raises."""
    try:
        shape = tensor.shape  # type: ignore[union-attr]
        if len(shape) != 4:
            return None
        _n, c, h, w = shape

        # Normalise to [0, 1] using the batch's own min/max so stats are
        # comparable regardless of the preprocessing normalisation applied.
        t_min = float(tensor.min().item())  # type: ignore[union-attr]
        t_max = float(tensor.max().item())  # type: ignore[union-attr]
        span = t_max - t_min
        if span > 0:
            norm = (tensor - t_min) / span  # type: ignore[operator]
        else:
            norm = tensor  # type: ignore[assignment]

        brightness_mean = round(float(norm.mean().item()), 4)  # type: ignore[union-attr]
        brightness_stddev = round(float(norm.std().item()), 4)  # type: ignore[union-attr]

        # 5-bucket histogram over normalised pixel values
        flat = norm.flatten()  # type: ignore[union-attr]
        edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        buckets = [
            int(((flat >= lo) & (flat < hi)).sum().item())
            for lo, hi in zip(edges, edges[1:])
        ]
        # last bucket is inclusive on both ends
        buckets[-1] += int((flat == 1.0).sum().item())

        return ImageInputMeta(
            width=int(w),
            height=int(h),
            channels=int(c),
            histogram_summary=HistogramSummary(
                brightness_mean=brightness_mean,
                brightness_stddev=brightness_stddev,
                brightness_buckets=buckets,
                contrast=brightness_stddev,  # RMS contrast ≡ std
            ),
        )
    except Exception as exc:
        _debug_pytorch_failure("image input meta extraction", exc)
        return None


def _build_imagenet_labels() -> list[str] | None:
    """Return a 1000-entry description list using timm's bundled ImageNet metadata."""
    try:
        info = _ImageNetInfo()  # type: ignore[misc]
        return [info.index_to_description(i) for i in range(1000)]
    except Exception as exc:
        _debug_pytorch_failure("imagenet label build", exc)
        return None


def _classification_output_meta(
    probs: object,
    num_classes: int,
    labels: list[str] | None = None,
    top_k: int = 5,
) -> ClassificationOutputMeta:
    """Build ClassificationOutputMeta from a (N, C) softmax probability tensor."""
    top_conf = probs.max(dim=-1).values  # type: ignore[union-attr]  # (N,)
    avg_confidence = round(float(top_conf.mean().item()), 4)

    # Average predicted distribution across the batch, then take top-k.
    # This gives the most-predicted classes for this batch as a whole and is
    # useful for tracking prediction distribution drift over time.
    k = min(top_k, num_classes)
    avg_probs = probs.mean(dim=0)  # type: ignore[union-attr]  # (C,)
    top_vals, top_idx = avg_probs.topk(k)  # type: ignore[union-attr]
    predictions = [
        TopKPrediction(
            label=labels[int(idx.item())] if labels else str(int(idx.item())),
            confidence=round(float(conf.item()), 4),
        )
        for conf, idx in zip(top_vals, top_idx)
    ]

    return ClassificationOutputMeta(
        avg_confidence=avg_confidence,
        num_predictions=num_classes,
        top_k=predictions,
    )


class PytorchExtractor(BaseExtractor):
    def can_handle(self, obj: object) -> bool:
        return _is_torch_module(obj)

    def extract_info(
        self, obj: object, overrides: dict
    ) -> tuple[str | None, ModelInfo]:
        class_name = type(obj).__name__
        model_id = overrides.pop("id", None) or class_name

        if class_name in _GENERIC_MODULE_NAMES:
            logger.warning(
                "wildedge: PyTorch model_id defaults to generic class name '%s' - "
                "pass model_id= to register_model() for a stable unique ID",
                class_name,
            )

        quantization = overrides.pop("quantization", None) or _detect_quantization(obj)
        family = overrides.pop("family", None)
        version = overrides.pop("version", "unknown")
        source = overrides.pop("source", "local")

        info = ModelInfo(
            model_name=class_name,
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
            params = sum(p.numel() * p.element_size() for p in obj.parameters())  # type: ignore[union-attr]
            buffers = sum(b.numel() * b.element_size() for b in obj.buffers())  # type: ignore[union-attr]
            return params + buffers
        except Exception as exc:
            _debug_pytorch_failure("model memory calculation", exc)
            return None

    def install_hooks(self, obj: object, handle: ModelHandle) -> None:
        handle.detected_accelerator = _detect_accelerator(obj)
        _local = threading.local()

        # Detect classifier at hook-installation time so the post_hook closure
        # doesn't repeat this check on every forward pass.
        num_classes: int = 0
        labels: list[str] | None = None
        try:
            nc = getattr(obj, "num_classes", 0)
            if isinstance(nc, int) and nc > 1:
                num_classes = nc
                if num_classes == 1000 and _timm is not None:
                    labels = _build_imagenet_labels()
        except Exception as exc:
            _debug_pytorch_failure("num_classes detection", exc)

        def pre_hook(module, args):
            _local.t0 = time.perf_counter()

        def post_hook(module, args, output):
            t0 = getattr(_local, "t0", None)
            duration_ms = elapsed_ms(t0) if t0 is not None else 0

            batch_size: int | None = None
            input_modality: str | None = None
            output_modality: str | None = None
            input_meta = None
            output_meta = None

            first_arg = args[0] if args else None
            if first_arg is not None and hasattr(first_arg, "shape"):
                try:
                    batch_size = int(first_arg.shape[0])
                except Exception as exc:
                    _debug_pytorch_failure("batch size extraction", exc)
                if len(getattr(first_arg, "shape", ())) == 4:
                    input_modality = "image"
                    input_meta = _image_input_meta(first_arg)

            if num_classes > 0:
                try:
                    probs = _torch.softmax(output, dim=-1)  # type: ignore[union-attr]
                    output_meta = _classification_output_meta(
                        probs, num_classes, labels
                    )
                    output_modality = "classification"
                except Exception as exc:
                    _debug_pytorch_failure("classification output metadata", exc)

            handle.track_inference(
                duration_ms=duration_ms,
                input_modality=input_modality,
                output_modality=output_modality,
                batch_size=batch_size,
                input_meta=input_meta,
                output_meta=output_meta,
                success=True,
            )

        obj.register_forward_pre_hook(pre_hook)  # type: ignore[union-attr]
        obj.register_forward_hook(post_hook)  # type: ignore[union-attr]

    @classmethod
    def install_timm_patch(cls, client_ref: object) -> None:
        """Patch timm.create_model for automatic load, download, and unload tracking.

        Called once at WildEdge client initialisation. Any subsequent
        ``timm.create_model(...)`` call is timed and registered automatically.
        When ``pretrained=True``, HuggingFace Hub downloads are intercepted for
        the duration of the call and emitted as a model_download event.
        Inference tracking is handled by the existing PyTorch forward hooks.
        """
        global _timm_patched
        if _timm_patched or _timm is None:
            return

        with _TIMM_PATCH_LOCK:
            if _timm_patched:
                return

            original_create_model = _timm.create_model
            if (
                getattr(original_create_model, "__wildedge_patch_name__", None)
                == TIMM_AUTO_LOAD_PATCH_NAME
            ):
                _timm_patched = True
                return

            def patched_create_model(*args, **kwargs):  # type: ignore[no-untyped-def]
                before = scan_model_caches()
                t0 = time.perf_counter()
                model = original_create_model(*args, **kwargs)
                load_ms = elapsed_ms(t0)
                downloads = downloads_from_cache_diff(
                    before, scan_model_caches(), load_ms
                )
                logger.debug(
                    "wildedge: timm.create_model done load_ms=%d download_records=%d",
                    load_ms,
                    len(downloads),
                )
                c = client_ref()  # type: ignore[call-arg]
                if c is not None and not c.closed:
                    c._on_model_auto_loaded(model, load_ms=load_ms, downloads=downloads)
                return model

            patched_create_model.__wildedge_patch_name__ = TIMM_AUTO_LOAD_PATCH_NAME  # type: ignore[attr-defined]
            patched_create_model.__wildedge_original_call__ = original_create_model  # type: ignore[attr-defined]
            _timm.create_model = patched_create_model
            _timm_patched = True
