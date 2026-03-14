"""HuggingFace Transformers integration."""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING

from wildedge import constants
from wildedge.events.inference import (
    ClassificationOutputMeta,
    EmbeddingOutputMeta,
    GenerationOutputMeta,
    TextInputMeta,
    TopKPrediction,
)
from wildedge.integrations.base import BaseExtractor, patch_instance_call_once
from wildedge.logging import logger
from wildedge.model import ModelInfo
from wildedge.timing import elapsed_ms

try:
    import transformers as _transformers
except ImportError:
    _transformers = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from wildedge.model import ModelHandle

# --- Patch state ---
_transformers_patched = False
_TRANSFORMERS_PATCH_LOCK = threading.Lock()
TRANSFORMERS_AUTO_LOAD_PATCH_NAME = "transformers_auto_load"

# --- Pipeline instance patching ---
PIPELINE_CALL_PATCH_NAME = "transformers_pipeline_call"
PIPELINE_HANDLE_ATTR = "__wildedge_pipeline_handle__"

# Thread-local flag: suppress from_pretrained tracking when called inside pipeline()
_tl = threading.local()


def _debug_failure(context: str, exc: BaseException) -> None:
    logger.debug("wildedge: transformers %s failed: %s", context, exc)


def _is_pretrained_model(obj: object) -> bool:
    """String-check avoids importing transformers when not installed."""
    for cls in type(obj).__mro__:
        if cls.__name__ == "PreTrainedModel" and "transformers" in cls.__module__:
            return True
    return False


def _is_pipeline(obj: object) -> bool:
    for cls in type(obj).__mro__:
        if cls.__name__ == "Pipeline" and "transformers" in cls.__module__:
            return True
    return False


def _extract_model_config(obj: object) -> tuple[str | None, str | None, str | None]:
    """Returns (name_or_path, model_type, architectures[0]). Never raises."""
    try:
        config = getattr(obj, "config", None)
        if config is None:
            inner = getattr(obj, "model", None)
            config = getattr(inner, "config", None) if inner is not None else None
        if config is None:
            return None, None, None
        name_or_path = getattr(config, "name_or_path", None) or None
        model_type = getattr(config, "model_type", None) or None
        archs = getattr(config, "architectures", None)
        arch = archs[0] if archs else None
        return name_or_path, model_type, arch
    except Exception as exc:
        _debug_failure("config extraction", exc)
        return None, None, None


def _is_local_path(name_or_path: str | None) -> bool:
    if not name_or_path:
        return False
    return os.path.sep in name_or_path or os.path.exists(name_or_path)


def _detect_quantization(obj: object) -> str | None:
    try:
        # Prefer quantization_config (bitsandbytes, GPTQ, AWQ, etc.)
        config = getattr(obj, "config", None)
        if config is None:
            inner = getattr(obj, "model", None)
            config = getattr(inner, "config", None) if inner is not None else None
        if config is not None:
            qconfig = getattr(config, "quantization_config", None)
            if qconfig is not None:
                quant_type = getattr(qconfig, "quant_type", None) or getattr(
                    qconfig, "quantization_type", None
                )
                bits = getattr(qconfig, "bits", None) or getattr(
                    qconfig, "num_bits", None
                )
                if quant_type:
                    return str(quant_type).lower()
                if bits:
                    return f"int{int(bits)}"
        # Fall back to model dtype
        model = getattr(obj, "model", obj)
        dtype = getattr(model, "dtype", None)
        if dtype is not None:
            s = str(dtype)
            if "bfloat16" in s:
                return "bf16"
            if "float16" in s:
                return "f16"
            if "int8" in s:
                return "int8"
    except Exception as exc:
        _debug_failure("quantization detection", exc)
    return None


def _detect_accelerator(obj: object) -> str:
    try:
        model = getattr(obj, "model", obj)
        first = next(model.parameters())  # type: ignore[union-attr]
        return str(getattr(first.device, "type", "cpu") or "cpu")
    except Exception:
        pass
    return "cpu"


def _infer_task_from_arch(arch: str | None) -> str | None:
    """Guess broad task category from architecture class name."""
    if not arch:
        return None
    lower = arch.lower()
    if any(k in lower for k in ("forsequenceclassification", "fortokenclassification")):
        return "classification"
    if any(
        k in lower for k in ("forcausallm", "forseq2seqlm", "forconditionalgeneration")
    ):
        return "generation"
    if lower.endswith("model"):
        return "embedding"
    return None


# ---------------------------------------------------------------------------
# Pipeline call patching
# ---------------------------------------------------------------------------


def _pipeline_input_meta(inputs: object) -> TextInputMeta | None:
    try:
        texts: list[str] = []
        if isinstance(inputs, str):
            texts = [inputs]
        elif isinstance(inputs, list):
            texts = [t for t in inputs if isinstance(t, str)]
        if not texts:
            return None
        char_count = sum(len(t) for t in texts)
        word_count = sum(len(t.split()) for t in texts)
        return TextInputMeta(char_count=char_count, word_count=word_count)
    except Exception as exc:
        _debug_failure("pipeline input meta", exc)
        return None


def _pipeline_output_meta(
    task: str | None, outputs: object
) -> ClassificationOutputMeta | GenerationOutputMeta | EmbeddingOutputMeta | None:
    if task is None:
        return None
    try:
        t = task.lower()
        if any(
            k in t
            for k in (
                "classification",
                "sentiment",
                "zero-shot",
                "ner",
                "token-class",
            )
        ):
            if isinstance(outputs, list) and outputs:
                first = outputs[0]
                if isinstance(first, dict) and "score" in first:
                    label = str(first.get("label", ""))
                    score = round(float(first["score"]), 4)
                    return ClassificationOutputMeta(
                        avg_confidence=score,
                        top_k=[TopKPrediction(label=label, confidence=score)],
                    )
            return ClassificationOutputMeta()

        if any(
            k in t for k in ("generation", "translation", "summariz", "conversational")
        ):
            tokens_out: int | None = None
            if isinstance(outputs, list) and outputs:
                first = outputs[0]
                if isinstance(first, dict):
                    text = (
                        first.get("generated_text")
                        or first.get("translation_text")
                        or first.get("summary_text")
                    )
                    if text:
                        tokens_out = len(str(text).split())
            return GenerationOutputMeta(tokens_out=tokens_out)

        if any(k in t for k in ("feature", "embed", "similarity")):
            dims: int | None = None
            if isinstance(outputs, list) and outputs:
                first = outputs[0]
                if (
                    isinstance(first, list)
                    and first
                    and isinstance(first[0], (int, float))
                ):
                    dims = len(first)
                elif isinstance(first, list) and first and isinstance(first[0], list):
                    # [[[token_embs]]] shape for feature-extraction
                    dims = len(first[0])
            return EmbeddingOutputMeta(dimensions=dims)

    except Exception as exc:
        _debug_failure("pipeline output meta", exc)
    return None


def _pipeline_modalities(task: str | None) -> tuple[str | None, str | None]:
    if not task:
        return "text", None
    t = task.lower()
    if any(k in t for k in ("classification", "sentiment", "zero-shot", "ner")):
        return "text", "classification"
    if any(k in t for k in ("generation", "translation", "summariz", "conversational")):
        return "text", "generation"
    if any(k in t for k in ("feature", "embed", "similarity")):
        return "text", "embedding"
    return "text", None


def _build_pipeline_patched_call(original_call):  # type: ignore[no-untyped-def]
    def patched_call(self_inner, inputs, *args, **kwargs):  # type: ignore[no-untyped-def]
        handle = getattr(self_inner, PIPELINE_HANDLE_ATTR, None)
        if handle is None:
            return original_call(self_inner, inputs, *args, **kwargs)

        task = getattr(self_inner, "task", None)
        batch_size: int | None = (
            len(inputs)
            if isinstance(inputs, list)
            else (1 if isinstance(inputs, str) else None)
        )
        input_meta = _pipeline_input_meta(inputs)
        input_modality, output_modality = _pipeline_modalities(task)

        t0 = time.perf_counter()
        try:
            outputs = original_call(self_inner, inputs, *args, **kwargs)
            duration_ms = elapsed_ms(t0)
            output_meta = _pipeline_output_meta(task, outputs)
            handle.track_inference(
                duration_ms=duration_ms,
                batch_size=batch_size,
                input_modality=input_modality,
                output_modality=output_modality,
                input_meta=input_meta,
                output_meta=output_meta,
                success=True,
            )
            return outputs
        except Exception as exc:
            handle.track_error(
                error_code="UNKNOWN",
                error_message=str(exc)[: constants.ERROR_MSG_MAX_LEN],
            )
            raise

    return patched_call


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class TransformersExtractor(BaseExtractor):
    def can_handle(self, obj: object) -> bool:
        return _is_pretrained_model(obj) or _is_pipeline(obj)

    def extract_info(
        self, obj: object, overrides: dict
    ) -> tuple[str | None, ModelInfo]:
        name_or_path, model_type, arch = _extract_model_config(obj)

        model_name = arch or model_type or type(obj).__name__
        model_id = overrides.pop("id", None) or name_or_path or model_name
        family = overrides.pop("family", None) or model_type
        version = overrides.pop("version", "unknown")
        source = overrides.pop("source", None)
        if source is None:
            source = "local" if _is_local_path(name_or_path) else "huggingface"
        quantization = overrides.pop("quantization", None) or _detect_quantization(obj)

        info = ModelInfo(
            model_name=model_name,
            model_version=version,
            model_source=source,
            model_format="transformers",
            model_family=family,
            quantization=quantization,
        )
        for k, v in overrides.items():
            if hasattr(info, k):
                setattr(info, k, v)

        return model_id, info

    def memory_bytes(self, obj: object) -> int | None:
        try:
            model = getattr(obj, "model", obj)
            params = sum(p.numel() * p.element_size() for p in model.parameters())  # type: ignore[union-attr]
            buffers = sum(b.numel() * b.element_size() for b in model.buffers())  # type: ignore[union-attr]
            return params + buffers
        except Exception as exc:
            _debug_failure("memory estimation", exc)
            return None

    def install_hooks(self, obj: object, handle: ModelHandle) -> None:
        handle.detected_accelerator = _detect_accelerator(obj)

        if _is_pipeline(obj):
            setattr(obj, PIPELINE_HANDLE_ATTR, handle)
            patch_instance_call_once(
                obj,
                patch_name=PIPELINE_CALL_PATCH_NAME,
                make_patched_call=_build_pipeline_patched_call,
            )
        else:
            # PreTrainedModel: use PyTorch forward hooks
            _local = threading.local()
            _, _, arch = _extract_model_config(obj)
            task_hint = _infer_task_from_arch(arch)

            def pre_hook(module, args):  # type: ignore[no-untyped-def]
                _local.t0 = time.perf_counter()

            def post_hook(module, args, output):  # type: ignore[no-untyped-def]
                t0 = getattr(_local, "t0", None)
                duration_ms = elapsed_ms(t0) if t0 is not None else 0

                batch_size: int | None = None
                input_meta: TextInputMeta | None = None
                input_modality, output_modality = "text", task_hint

                try:
                    # args[0] is typically input_ids: (batch, seq_len)
                    input_ids = args[0] if args else None
                    if input_ids is not None and hasattr(input_ids, "shape"):
                        shape = input_ids.shape
                        if len(shape) >= 1:
                            batch_size = int(shape[0])
                        if len(shape) >= 2:
                            input_meta = TextInputMeta(token_count=int(shape[1]))
                except Exception as exc:
                    _debug_failure("forward hook input extraction", exc)

                handle.track_inference(
                    duration_ms=duration_ms,
                    batch_size=batch_size,
                    input_modality=input_modality,
                    output_modality=output_modality,
                    input_meta=input_meta,
                    output_meta=None,
                    success=True,
                )

            obj.register_forward_pre_hook(pre_hook)  # type: ignore[union-attr]
            obj.register_forward_hook(post_hook)  # type: ignore[union-attr]

    # -----------------------------------------------------------------------
    # Auto-load patches
    # -----------------------------------------------------------------------

    @classmethod
    def install_auto_load_patch(cls, client_ref: object) -> None:
        """Patch transformers.pipeline and PreTrainedModel.from_pretrained.

        Called once at WildEdge client initialisation. Any subsequent
        ``pipeline(...)`` or ``AutoModel.from_pretrained(...)`` call is timed
        and registered automatically. HuggingFace Hub downloads are intercepted
        for the duration of the call and emitted as a model_download event.
        A thread-local guard prevents double-tracking when pipeline() calls
        from_pretrained() internally.
        """
        global _transformers_patched
        if _transformers_patched or _transformers is None:
            return

        with _TRANSFORMERS_PATCH_LOCK:
            if _transformers_patched:
                return
            cls._patch_pipeline(client_ref)
            cls._patch_from_pretrained(client_ref)
            _transformers_patched = True

    @classmethod
    def _patch_pipeline(cls, client_ref: object) -> None:
        original_pipeline = _transformers.pipeline
        if (
            getattr(original_pipeline, "__wildedge_patch_name__", None)
            == TRANSFORMERS_AUTO_LOAD_PATCH_NAME
        ):
            return

        def patched_pipeline(*args, **kwargs):  # type: ignore[no-untyped-def]
            c = client_ref()  # type: ignore[call-arg]
            hub_before = (
                c._snapshot_hub_caches() if c is not None and not c.closed else {}
            )
            t0 = time.perf_counter()
            _tl.inside_pipeline = True
            try:
                pipe = original_pipeline(*args, **kwargs)
            finally:
                _tl.inside_pipeline = False
            load_ms = elapsed_ms(t0)
            if c is not None and not c.closed:
                downloads = c._diff_hub_caches(hub_before, load_ms) or None
                c._on_model_auto_loaded(pipe, load_ms=load_ms, downloads=downloads)
            return pipe

        patched_pipeline.__wildedge_patch_name__ = TRANSFORMERS_AUTO_LOAD_PATCH_NAME  # type: ignore[attr-defined]
        patched_pipeline.__wildedge_original_call__ = original_pipeline  # type: ignore[attr-defined]
        _transformers.pipeline = patched_pipeline

    @classmethod
    def _patch_from_pretrained(cls, client_ref: object) -> None:
        original_bound = _transformers.PreTrainedModel.from_pretrained
        if (
            getattr(original_bound, "__wildedge_patch_name__", None)
            == TRANSFORMERS_AUTO_LOAD_PATCH_NAME
        ):
            return

        original_func = original_bound.__func__

        def patched_from_pretrained(model_cls, *args, **kwargs):  # type: ignore[no-untyped-def]
            # Don't double-track models loaded inside pipeline()
            if getattr(_tl, "inside_pipeline", False):
                return original_func(model_cls, *args, **kwargs)
            c = client_ref()  # type: ignore[call-arg]
            hub_before = (
                c._snapshot_hub_caches() if c is not None and not c.closed else {}
            )
            t0 = time.perf_counter()
            model = original_func(model_cls, *args, **kwargs)
            load_ms = elapsed_ms(t0)
            if c is not None and not c.closed:
                downloads = c._diff_hub_caches(hub_before, load_ms) or None
                c._on_model_auto_loaded(model, load_ms=load_ms, downloads=downloads)
            return model

        patched_from_pretrained.__wildedge_patch_name__ = (
            TRANSFORMERS_AUTO_LOAD_PATCH_NAME  # type: ignore[attr-defined]
        )
        patched_from_pretrained.__wildedge_original_call__ = original_func  # type: ignore[attr-defined]
        _transformers.PreTrainedModel.from_pretrained = classmethod(  # type: ignore[assignment]
            patched_from_pretrained
        )
