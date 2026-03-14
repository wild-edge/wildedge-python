"""MLX / mlx-lm integration."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from wildedge import constants
from wildedge.events.inference import GenerationOutputMeta, TextInputMeta
from wildedge.integrations.base import BaseExtractor, patch_instance_call_once
from wildedge.logging import logger
from wildedge.model import ModelInfo
from wildedge.timing import elapsed_ms

try:
    import mlx.core as _mx
    import mlx.nn as _mlx_nn
    from mlx.utils import tree_flatten as _tree_flatten
except ImportError:
    _mx = None  # type: ignore[assignment]
    _mlx_nn = None  # type: ignore[assignment]
    _tree_flatten = None  # type: ignore[assignment]

try:
    import mlx_lm as _mlx_lm
except ImportError:
    _mlx_lm = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from wildedge.model import ModelHandle

# --- Patch state ---
_mlx_patched = False
_MLX_PATCH_LOCK = threading.Lock()
MLX_AUTO_LOAD_PATCH_NAME = "mlx_auto_load"
MLX_GENERATE_PATCH_NAME = "mlx_generate"
MLX_CALL_PATCH_NAME = "mlx_call"
MLX_HANDLE_ATTR = "__wildedge_mlx_handle__"

# Thread-local flag: suppress __call__ tracking inside mlx_lm.generate's
# autoregressive loop (which calls model() once per token).
_inside_mlx_generate = threading.local()


def _debug_failure(context: str, exc: BaseException) -> None:
    logger.debug("wildedge: mlx %s failed: %s", context, exc)


def _is_mlx_module(obj: object) -> bool:
    for cls in type(obj).__mro__:
        if cls.__name__ == "Module" and "mlx" in cls.__module__:
            return True
    return False


def _extract_model_args(obj: object) -> tuple[str | None, str | None]:
    """Returns (model_type, quantization_str) from model.args. Never raises."""
    try:
        args = getattr(obj, "args", None)
        if args is None:
            return None, None
        model_type = getattr(args, "model_type", None) or None
        quant = getattr(args, "quantization", None)
        if quant is not None:
            bits = getattr(quant, "bits", None)
            q_str = f"q{int(bits)}" if bits else "quantized"
        else:
            q_str = _detect_quantization_from_layers(obj)
        return model_type, q_str
    except Exception as exc:
        _debug_failure("model args extraction", exc)
        return None, None


def _detect_quantization_from_layers(obj: object) -> str | None:
    """Inspect layer class names for quantized linear layers."""
    try:
        for _, module in obj.named_modules():  # type: ignore[union-attr]
            cls_name = type(module).__name__
            if "Quantized" in cls_name or "quantized" in cls_name:
                return "quantized"
    except Exception:
        pass
    return None


def _count_tokens(tokenizer: object, text: str) -> int | None:
    try:
        return len(tokenizer.encode(text))  # type: ignore[union-attr]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Direct __call__ patch (non-LM / manual-registration use case)
# ---------------------------------------------------------------------------


def _build_mlx_call_patch(original_call):  # type: ignore[no-untyped-def]
    def patched_call(self_inner, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Suppress during mlx_lm.generate's autoregressive token loop
        if getattr(_inside_mlx_generate, "active", False):
            return original_call(self_inner, *args, **kwargs)

        handle = getattr(self_inner, MLX_HANDLE_ATTR, None)
        if handle is None:
            return original_call(self_inner, *args, **kwargs)

        t0 = time.perf_counter()
        try:
            result = original_call(self_inner, *args, **kwargs)
            handle.track_inference(duration_ms=elapsed_ms(t0), success=True)
            return result
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


class MlxExtractor(BaseExtractor):
    def can_handle(self, obj: object) -> bool:
        return _is_mlx_module(obj)

    def extract_info(
        self, obj: object, overrides: dict
    ) -> tuple[str | None, ModelInfo]:
        model_type, quantization = _extract_model_args(obj)

        model_name = model_type or type(obj).__name__
        model_id = overrides.pop("id", None) or model_name
        family = overrides.pop("family", None) or model_type
        version = overrides.pop("version", "unknown")
        source = overrides.pop("source", "huggingface")
        quantization = overrides.pop("quantization", None) or quantization

        info = ModelInfo(
            model_name=model_name,
            model_version=version,
            model_source=source,
            model_format="mlx",
            model_family=family,
            quantization=quantization,
        )
        for k, v in overrides.items():
            if hasattr(info, k):
                setattr(info, k, v)

        return model_id, info

    def memory_bytes(self, obj: object) -> int | None:
        if _tree_flatten is None:
            return None
        try:
            return sum(
                v.nbytes
                for _, v in _tree_flatten(obj.parameters())  # type: ignore[union-attr]
                if hasattr(v, "nbytes")
            )
        except Exception as exc:
            _debug_failure("memory estimation", exc)
            return None

    def install_hooks(self, obj: object, handle: ModelHandle) -> None:
        setattr(obj, MLX_HANDLE_ATTR, handle)
        patch_instance_call_once(
            obj,
            patch_name=MLX_CALL_PATCH_NAME,
            make_patched_call=_build_mlx_call_patch,
        )

    # -----------------------------------------------------------------------
    # Auto-load patches
    # -----------------------------------------------------------------------

    @classmethod
    def install_auto_load_patch(cls, client_ref: object) -> None:
        """Patch mlx_lm.load and mlx_lm.generate for automatic tracking.

        Called once at WildEdge client initialisation.

        - ``mlx_lm.load(path_or_repo)`` is timed; model ID is captured from
          the path argument; HuggingFace Hub downloads are recorded.
        - ``mlx_lm.generate(model, tokenizer, prompt, ...)`` is patched to
          emit a single inference event per call with token counts and
          tokens/second. The autoregressive ``model()`` loop inside generate
          is suppressed via a thread-local guard so it does not double-count.
        """
        global _mlx_patched
        if _mlx_patched or _mlx_lm is None:
            return

        with _MLX_PATCH_LOCK:
            if _mlx_patched:
                return
            cls._patch_load(client_ref)
            cls._patch_generate(client_ref)
            _mlx_patched = True

    @classmethod
    def _patch_load(cls, client_ref: object) -> None:
        original_load = _mlx_lm.load
        if (
            getattr(original_load, "__wildedge_patch_name__", None)
            == MLX_AUTO_LOAD_PATCH_NAME
        ):
            return

        def patched_load(path_or_hf_repo, *args, **kwargs):  # type: ignore[no-untyped-def]
            c = client_ref()  # type: ignore[call-arg]
            hub_before = (
                c._snapshot_hub_caches() if c is not None and not c.closed else {}
            )
            t0 = time.perf_counter()
            result = original_load(path_or_hf_repo, *args, **kwargs)
            load_ms = elapsed_ms(t0)

            # mlx_lm.load returns (model, tokenizer)
            model = result[0] if isinstance(result, tuple) else result

            if c is not None and not c.closed:
                downloads = c._diff_hub_caches(hub_before, load_ms) or None
                model_id = str(path_or_hf_repo) if path_or_hf_repo else None
                c._on_model_auto_loaded(
                    model,
                    load_ms=load_ms,
                    downloads=downloads,
                    model_id=model_id,
                )

            return result

        patched_load.__wildedge_patch_name__ = MLX_AUTO_LOAD_PATCH_NAME  # type: ignore[attr-defined]
        patched_load.__wildedge_original_call__ = original_load  # type: ignore[attr-defined]
        _mlx_lm.load = patched_load

    @classmethod
    def _patch_generate(cls, client_ref: object) -> None:  # noqa: ARG003
        original_generate = _mlx_lm.generate
        if (
            getattr(original_generate, "__wildedge_patch_name__", None)
            == MLX_GENERATE_PATCH_NAME
        ):
            return

        def patched_generate(model, tokenizer, prompt, *args, **kwargs):  # type: ignore[no-untyped-def]
            handle: ModelHandle | None = getattr(model, MLX_HANDLE_ATTR, None)

            tokens_in = _count_tokens(tokenizer, prompt) if tokenizer else None
            input_meta = TextInputMeta(token_count=tokens_in) if tokens_in else None

            _inside_mlx_generate.active = True
            t0 = time.perf_counter()
            try:
                result = original_generate(model, tokenizer, prompt, *args, **kwargs)
                duration_ms = elapsed_ms(t0)
            except Exception as exc:
                _inside_mlx_generate.active = False
                if handle is not None:
                    handle.track_error(
                        error_code="UNKNOWN",
                        error_message=str(exc)[: constants.ERROR_MSG_MAX_LEN],
                    )
                raise
            finally:
                _inside_mlx_generate.active = False

            if handle is not None:
                output_text = (
                    result.text
                    if hasattr(result, "text")
                    else (result if isinstance(result, str) else None)
                )
                tokens_out: int | None = None
                tps: float | None = None
                if output_text and tokenizer:
                    tokens_out = _count_tokens(tokenizer, output_text)
                    if tokens_out and duration_ms > 0:
                        tps = round(tokens_out / (duration_ms / 1000), 1)

                handle.track_inference(
                    duration_ms=duration_ms,
                    batch_size=1,
                    input_modality="text",
                    output_modality="generation",
                    input_meta=input_meta,
                    output_meta=GenerationOutputMeta(
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        tokens_per_second=tps,
                    ),
                    success=True,
                )

            return result

        patched_generate.__wildedge_patch_name__ = MLX_GENERATE_PATCH_NAME  # type: ignore[attr-defined]
        patched_generate.__wildedge_original_call__ = original_generate  # type: ignore[attr-defined]
        _mlx_lm.generate = patched_generate
