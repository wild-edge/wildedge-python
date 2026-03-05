from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from wildedge.events import (
    AudioInputMeta,
    ClassificationOutputMeta,
    DetectionOutputMeta,
    EmbeddingOutputMeta,
    ErrorCode,
    ErrorEvent,
    FeedbackEvent,
    FeedbackType,
    GenerationConfig,
    GenerationOutputMeta,
    ImageInputMeta,
    InferenceEvent,
    ModelDownloadEvent,
    ModelLoadEvent,
    ModelUnloadEvent,
    TextInputMeta,
)
from wildedge.logging import logger


@dataclass
class ModelInfo:
    model_name: str
    model_version: str
    model_source: str
    model_format: str
    model_family: str | None = None
    quantization: str | None = None

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_source": self.model_source,
            "model_format": self.model_format,
            "model_family": self.model_family,
            "quantization": self.quantization,
        }


class ModelHandle:
    """Handle returned by register_model(). All events for a model go through here."""

    def __init__(self, model_id: str, info: ModelInfo, publish: Callable[[dict], None]):
        self.model_id = model_id
        self.info = info
        self.publish = publish
        self.detected_accelerator: str = "cpu"
        self.last_inference_id: str | None = None

    def track_load(
        self,
        duration_ms: int,
        *,
        memory_bytes: int | None = None,
        accelerator: str | None = None,
        success: bool = True,
        error_code: str | None = None,
        **kwargs: Any,
    ) -> None:
        event = ModelLoadEvent(
            model_id=self.model_id,
            duration_ms=duration_ms,
            memory_bytes=memory_bytes,
            accelerator=accelerator or self.detected_accelerator,
            success=success,
            error_code=error_code,
            **kwargs,
        )
        self.publish(event.to_dict())

    def track_unload(
        self,
        duration_ms: int,
        reason: str,
        *,
        memory_freed_bytes: int | None = None,
        peak_memory_bytes: int | None = None,
        uptime_ms: int | None = None,
    ) -> None:
        event = ModelUnloadEvent(
            model_id=self.model_id,
            duration_ms=duration_ms,
            reason=reason,
            memory_freed_bytes=memory_freed_bytes,
            peak_memory_bytes=peak_memory_bytes,
            uptime_ms=uptime_ms,
        )
        self.publish(event.to_dict())

    def track_download(
        self,
        source_url: str,
        source_type: str,
        file_size_bytes: int,
        downloaded_bytes: int,
        duration_ms: int,
        network_type: str,
        resumed: bool,
        cache_hit: bool,
        success: bool,
        **kwargs: Any,
    ) -> None:
        event = ModelDownloadEvent(
            model_id=self.model_id,
            source_url=source_url,
            source_type=source_type,
            file_size_bytes=file_size_bytes,
            downloaded_bytes=downloaded_bytes,
            duration_ms=duration_ms,
            network_type=network_type,
            resumed=resumed,
            cache_hit=cache_hit,
            success=success,
            **kwargs,
        )
        self.publish(event.to_dict())

    def track_inference(
        self,
        duration_ms: int,
        *,
        input_modality: str | None = None,
        output_modality: str | None = None,
        batch_size: int | None = None,
        success: bool = True,
        error_code: str | None = None,
        input_meta: ImageInputMeta | AudioInputMeta | TextInputMeta | None = None,
        output_meta: DetectionOutputMeta
        | ClassificationOutputMeta
        | GenerationOutputMeta
        | EmbeddingOutputMeta
        | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> str:
        event = InferenceEvent(
            model_id=self.model_id,
            duration_ms=duration_ms,
            input_modality=input_modality,
            output_modality=output_modality,
            batch_size=batch_size,
            success=success,
            error_code=error_code,
            input_meta=input_meta,
            output_meta=output_meta,
            generation_config=generation_config,
        )
        self.last_inference_id = event.inference_id
        self.publish(event.to_dict())
        return event.inference_id

    def track_feedback(
        self,
        related_inference_id: str,
        feedback_type: str | FeedbackType,
        *,
        delay_ms: int | None = None,
        edit_distance: int | None = None,
    ) -> None:
        event = FeedbackEvent(
            model_id=self.model_id,
            related_inference_id=related_inference_id,
            feedback_type=feedback_type,
            delay_ms=delay_ms,
            edit_distance=edit_distance,
        )
        self.publish(event.to_dict())

    def feedback(
        self,
        feedback_type: str | FeedbackType,
        *,
        delay_ms: int | None = None,
    ) -> None:
        """Emit feedback linked to the most recent inference on this handle."""
        if self.last_inference_id is None:
            logger.warning(
                "wildedge: feedback() called before any inference was tracked"
            )
            return
        self.track_feedback(self.last_inference_id, feedback_type, delay_ms=delay_ms)

    def track_error(
        self,
        error_code: str | ErrorCode,
        *,
        error_message: str | None = None,
        stack_trace_hash: str | None = None,
        related_event_id: str | None = None,
    ) -> None:
        event = ErrorEvent(
            model_id=self.model_id,
            error_code=error_code,
            error_message=error_message,
            stack_trace_hash=stack_trace_hash,
            related_event_id=related_event_id,
        )
        self.publish(event.to_dict())


class ModelRegistry:
    """Thread-safe registry mapping model_id to ModelInfo."""

    def __init__(self) -> None:
        self.models: dict[str, ModelInfo] = {}
        self.handles: dict[str, ModelHandle] = {}

    def register(
        self, model_id: str, info: ModelInfo, publish: Callable[[dict], None]
    ) -> tuple[ModelHandle, bool]:
        """Return (handle, is_new). is_new=False means already registered; skip install_hooks."""
        if model_id in self.handles:
            return self.handles[model_id], False
        handle = ModelHandle(model_id=model_id, info=info, publish=publish)
        self.models[model_id] = info
        self.handles[model_id] = handle
        return handle, True

    def snapshot(self) -> dict[str, dict]:
        return {mid: info.to_dict() for mid, info in self.models.items()}
