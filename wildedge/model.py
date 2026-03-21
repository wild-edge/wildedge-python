from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wildedge.events import (
    ApiMeta,
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
from wildedge.platforms import capture_hardware, is_sampling
from wildedge.platforms.hardware import HardwareContext
from wildedge.tracing import merge_correlation_fields


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
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        step_index: int | None = None,
        conversation_id: str | None = None,
        attributes: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        correlation = merge_correlation_fields(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            run_id=run_id,
            agent_id=agent_id,
            step_index=step_index,
            conversation_id=conversation_id,
            attributes=attributes,
        )
        event = ModelLoadEvent(
            model_id=self.model_id,
            duration_ms=duration_ms,
            memory_bytes=memory_bytes,
            accelerator=accelerator or self.detected_accelerator,
            success=success,
            error_code=error_code,
            **correlation,
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
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        step_index: int | None = None,
        conversation_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        correlation = merge_correlation_fields(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            run_id=run_id,
            agent_id=agent_id,
            step_index=step_index,
            conversation_id=conversation_id,
            attributes=attributes,
        )
        event = ModelUnloadEvent(
            model_id=self.model_id,
            duration_ms=duration_ms,
            reason=reason,
            memory_freed_bytes=memory_freed_bytes,
            peak_memory_bytes=peak_memory_bytes,
            uptime_ms=uptime_ms,
            **correlation,
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
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        step_index: int | None = None,
        conversation_id: str | None = None,
        attributes: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        correlation = merge_correlation_fields(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            run_id=run_id,
            agent_id=agent_id,
            step_index=step_index,
            conversation_id=conversation_id,
            attributes=attributes,
        )
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
            **correlation,
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
        hardware: HardwareContext | None = None,
        api_meta: ApiMeta | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        step_index: int | None = None,
        conversation_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        if hardware is None and is_sampling():
            hardware = capture_hardware()
        correlation = merge_correlation_fields(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            run_id=run_id,
            agent_id=agent_id,
            step_index=step_index,
            conversation_id=conversation_id,
            attributes=attributes,
        )
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
            hardware=hardware,
            api_meta=api_meta,
            **correlation,
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
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        step_index: int | None = None,
        conversation_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        correlation = merge_correlation_fields(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            run_id=run_id,
            agent_id=agent_id,
            step_index=step_index,
            conversation_id=conversation_id,
            attributes=attributes,
        )
        event = FeedbackEvent(
            model_id=self.model_id,
            related_inference_id=related_inference_id,
            feedback_type=feedback_type,
            delay_ms=delay_ms,
            edit_distance=edit_distance,
            **correlation,
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
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        step_index: int | None = None,
        conversation_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        correlation = merge_correlation_fields(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            run_id=run_id,
            agent_id=agent_id,
            step_index=step_index,
            conversation_id=conversation_id,
            attributes=attributes,
        )
        event = ErrorEvent(
            model_id=self.model_id,
            error_code=error_code,
            error_message=error_message,
            stack_trace_hash=stack_trace_hash,
            related_event_id=related_event_id,
            **correlation,
        )
        self.publish(event.to_dict())


class ModelRegistry:
    """Thread-safe registry mapping model_id to ModelInfo."""

    def __init__(self, persist_path: str | None = None) -> None:
        self.models: dict[str, ModelInfo] = {}
        self.handles: dict[str, ModelHandle] = {}
        self.persist_path = Path(persist_path).expanduser() if persist_path else None
        if self.persist_path is not None:
            self.load_from_disk()

    def load_from_disk(self) -> None:
        if self.persist_path is None or not self.persist_path.exists():
            return
        try:
            raw = json.loads(self.persist_path.read_text())
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        for model_id, data in raw.items():
            if not isinstance(model_id, str) or not isinstance(data, dict):
                continue
            try:
                self.models[model_id] = ModelInfo(
                    model_name=str(data["model_name"]),
                    model_version=str(data["model_version"]),
                    model_source=str(data["model_source"]),
                    model_format=str(data["model_format"]),
                    model_family=data.get("model_family"),
                    quantization=data.get("quantization"),
                )
            except KeyError:
                continue

    def save_to_disk(self) -> None:
        if self.persist_path is None:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.snapshot()
        self.persist_path.write_text(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        )

    def register(
        self, model_id: str, info: ModelInfo, publish: Callable[[dict], None]
    ) -> tuple[ModelHandle, bool]:
        """Return (handle, is_new). is_new=False means already registered; skip install_hooks."""
        if model_id in self.handles:
            return self.handles[model_id], False

        if model_id not in self.models:
            self.models[model_id] = info
            self.save_to_disk()

        handle = ModelHandle(
            model_id=model_id, info=self.models[model_id], publish=publish
        )
        self.handles[model_id] = handle
        return handle, True

    def snapshot(self) -> dict[str, dict]:
        return {mid: info.to_dict() for mid, info in self.models.items()}
