from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class HistogramSummary:
    brightness_mean: float | None = None
    brightness_stddev: float | None = None
    brightness_buckets: list[int] | None = None
    contrast: float | None = None
    saturation_mean: float | None = None
    blur_score: float | None = None
    noise_score: float | None = None

    def to_dict(self) -> dict:
        return {
            k: v
            for k, v in {
                "brightness_mean": self.brightness_mean,
                "brightness_stddev": self.brightness_stddev,
                "brightness_buckets": self.brightness_buckets,
                "contrast": self.contrast,
                "saturation_mean": self.saturation_mean,
                "blur_score": self.blur_score,
                "noise_score": self.noise_score,
            }.items()
            if v is not None
        }


@dataclass
class ImageInputMeta:
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    format: str | None = None
    source: str | None = None
    histogram_summary: HistogramSummary | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {}
        for k, v in [
            ("width", self.width),
            ("height", self.height),
            ("channels", self.channels),
            ("format", self.format),
            ("source", self.source),
        ]:
            if v is not None:
                d[k] = v
        if self.histogram_summary is not None:
            d["histogram_summary"] = self.histogram_summary.to_dict()
        return d


@dataclass
class AudioInputMeta:
    duration_ms: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bit_depth: int | None = None
    format: str | None = None
    codec: str | None = None
    source: str | None = None
    is_streaming: bool | None = None
    snr_db: float | None = None
    volume_db: float | None = None
    speech_ratio: float | None = None
    clipping_detected: bool | None = None

    def to_dict(self) -> dict:
        return {
            k: v
            for k, v in {
                "duration_ms": self.duration_ms,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "bit_depth": self.bit_depth,
                "format": self.format,
                "codec": self.codec,
                "source": self.source,
                "is_streaming": self.is_streaming,
                "snr_db": self.snr_db,
                "volume_db": self.volume_db,
                "speech_ratio": self.speech_ratio,
                "clipping_detected": self.clipping_detected,
            }.items()
            if v is not None
        }


@dataclass
class TextInputMeta:
    char_count: int | None = None
    word_count: int | None = None
    token_count: int | None = None
    language: str | None = None
    language_confidence: float | None = None
    encoding: str | None = None
    contains_code: bool | None = None
    prompt_type: str | None = None
    turn_index: int | None = None
    has_attachments: bool | None = None

    def to_dict(self) -> dict:
        return {
            k: v
            for k, v in {
                "char_count": self.char_count,
                "word_count": self.word_count,
                "token_count": self.token_count,
                "language": self.language,
                "language_confidence": self.language_confidence,
                "encoding": self.encoding,
                "contains_code": self.contains_code,
                "prompt_type": self.prompt_type,
                "turn_index": self.turn_index,
                "has_attachments": self.has_attachments,
            }.items()
            if v is not None
        }


@dataclass
class TopKPrediction:
    label: str
    confidence: float | None = None
    bbox: list[int] | None = None
    coverage_ratio: float | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"label": self.label}
        if self.confidence is not None:
            d["confidence"] = self.confidence
        if self.bbox is not None:
            d["bbox"] = self.bbox
        if self.coverage_ratio is not None:
            d["coverage_ratio"] = self.coverage_ratio
        return d


@dataclass
class DetectionOutputMeta:
    task: str = "detection"
    num_predictions: int | None = None
    top_k: list[TopKPrediction] | None = None
    avg_confidence: float | None = None
    mask_width: int | None = None
    mask_height: int | None = None
    num_classes: int | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"task": self.task}
        if self.num_predictions is not None:
            d["num_predictions"] = self.num_predictions
        if self.top_k is not None:
            d["top_k"] = [p.to_dict() for p in self.top_k]
        for k, v in [
            ("avg_confidence", self.avg_confidence),
            ("mask_width", self.mask_width),
            ("mask_height", self.mask_height),
            ("num_classes", self.num_classes),
        ]:
            if v is not None:
                d[k] = v
        return d


@dataclass
class ClassificationOutputMeta:
    task: str = "classification"
    num_predictions: int | None = None
    top_k: list[TopKPrediction] | None = None
    avg_confidence: float | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"task": self.task}
        if self.num_predictions is not None:
            d["num_predictions"] = self.num_predictions
        if self.top_k is not None:
            d["top_k"] = [p.to_dict() for p in self.top_k]
        if self.avg_confidence is not None:
            d["avg_confidence"] = self.avg_confidence
        return d


@dataclass
class GenerationConfig:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop_sequences_count: int | None = None

    def to_dict(self) -> dict:
        return {
            k: v
            for k, v in {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_tokens": self.max_tokens,
                "repetition_penalty": self.repetition_penalty,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "seed": self.seed,
                "stop_sequences_count": self.stop_sequences_count,
            }.items()
            if v is not None
        }


@dataclass
class GenerationOutputMeta:
    task: str = "generation"
    tokens_in: int | None = None
    tokens_out: int | None = None
    time_to_first_token_ms: int | None = None
    tokens_per_second: float | None = None
    stop_reason: str | None = None
    context_used: int | None = None
    avg_token_entropy: float | None = None
    safety_triggered: bool | None = None

    def to_dict(self) -> dict:
        return {
            k: v
            for k, v in {
                "task": self.task,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "time_to_first_token_ms": self.time_to_first_token_ms,
                "tokens_per_second": self.tokens_per_second,
                "stop_reason": self.stop_reason,
                "context_used": self.context_used,
                "avg_token_entropy": self.avg_token_entropy,
                "safety_triggered": self.safety_triggered,
            }.items()
            if v is not None
        }


@dataclass
class EmbeddingOutputMeta:
    task: str = "embedding"
    dimensions: int | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"task": self.task}
        if self.dimensions is not None:
            d["dimensions"] = self.dimensions
        return d


@dataclass
class InferenceEvent:
    model_id: str
    duration_ms: int
    input_modality: str | None = None
    output_modality: str | None = None
    success: bool = True
    error_code: str | None = None
    batch_size: int | None = None
    input_meta: ImageInputMeta | AudioInputMeta | TextInputMeta | None = None
    output_meta: (
        DetectionOutputMeta
        | ClassificationOutputMeta
        | GenerationOutputMeta
        | EmbeddingOutputMeta
        | None
    ) = None
    generation_config: GenerationConfig | None = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    inference_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        inference_data: dict[str, Any] = {
            "inference_id": self.inference_id,
            "duration_ms": self.duration_ms,
            "success": self.success,
        }
        if self.input_modality is not None:
            inference_data["input_modality"] = self.input_modality
        if self.output_modality is not None:
            inference_data["output_modality"] = self.output_modality
        if self.batch_size is not None:
            inference_data["batch_size"] = self.batch_size
        if self.error_code is not None:
            inference_data["error_code"] = self.error_code
        if self.input_meta is not None:
            inference_data["input_meta"] = self.input_meta.to_dict()
        if self.output_meta is not None:
            inference_data["output_meta"] = self.output_meta.to_dict()
        if self.generation_config is not None:
            inference_data["generation_config"] = self.generation_config.to_dict()

        return {
            "event_id": self.event_id,
            "event_type": "inference",
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "inference": inference_data,
        }
