"""Shared metadata and opt-in capture for local Whisper runtimes."""

from __future__ import annotations

import io
import json
import mimetypes
import wave
from dataclasses import dataclass
from pathlib import Path

from wildedge.attachments import Attachment
from wildedge.events.inference import AudioInputMeta
from wildedge.integrations.base import BaseExtractor
from wildedge.integrations.common import debug_failure
from wildedge.model import ModelHandle, ModelInfo

DESCRIPTOR_ATTR = "__wildedge_whisper_descriptor__"
HANDLE_ATTR = "__wildedge_whisper_handle__"
SAMPLE_RATE = 16000


@dataclass
class WhisperDescriptor:
    path: str
    backend: str
    accelerator: str
    quantization: str | None = None

    @property
    def name(self) -> str:
        return Path(self.path).name if Path(self.path).exists() else self.path

    @property
    def model_id(self) -> str:
        return f"{self.backend}:{self.name}"


def descriptor(obj: object) -> WhisperDescriptor | None:
    return (
        obj
        if isinstance(obj, WhisperDescriptor)
        else getattr(obj, DESCRIPTOR_ATTR, None)
    )


class WhisperExtractor(BaseExtractor):
    def can_handle(self, obj: object) -> bool:
        return isinstance(descriptor(obj), WhisperDescriptor)

    def extract_info(self, obj: object, overrides: dict) -> tuple[str, ModelInfo]:
        info = descriptor(obj)
        assert info is not None
        return overrides.pop("id", None) or info.model_id, ModelInfo(
            model_name=info.name,
            model_version=overrides.pop("version", "unknown"),
            model_source=overrides.pop(
                "source", "local" if Path(info.path).exists() else "huggingface"
            ),
            model_format=overrides.pop(
                "format", "mlx" if info.backend == "mlx_whisper" else "ctranslate2"
            ),
            model_family=overrides.pop("family", "whisper"),
            quantization=overrides.pop("quantization", None) or info.quantization,
        )

    def install_hooks(self, obj: object, handle: ModelHandle) -> None:
        info = descriptor(obj)
        assert info is not None
        handle.detected_accelerator = info.accelerator


def mlx_descriptor(path: object, dtype: object = "float16") -> WhisperDescriptor:
    quantization = "fp16" if "16" in str(dtype) else "fp32"
    try:
        config = Path(str(path)) / "config.json"
        if config.is_file():
            quant = json.loads(config.read_text()).get("quantization") or {}
            if isinstance(quant.get("bits"), int):
                quantization = f"q{quant['bits']}"
    except Exception as exc:
        debug_failure("mlx_whisper", "quantization extraction", exc)
    return WhisperDescriptor(str(path), "mlx_whisper", "gpu", quantization)


def audio_meta(audio: object, duration: float | None = None) -> AudioInputMeta:
    meta = AudioInputMeta()
    if isinstance(audio, (str, Path)):
        try:
            with wave.open(str(audio), "rb") as wav:
                meta.duration_ms = round(wav.getnframes() / wav.getframerate() * 1000)
                meta.sample_rate = wav.getframerate()
                meta.channels = wav.getnchannels()
                meta.bit_depth = wav.getsampwidth() * 8
                meta.format = "wav"
        except (OSError, EOFError, wave.Error):
            pass
    elif getattr(audio, "ndim", None) == 1:
        meta.duration_ms = round(audio.shape[0] / SAMPLE_RATE * 1000)
        meta.sample_rate = SAMPLE_RATE
        meta.channels = 1
    if duration is not None:
        meta.duration_ms = round(duration * 1000)
    return meta


def audio_attachments(audio: object, limit: int) -> list[Attachment]:
    if isinstance(audio, (str, Path)):
        path = Path(audio)
        if path.is_file() and path.stat().st_size <= limit:
            content_type = (
                mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            )
            return [
                Attachment(
                    content_type=content_type,
                    role="input",
                    path=str(path),
                    name=path.name,
                )
            ]
    elif getattr(audio, "ndim", None) == 1 and audio.shape[0] * 2 + 44 <= limit:
        import numpy as np  # noqa: PLC0415 -- optional; provided by the active Whisper runtime

        pcm = (np.clip(np.asarray(audio), -1, 1) * 32767).astype("<i2")
        data = io.BytesIO()
        with wave.open(data, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm.tobytes())
        return [
            Attachment(
                content_type="audio/wav",
                role="input",
                data=data.getvalue(),
                name="input.wav",
            )
        ]
    return []


def track_transcription(
    handle: ModelHandle,
    audio: object,
    text: str,
    language: str | None,
    duration_ms: int,
    *,
    probability: float | None = None,
    duration: float | None = None,
    outcome: str = "completed",
    words: int | None = None,
) -> None:
    context = {"transcript_word_count": len(text.split()) if words is None else words}
    if language is not None:
        context["language"] = language
    if probability is not None:
        context["language_confidence"] = probability
    attachments = None
    if handle.capture_attachments is not None:
        try:
            # The manager enforces its own configured size/count limits as well.
            limit = 10 * 1024 * 1024
            attachments = audio_attachments(audio, limit)
            if text and len(text.encode()) <= limit:
                attachments.append(
                    Attachment(
                        content_type="text/plain",
                        role="output",
                        data=text.encode(),
                        name="transcript.txt",
                    )
                )
        except Exception as exc:
            debug_failure("whisper", "attachment capture", exc)
    handle.track_inference(
        duration_ms=duration_ms,
        input_modality="audio",
        output_modality="text",
        input_meta=audio_meta(audio, duration),
        context=context,
        attachments=attachments,
        success=outcome == "completed",
        error_code=None if outcome == "completed" else "STREAM_" + outcome.upper(),
    )
