"""Shared metadata and opt-in capture for local Whisper runtimes."""

from __future__ import annotations

import io
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
CLIENT_ATTR = "__wildedge_whisper_client__"
SAMPLE_RATE = 16000

# Upper bound for each captured audio or transcript attachment. The attachment
# manager may apply a stricter limit of its own.
MAX_CAPTURE_BYTES = 10 * 1024 * 1024

# Languages written without spaces between words, where splitting on
# whitespace does not give a word count.
UNSPACED_LANGUAGES = frozenset({"bo", "ja", "km", "lo", "my", "th", "yue", "zh"})


@dataclass
class WhisperDescriptor:
    """Model identity, worked out once when the model is loaded."""

    name: str
    source: str  # "local" or "huggingface"
    backend: str
    accelerator: str
    quantization: str | None = None

    @classmethod
    def from_path(
        cls,
        path: object,
        backend: str,
        accelerator: str,
        quantization: str | None = None,
    ) -> WhisperDescriptor:
        name, source = model_name_and_source(str(path))
        return cls(name, source, backend, accelerator, quantization)

    @property
    def model_id(self) -> str:
        return f"{self.backend}:{self.name}"


def model_name_and_source(path: str) -> tuple[str, str]:
    """Name a model by its hub repo, its local folder, or the ID as given.

    A Hugging Face cache path (``.../models--org--name/snapshots/<sha>``) is
    named ``org/name`` rather than after the snapshot hash.
    """
    for part in path.replace("\\", "/").split("/"):
        if part.startswith("models--"):
            return part[len("models--") :].replace("--", "/", 1), "huggingface"
    local = Path(path).expanduser()
    if local.exists():
        return local.resolve().name, "local"
    return path, "huggingface"


def descriptor(obj: object) -> WhisperDescriptor | None:
    return (
        obj
        if isinstance(obj, WhisperDescriptor)
        else getattr(obj, DESCRIPTOR_ATTR, None)
    )


def report_load(client, model: object, load_ms: int, **kwargs) -> None:
    """Record a model load, including a reload of a model ID seen before.

    The client skips the load event for an ID it already knows, which suits
    wrappers around an already loaded model. Whisper loads always read the
    weights again, for example when MLX switches back to an earlier model.
    """
    info = descriptor(model)
    if info is not None:
        client.auto_loaded.discard(info.model_id)
    client._on_model_auto_loaded(model, load_ms=load_ms, **kwargs)


class WhisperExtractor(BaseExtractor):
    def can_handle(self, obj: object) -> bool:
        return isinstance(descriptor(obj), WhisperDescriptor)

    def extract_info(self, obj: object, overrides: dict) -> tuple[str, ModelInfo]:
        info = descriptor(obj)
        assert info is not None
        return overrides.pop("id", None) or info.model_id, ModelInfo(
            model_name=info.name,
            model_version=overrides.pop("version", "unknown"),
            model_source=overrides.pop("source", info.source),
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


def audio_meta(audio: object, audio_duration_s: float | None = None) -> AudioInputMeta:
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
    if audio_duration_s is not None:
        meta.duration_ms = round(audio_duration_s * 1000)
    return meta


def audio_attachments(audio: object) -> list[Attachment]:
    if isinstance(audio, (str, Path)):
        path = Path(audio)
        if path.is_file() and path.stat().st_size <= MAX_CAPTURE_BYTES:
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
    elif (
        getattr(audio, "ndim", None) == 1
        and audio.shape[0] * 2 + 44 <= MAX_CAPTURE_BYTES
    ):
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


def capture(audio: object, text: str) -> list[Attachment]:
    attachments = audio_attachments(audio)
    encoded = text.encode()
    if encoded and len(encoded) <= MAX_CAPTURE_BYTES:
        attachments.append(
            Attachment(
                content_type="text/plain",
                role="output",
                data=encoded,
                name="transcript.txt",
            )
        )
    return attachments


def track_transcription(
    handle: ModelHandle,
    audio: object,
    text: str,
    *,
    duration_ms: int,
    language: str | None = None,
    language_probability: float | None = None,
    audio_duration_s: float | None = None,
    word_count: int | None = None,
    first_segment_ms: int | None = None,
    outcome: str = "completed",
) -> None:
    """Record one transcription.

    ``duration_ms`` is how long the transcription took; ``audio_duration_s``
    is the length of the audio when the runtime reports it.
    """
    context: dict = {}
    if language not in UNSPACED_LANGUAGES:
        context["transcript_word_count"] = (
            len(text.split()) if word_count is None else word_count
        )
    if language is not None:
        context["language"] = language
    if language_probability is not None:
        context["language_confidence"] = language_probability
    if first_segment_ms is not None:
        context["time_to_first_segment_ms"] = first_segment_ms
    attachments = None
    if handle.capture_attachments is not None:
        try:
            attachments = capture(audio, text)
        except Exception as exc:
            debug_failure("whisper", "attachment capture", exc)
    handle.track_inference(
        duration_ms=duration_ms,
        input_modality="audio",
        output_modality="text",
        input_meta=audio_meta(audio, audio_duration_s),
        context=context,
        attachments=attachments,
        success=outcome == "completed",
        error_code=None if outcome == "completed" else "STREAM_" + outcome.upper(),
    )
