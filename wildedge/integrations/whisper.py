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
from wildedge.integrations.common import (
    debug_failure,
    hf_repo_from_cache_path,
    stream_error_code,
)
from wildedge.modalities import Modality
from wildedge.model import ModelHandle, ModelInfo

# Backends, used in model IDs and as the integration names.
FASTER_WHISPER = "faster_whisper"
MLX_WHISPER = "mlx_whisper"
MODEL_FORMATS = {FASTER_WHISPER: "ctranslate2", MLX_WHISPER: "mlx"}

SOURCE_LOCAL = "local"
SOURCE_HUGGINGFACE = "huggingface"

DESCRIPTOR_ATTR = "__wildedge_whisper_descriptor__"
HANDLE_ATTR = "__wildedge_whisper_handle__"
CLIENT_ATTR = "__wildedge_whisper_client__"

# Both libraries expect 1-D sample arrays at this rate.
SAMPLE_RATE = 16000

# Upper bound for each captured audio or transcript attachment. The attachment
# manager may apply a stricter limit of its own.
MAX_CAPTURE_BYTES = 10 * 1024 * 1024

# Captured arrays are encoded as mono 16-bit PCM WAV.
PCM16_SAMPLE_BYTES = 2
PCM16_MAX = 32767
WAV_HEADER_BYTES = 44

# Languages written without spaces between words, where splitting on
# whitespace does not give a word count.
UNSPACED_LANGUAGES = frozenset({"bo", "ja", "km", "lo", "my", "th", "yue", "zh"})


@dataclass
class WhisperDescriptor:
    """Model identity, worked out once when the model is loaded."""

    name: str
    source: str
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

    @property
    def model_format(self) -> str:
        return MODEL_FORMATS[self.backend]


def model_name_and_source(path: str) -> tuple[str, str]:
    """Name a model by its hub repo, its local folder, or the ID as given.

    A Hugging Face cache path (``.../models--org--name/snapshots/<sha>``) is
    named ``org/name`` rather than after the snapshot hash.
    """
    repo = hf_repo_from_cache_path(path)
    if repo is not None:
        return repo, SOURCE_HUGGINGFACE
    local = Path(path).expanduser()
    if local.exists():
        return local.resolve().name, SOURCE_LOCAL
    return path, SOURCE_HUGGINGFACE


def descriptor(obj: object) -> WhisperDescriptor | None:
    return (
        obj
        if isinstance(obj, WhisperDescriptor)
        else getattr(obj, DESCRIPTOR_ATTR, None)
    )


def report_load(client, model: object, load_ms: int) -> None:
    """Record a model load. Whisper loads always read the weights again, so a
    model ID seen before is still a new load, for example when MLX switches
    back to an earlier model."""
    client._on_model_auto_loaded(model, load_ms=load_ms, reload=True)


def report_cached_load(client, model: object) -> None:
    """Record a model that was already in memory before this client existed."""
    client._on_model_auto_loaded(
        model, load_ms=0, load_kwargs={"context": {"model_cache_hit": True}}
    )


class WhisperExtractor(BaseExtractor):
    """Must come before MlxExtractor in ``EXTRACTOR_SPECS``, which
    would otherwise claim MLX Whisper models as plain MLX models."""

    def can_handle(self, obj: object) -> bool:
        return isinstance(descriptor(obj), WhisperDescriptor)

    def extract_info(self, obj: object, overrides: dict) -> tuple[str, ModelInfo]:
        info = descriptor(obj)
        assert info is not None
        return overrides.pop("id", None) or info.model_id, ModelInfo(
            model_name=info.name,
            model_version=overrides.pop("version", "unknown"),
            model_source=overrides.pop("source", info.source),
            model_format=overrides.pop("format", info.model_format),
            model_family=overrides.pop("family", "whisper"),
            quantization=overrides.pop("quantization", None) or info.quantization,
        )

    def install_hooks(self, obj: object, handle: ModelHandle) -> None:
        info = descriptor(obj)
        assert info is not None
        handle.detected_accelerator = info.accelerator


def is_sample_array(audio: object) -> bool:
    return getattr(audio, "ndim", None) == 1


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
    elif is_sample_array(audio):
        meta.duration_ms = round(audio.shape[0] / SAMPLE_RATE * 1000)
        meta.sample_rate = SAMPLE_RATE
        meta.channels = 1
    if audio_duration_s is not None:
        meta.duration_ms = round(audio_duration_s * 1000)
    return meta


def pcm16_wav_size(sample_count: int) -> int:
    return WAV_HEADER_BYTES + sample_count * PCM16_SAMPLE_BYTES


def pcm16_wav(samples: object) -> bytes:
    """Encode float samples in [-1, 1] as a mono 16-bit WAV file."""
    import numpy as np  # noqa: PLC0415 -- optional; provided by the active Whisper runtime

    pcm = (np.clip(np.asarray(samples), -1, 1) * PCM16_MAX).astype("<i2")
    data = io.BytesIO()
    with wave.open(data, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(PCM16_SAMPLE_BYTES)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    return data.getvalue()


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
                    modality=Modality.AUDIO,
                )
            ]
    elif is_sample_array(audio) and pcm16_wav_size(audio.shape[0]) <= MAX_CAPTURE_BYTES:
        return [
            Attachment(
                content_type="audio/wav",
                role="input",
                data=pcm16_wav(audio),
                name="input.wav",
            )
        ]
    return []


def transcription_attachments(
    handle: ModelHandle, audio: object, text: str
) -> list[Attachment]:
    """The input audio and the transcript, each if the client allows it."""
    attachments = []
    if handle.wants_attachment(Modality.AUDIO):
        attachments.extend(audio_attachments(audio))
    encoded = text.encode()
    if (
        encoded
        and len(encoded) <= MAX_CAPTURE_BYTES
        and handle.wants_attachment(Modality.TEXT)
    ):
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
            attachments = transcription_attachments(handle, audio, text)
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
        error_code=stream_error_code(outcome),
    )
