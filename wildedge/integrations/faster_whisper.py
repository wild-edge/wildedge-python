"""Opt-in instrumentation for faster-whisper's lazy transcription API."""

from __future__ import annotations

import functools
import inspect
import threading
import time

from wildedge import accelerators, quantizations
from wildedge.instrumentation import bind_client_reference
from wildedge.integrations.common import (
    SyncStreamWrapper,
    debug_failure,
    track_api_error,
)
from wildedge.integrations.whisper import (
    CLIENT_ATTR,
    DESCRIPTOR_ATTR,
    FASTER_WHISPER,
    HANDLE_ATTR,
    MAX_CAPTURE_BYTES,
    WhisperDescriptor,
    report_load,
    track_transcription,
)
from wildedge.modalities import Modality
from wildedge.timing import elapsed_ms

_PATCH_LOCK = threading.Lock()

# WhisperModel.__init__'s first parameter: a size name, hub repo or local path.
MODEL_PATH_PARAM = "model_size_or_path"

# CTranslate2 compute types, labelled by weight precision. The activation type
# after "int8_" isn't reported. Unknown types pass through as they are.
COMPUTE_TYPE_QUANTIZATION = {
    "float32": quantizations.F32,
    "float16": quantizations.F16,
    "bfloat16": quantizations.BF16,
    "int8": quantizations.INT8,
    "int8_float32": quantizations.INT8,
    "int8_float16": quantizations.INT8,
    "int8_bfloat16": quantizations.INT8,
    "int16": quantizations.int_bits(16),
}

# Outcome of a segment loop that stopped without reading to the end.
OUTCOME_ABANDONED = "abandoned"


class TranscriptCollector:
    """Counts words as segments arrive and keeps the text when capture is on."""

    def __init__(self, t0: float, *, capture: bool) -> None:
        self._t0 = t0
        self.words = 0
        self.last_segment_ms: int | None = None
        self._parts: list[str] | None = [] if capture else None
        self._size = 0

    def add(self, segment) -> None:
        self.last_segment_ms = elapsed_ms(self._t0)
        text = segment.text
        self.words += len(text.split())
        if self._parts is None:
            return
        self._size += len(text.encode())
        if self._size > MAX_CAPTURE_BYTES:
            # A cut-off transcript is not worth uploading.
            self._parts = None
        else:
            self._parts.append(text)

    @property
    def text(self) -> str:
        return "" if self._parts is None else "".join(self._parts)


class SegmentStream(SyncStreamWrapper):
    """Segment iterator that also reports a loop that stopped early.

    ``for segment in segments: break`` drops the generator without closing
    it. The event is then sent when the iterator is garbage collected, as
    ``STREAM_ABANDONED`` with the time up to the last segment received. A
    stream that was never iterated, for example when only ``info.language``
    was needed, decoded nothing and sends no event.
    """

    # Class-level defaults keep __del__ safe if __init__ never finished;
    # otherwise these lookups would fall through to the wrapped generator.
    _finished = True
    _iterator = None

    def __del__(self) -> None:
        if self.started and not self.finished:
            self._finish(OUTCOME_ABANDONED)


def tracked_handle(model):
    """The model's handle while its SDK client is still open, else None."""
    client_ref = getattr(model, CLIENT_ATTR, None)
    if client_ref is None or client_ref() is None:
        return None
    return getattr(model, HANDLE_ATTR, None)


def wrap_transcribe(original, model_of=lambda owner: owner):
    """Wrap ``WhisperModel.transcribe`` or ``BatchedInferencePipeline.transcribe``.

    ``model_of`` finds the ``WhisperModel`` behind the object being called.
    """

    @functools.wraps(original)
    def transcribe(self, audio, *args, **kwargs):
        handle = tracked_handle(model_of(self))
        if handle is None:
            return original(self, audio, *args, **kwargs)
        t0 = time.perf_counter()
        try:
            segments, info = original(self, audio, *args, **kwargs)
        except Exception as exc:
            try:
                track_api_error(handle, exc, elapsed_ms(t0))
            except Exception as telemetry_exc:
                debug_failure(FASTER_WHISPER, "error tracking", telemetry_exc)
            raise
        collector = TranscriptCollector(
            t0, capture=handle.wants_attachment(Modality.TEXT)
        )

        def done(duration_ms, first_segment_ms, *, outcome="completed"):
            if outcome == OUTCOME_ABANDONED and collector.last_segment_ms is not None:
                duration_ms = collector.last_segment_ms
            track_transcription(
                handle,
                audio,
                collector.text,
                duration_ms=duration_ms,
                language=getattr(info, "language", None),
                language_probability=getattr(info, "language_probability", None),
                audio_duration_s=getattr(info, "duration", None),
                word_count=collector.words,
                first_segment_ms=first_segment_ms,
                outcome=outcome,
            )

        return SegmentStream(segments, handle, t0, collector.add, done), info

    return transcribe


def install_auto_load_patch(client_ref):
    with _PATCH_LOCK:
        _install(client_ref)


def _install(client_ref):
    import faster_whisper  # noqa: PLC0415 -- optional runtime

    cls = faster_whisper.WhisperModel
    original = cls.__init__
    if getattr(original, "__wildedge_patch_name__", None) == FASTER_WHISPER:
        return
    signature = inspect.signature(original)

    @functools.wraps(original)
    def init(self, *args, **kwargs):
        client = client_ref()
        t0 = time.perf_counter()
        original(self, *args, **kwargs)
        if client is None or client.closed:
            return
        try:
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            # The ctranslate2.models.Whisper instance. Its device is the
            # resolved one ("cpu" or "cuda"), already an accelerator name.
            runtime = self.model
            device = getattr(runtime, "device", None) or accelerators.CPU
            compute = getattr(
                runtime, "compute_type", bound.arguments.get("compute_type")
            )
            setattr(
                self,
                DESCRIPTOR_ATTR,
                WhisperDescriptor.from_path(
                    bound.arguments[MODEL_PATH_PARAM],
                    FASTER_WHISPER,
                    device,
                    COMPUTE_TYPE_QUANTIZATION.get(compute, compute),
                ),
            )
            report_load(client, self, elapsed_ms(t0))
            # Already registered by the load above; this returns its handle.
            setattr(self, HANDLE_ATTR, client.register_model(self))
            # Keep this model with the client that loaded it, as provider
            # clients do. After a client replacement it warns once instead.
            setattr(self, CLIENT_ATTR, bind_client_reference(client_ref, client))
        except Exception as exc:
            debug_failure(FASTER_WHISPER, "model registration", exc)

    init.__wildedge_patch_name__ = FASTER_WHISPER
    cls.__init__ = init
    cls.transcribe = wrap_transcribe(cls.transcribe)
    batched = getattr(faster_whisper, "BatchedInferencePipeline", None)
    if batched is not None:
        batched.transcribe = wrap_transcribe(
            batched.transcribe,
            model_of=lambda pipeline: getattr(pipeline, "model", None),
        )
