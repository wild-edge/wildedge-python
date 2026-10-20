"""Opt-in instrumentation for faster-whisper's lazy transcription API."""

from __future__ import annotations

import functools
import inspect
import threading
import time

from wildedge.integrations.common import (
    SyncStreamWrapper,
    debug_failure,
    track_api_error,
)
from wildedge.integrations.whisper import (
    DESCRIPTOR_ATTR,
    HANDLE_ATTR,
    WhisperDescriptor,
    track_transcription,
)
from wildedge.timing import elapsed_ms


def wrap_transcribe(original):
    @functools.wraps(original)
    def transcribe(self, audio, *args, **kwargs):
        handle = getattr(self, HANDLE_ATTR, None)
        if handle is None:
            return original(self, audio, *args, **kwargs)
        t0 = time.perf_counter()
        try:
            segments, info = original(self, audio, *args, **kwargs)
        except Exception as exc:
            try:
                track_api_error(handle, exc, elapsed_ms(t0))
            except Exception as telemetry_exc:
                debug_failure("faster_whisper", "error tracking", telemetry_exc)
            raise
        words = 0
        parts = []
        captured_bytes = 0

        def chunk(segment):
            nonlocal words, captured_bytes
            text = segment.text
            words += len(text.split())
            if (
                handle.capture_attachments is not None
                and captured_bytes <= 10 * 1024 * 1024
            ):
                captured_bytes += len(text.encode())
                if captured_bytes <= 10 * 1024 * 1024:
                    parts.append(text)
                else:
                    parts.clear()

        def done(duration_ms, _ttft_ms, *, outcome="completed"):
            track_transcription(
                handle,
                audio,
                "".join(parts),
                getattr(info, "language", None),
                duration_ms,
                probability=getattr(info, "language_probability", None),
                duration=getattr(info, "duration", None),
                words=words,
                outcome=outcome,
            )

        return SyncStreamWrapper(segments, handle, t0, chunk, done), info

    return transcribe


_PATCH_LOCK = threading.Lock()


def install_auto_load_patch(client_ref):
    with _PATCH_LOCK:
        _install(client_ref)


def _install(client_ref):
    import faster_whisper  # noqa: PLC0415 -- optional runtime

    cls = faster_whisper.WhisperModel
    original = cls.__init__
    if getattr(original, "__wildedge_patch_name__", None) == "faster_whisper":
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
            path = bound.arguments["model_size_or_path"]
            runtime = self.model
            device = getattr(runtime, "device", bound.arguments.get("device", "cpu"))
            compute = getattr(
                runtime, "compute_type", bound.arguments.get("compute_type")
            )
            setattr(
                self,
                DESCRIPTOR_ATTR,
                WhisperDescriptor(
                    str(path),
                    "faster_whisper",
                    "gpu" if device == "cuda" else "cpu",
                    compute,
                ),
            )
            client._on_model_auto_loaded(self, load_ms=elapsed_ms(t0))
            setattr(self, HANDLE_ATTR, client.register_model(self))
        except Exception as exc:
            debug_failure("faster_whisper", "model registration", exc)

    init.__wildedge_patch_name__ = "faster_whisper"
    cls.__init__ = init
    cls.transcribe = wrap_transcribe(cls.transcribe)
