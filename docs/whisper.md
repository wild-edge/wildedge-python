# Local Whisper speech recognition

Enable `mlx_whisper` on Apple Silicon or `faster_whisper` for CTranslate2 before loading the model or importing a transcription function by name:

```python
import wildedge

client = wildedge.init(integrations=["mlx_whisper"], attachments_enabled=False)
import mlx_whisper

result = mlx_whisper.transcribe("speech.wav", path_or_hf_repo="/models/whisper")
```

For faster-whisper, consume the returned segments to finish the inference:

```python
client = wildedge.init(integrations=["faster_whisper"], attachments_enabled=False)
from faster_whisper import WhisperModel

model = WhisperModel("/models/whisper", device="cpu", compute_type="int8")
segments, info = model.transcribe("speech.wav")
text = "".join(segment.text for segment in segments)
client.flush()
client.close()
```

The SDK records model load time, backend-specific model identity, format (`mlx` or `ctranslate2`), family (`whisper`), accelerator and compute type/quantization. MLX's internal model cache produces one load per SDK owner; an already cached model is reported with zero load time and `model_cache_hit=true`. Warm-up transcriptions are tracked too.

Transcriptions include wall-clock duration, audio metadata, `transcript_word_count` and detected `language` attributes. `language_confidence` is included when faster-whisper supplies it; MLX does not expose it. WAV paths provide original duration, sample rate, channels and bit depth. One-dimensional arrays are treated as 16 kHz mono, as required by these APIs. Other file formats retain the duration supplied by faster-whisper when available; the SDK does not decode files a second time to inspect them. These APIs do not supply text token usage or token-level TTFT, so the SDK does not invent those values.

faster-whisper timing runs from `transcribe()` through segment exhaustion, including time between iterator pulls. Keep consuming within the trace/span that should own the operation. Explicit `segments.close()` emits one unsuccessful inference with `STREAM_CLOSED` and the word count consumed so far. Decoder failures emit one error event and propagate unchanged. Merely abandoning an iterator does not imply completion; explicitly close it when stopping early. Existing faster-whisper model objects stay bound to their original SDK client after replacement; recreate them for a new client.

Audio and transcript attachments require `attachments_enabled=True`. Capture is off by default. The adapter caps retained transcript/audio attachment size at 10 MiB, and the configured attachment manager may apply stricter limits or a filter. No transcript text enters scalar event attributes. Use an `attachment_filter` when an application allows transcript capture but forbids audio upload. Raw audio file attachments must still exist when the transcription finishes.

See [the runnable example](../examples/whisper_example.py). Compatibility CI tests real decoding with random-weight MLX Whisper and the pinned public faster-whisper tiny model. Offline faster-whisper tests can use `WILDEDGE_FASTER_WHISPER_MODEL=/path/to/model`.
