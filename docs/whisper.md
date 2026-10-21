# Local Whisper speech recognition

WildEdge tracks local Whisper transcription with two integrations:

- `mlx_whisper` for [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) on Apple Silicon
- `faster_whisper` for [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2)

Initialize the SDK before loading a model or importing `transcribe` by name:

```python
import wildedge

client = wildedge.init(integrations=["mlx_whisper"])
import mlx_whisper

result = mlx_whisper.transcribe("speech.wav", path_or_hf_repo="/models/whisper")
```

faster-whisper decodes lazily, so the transcription is recorded once you have
read all of its segments:

```python
client = wildedge.init(integrations=["faster_whisper"])
from faster_whisper import WhisperModel

model = WhisperModel("/models/whisper", device="cpu", compute_type="int8")
segments, info = model.transcribe("speech.wav")
text = "".join(segment.text for segment in segments)
```

`BatchedInferencePipeline(model).transcribe(...)` is tracked the same way.

## What's recorded

**Model loads**

- Load time, format (`mlx` or `ctranslate2`), family (`whisper`), accelerator
  and compute type or quantization.
- The model ID is the backend plus the model name, for example
  `faster_whisper:Systran/faster-whisper-tiny`. A Hugging Face repo keeps its
  repo name, also when loaded from a cache path. A local folder is named after
  the folder.
- Every load is recorded, including a second `WhisperModel` for the same model.
- mlx-whisper caches one model at a time and reloads when you switch models.
  Each reload is recorded. A model that was already cached before the SDK
  started is recorded once, with zero load time and `model_cache_hit=true`.

**Transcriptions**

- Wall-clock duration and `language`.
- `language_confidence` (faster-whisper only; mlx-whisper doesn't report it).
- `transcript_word_count`, except for languages written without spaces between
  words, such as Chinese, Japanese and Thai.
- `time_to_first_segment_ms` (faster-whisper only).
- Audio metadata:
  - WAV files: duration, sample rate, channels and bit depth.
  - 1-D arrays: treated as 16 kHz mono, as both libraries expect.
  - Other formats: duration only, when faster-whisper reports it. The SDK
    doesn't decode the audio a second time.

Token counts aren't reported; neither library exposes them.

## faster-whisper segment lifecycle

Timing runs from `transcribe()` until the last segment is read, including time
your code spends between segments. Read the segments inside the trace or span
that should own the transcription.

| How the segments end | Recorded as |
| --- | --- |
| All segments read | One successful transcription |
| `segments.close()` | One unsuccessful transcription, `STREAM_CLOSED` |
| Loop exits early without `close()` | One unsuccessful transcription, `STREAM_ABANDONED`, sent when the iterator is garbage collected and timed up to the last segment read |
| Never iterated (for example, only `info.language` was used) | Nothing |
| Decoder error | One error event; the exception reaches your code unchanged |

A `WhisperModel` stays with the SDK client that was active when it was created.
After you replace the client, create the model again; until then, it isn't
tracked and the SDK warns once.

## Explicit audio and transcript attachments

Automatic instrumentation records metadata only, even with
`attachments_enabled=True`. To upload audio or transcript text, supply the
attachments explicitly through [manual tracking](manual-tracking.md#track-attachments).
For example, with a faster-whisper model:

```python
import time
import wildedge
from wildedge import Attachment

# Use manual tracking for this call so it produces one inference event.
client = wildedge.init(integrations=[], attachments_enabled=True)
from faster_whisper import WhisperModel

model = WhisperModel("/models/whisper", device="cpu", compute_type="int8")
handle = client.register_model(model, auto_instrument=False)
started = time.perf_counter()
segments, info = model.transcribe("speech.wav")
text = "".join(segment.text for segment in segments)
handle.track_inference(
    duration_ms=round((time.perf_counter() - started) * 1000),
    input_modality="audio",
    output_modality="text",
    attachments=[
        Attachment(content_type="audio/wav", role="input", path="speech.wav"),
        Attachment(content_type="text/plain", role="output", data=text.encode()),
    ],
)
client.close(timeout=10)
```

Choose either attachment or both. Files must exist when `track_inference` runs;
normal attachment size limits and the optional `attachment_filter` apply.

## Examples and tests

See [the runnable example](../examples/whisper_example.py). Compatibility CI
decodes real audio with a random-weight MLX Whisper model and the pinned public
faster-whisper tiny model. To run the faster-whisper tests offline, set
`WILDEDGE_FASTER_WHISPER_MODEL=/path/to/model`.
