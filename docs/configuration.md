# Configuration

Full reference for all `WildEdge` client parameters. For runtime behavior
(no-DSN mode, strict mode and exit codes, forking servers), see
[Deployment](deployment.md).

## Core

| Parameter | Default | Env var | Description |
|---|---|---|---|
| `dsn` | - | `WILDEDGE_DSN` | `https://<secret>@ingest.wildedge.dev/<key>`. If unset, the client is a no-op. |
| `app_version` | `None` | - | Your app's version string |
| `app_identity` | `<project_key>` | `WILDEDGE_APP_IDENTITY` | Namespace for offline persistence. Set per-app in multi-process workloads |
| `enable_offline_persistence` | `true` | - | Persist unsent events to disk and replay on restart |
| `sampling_interval_s` | `30.0` | `WILDEDGE_SAMPLING_INTERVAL_S` | Seconds between background hardware snapshots. Set to `0` or `None` to disable |

## Advanced

| Parameter | Default | Env var | Description |
|---|---|---|---|
| `batch_size` | `10` | - | Events per transmission (1-100) |
| `flush_interval_sec` | `60` | - | Max seconds between flushes (1-3600) |
| `max_queue_size` | `200` | - | In-memory buffer limit (10-10000) |
| `max_event_age_sec` | `900` | - | Max age in seconds before an event is dropped |
| `enable_dead_letter_persistence` | `false` | - | Persist dropped batches to disk for later inspection |
| `debug` | `false` | `WILDEDGE_DEBUG` | Log SDK internals to console |

## Attachments

Opt-in upload of raw inference inputs and outputs alongside telemetry. Off by
default, and requires the attachments feature to be enabled on your project.
Bytes are buffered to disk and uploaded in the background via presigned URLs;
event flushes never block on an upload.

| Parameter | Default | Env var | Description |
|---|---|---|---|
| `attachments_enabled` | `false` | `WILDEDGE_ATTACHMENTS_ENABLED` | Enable raw input/output upload |
| `attachment_modalities` | `["text"]` | `WILDEDGE_ATTACHMENT_MODALITIES` | Kinds of data that may be uploaded: `text`, `image`, `audio`, `video`, `other`, or `"all"` |
| `max_attachments_per_inference` | `10` | - | Cap on attachments buffered per inference event |
| `max_attachment_size_bytes` | `10485760` | - | Per-attachment size limit (10 MB); larger ones are dropped |
| `attachment_storage_strategy` | `"file"` | - | `"file"` (bytes on disk) or `"inline"` (small blobs embedded in the record) |
| `attachment_filter` | `None` | - | `Callable[[list[Attachment]], list[Attachment]]` to redact/drop before buffering |
| `attachment_dir` | per `app_identity` | - | Override the local buffer directory |

### Choosing what may be uploaded

`attachment_modalities` decides which kinds of data leave the device. It
applies to every attachment, whether an integration captured it or you passed
it to `track_inference`. The default is text only: prompts, replies and
transcripts. Voice recordings, images and video can identify a person, so they
are uploaded only when you list them:

```python
from wildedge import Modality

wildedge.init(
    attachments_enabled=True,
    attachment_modalities=[Modality.TEXT, Modality.AUDIO],
)
```

Plain strings work too (`["text", "audio"]`), and `"all"` allows every
modality. From the CLI: `wildedge run --attachments --attachment-modalities text,audio`.

Each attachment's modality comes from its content type (`audio/wav` is
`audio`, `application/json` is `text`). Anything else, such as raw tensors,
is `other`. The first time an attachment is dropped this way, the SDK logs a
warning naming the setting to change (only a debug message if you set
`attachment_modalities` yourself). Blocked attachments are dropped before
`attachment_filter` runs, and integrations skip building them at all, so a
Whisper client without `audio` never encodes the audio.

`attachment_modalities` doesn't turn uploads on by itself:
`attachments_enabled` is still required, and the SDK warns if the list is set
without it.

For finer rules, such as uploading inputs but not outputs, use
`attachment_filter`.

See [Track attachments](manual-tracking.md#track-attachments) for usage.

### When ML libraries are imported

`import wildedge` does not import any ML framework or API client. The SDK imports a library only when you use it:

- activating an integration (`wildedge.init(integrations=[...])`, `client.instrument(...)` or `wildedge run`) imports that library;
- registering a model imports the library the model came from, and nothing else. For example, registering a PyTorch model does not import timm (unless it is a 1000-class classifier, for which timm supplies the ImageNet labels), and registering a Keras model does not import TensorFlow.

Which integrations are activated by default is unchanged. `wildedge.init()` activates only the ones you list, while `wildedge run` activates every installed one unless you pass `--integrations`.
