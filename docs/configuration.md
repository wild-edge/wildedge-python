# Configuration

Full reference for all `WildEdge` client parameters.

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

Opt-in upload of raw inference inputs/outputs (e.g. the source image or the
generated text) alongside telemetry. Off by default; requires the attachment
feature enabled on your project. Bytes are buffered locally and uploaded in the
background via presigned URLs, so the event flush never waits on the upload.

| Parameter | Default | Env var | Description |
|---|---|---|---|
| `attachments_enabled` | `false` | `WILDEDGE_ATTACHMENTS_ENABLED` | Enable raw input/output upload |
| `max_attachments_per_inference` | `10` | - | Cap on attachments buffered per inference event |
| `max_attachment_size_bytes` | `10485760` | - | Per-attachment size limit (10 MB); larger ones are dropped |
| `attachment_storage_strategy` | `"file"` | - | `"file"` (bytes on disk) or `"inline"` (small blobs embedded in the record) |
| `attachment_filter` | `None` | - | `Callable[[list[Attachment]], list[Attachment]]` to redact/drop before buffering |
| `attachment_dir` | per `app_identity` | - | Override the local buffer directory |

See [Track attachments](manual-tracking.md#track-attachments) for usage.
