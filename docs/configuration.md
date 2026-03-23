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
