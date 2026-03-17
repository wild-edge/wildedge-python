# WildEdge Python SDK

[![CI](https://github.com/wildedge/wildedge-python/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/wildedge/wildedge-python/actions/workflows/ci.yml)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://pypi.org/project/wildedge-sdk/)
[![Tested on Linux](https://img.shields.io/badge/tested%20on-linux-blue)](https://github.com/wildedge/wildedge-python/actions/workflows/ci.yml)
[![Tested on macOS](https://img.shields.io/badge/tested%20on-macOS-blue)](https://github.com/wildedge/wildedge-python/actions/workflows/ci.yml)
[![Tested on Windows](https://img.shields.io/badge/tested%20on-windows-blue)](https://github.com/wildedge/wildedge-python/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/wildedge/wildedge-python/branch/main/graph/badge.svg)](https://codecov.io/gh/wildedge/wildedge-python)

On-device ML inference monitoring for Python. Tracks latency, errors, and model metadata without any code modifications.

> **Pre-release:** The API is unstable and may change between versions. Semantic versioning will apply from the first stable release.

## Install

```bash
uv add wildedge-sdk
```

## CLI

Drop `wildedge run` in front of your existing command. WildEdge instruments the runtime before your code starts. No SDK calls required in user code.

```bash
WILDEDGE_DSN="https://<secret>@ingest.wildedge.dev/<key>" \
wildedge run --integrations timm -- python app.py
```

Validate your environment before deploying:

```bash
wildedge doctor --integrations all --network-check
```

Useful flags:

| Flag | Description |
|---|---|
| `--integrations` | Comma-separated list of integrations to activate (or `all`) |
| `--hubs` | Hub trackers to activate: `huggingface`, `torchhub` |
| `--print-startup-report` | Print per-integration status at startup |
| `--strict-integrations` | Fail if a requested integration can't be loaded |
| `--no-propagate` | Don't pass WildEdge env vars to child processes |

## SDK

```python
import wildedge

client = wildedge.WildEdge(dsn="...")  # or WILDEDGE_DSN env var
client.instrument("transformers", hubs=["huggingface"])

# models loaded after this point are tracked automatically
```

## Supported integrations

**On-device**

| Integration | Example |
|---|---|
| `transformers` | [transformers_example.py](examples/transformers_example.py) |
| `mlx` | [mlx_example.py](examples/mlx_example.py) |
| `timm` | [timm_example.py](examples/timm_example.py) |
| `gguf` | [gguf_example.py](examples/gguf_example.py) |
| `onnx` | [onnx_example.py](examples/onnx_example.py) |
| `ultralytics` | - |
| `tensorflow` | [tensorflow_example.py](examples/tensorflow_example.py) |
| `torch` | [pytorch_example.py](examples/pytorch_example.py) |
| `keras` | [keras_example.py](examples/keras_example.py) |

**Remote models**

| Integration | Example |
|---|---|
| `openai` | [openai_example.py](examples/openai_example.py) |

**Hub tracking**

Pass `hubs=` to track model download provenance. Hubs are framework-agnostic and can be combined with any integration.

| Hub | Tracks |
|---|---|
| `huggingface` | Downloads via `huggingface_hub` |
| `torchhub` | Downloads via `torch.hub` |

For unsupported frameworks, see [Manual tracking](docs/manual-tracking.md).

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `dsn` | - | `https://<secret>@ingest.wildedge.dev/<key>` (or `WILDEDGE_DSN`) |
| `app_version` | `None` | Your app's version string |
| `app_identity` | `<project_key>` | Namespace for offline persistence. Set per-app in multi-process workloads (or `WILDEDGE_APP_IDENTITY`) |
| `enable_offline_persistence` | `true` | Persist unsent events to disk and replay on restart |
| `sampling_interval_s` | `30.0` | Seconds between background hardware snapshots. Set to `0` or `None` to disable (or `WILDEDGE_SAMPLING_INTERVAL_S`) |

For advanced options (batching, queue tuning, dead-letter storage), see [Configuration](docs/configuration.md).

## Privacy

Report security & priact issues to: wildedge@googlegroups.com.

## Links

- [Documentation](https://docs.wildedge.dev)
- [Compatibility Matrix](docs/compatibility.md)
- [Changelog](CHANGELOG.md)
- [License](LICENSE)
