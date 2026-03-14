# WildEdge Python SDK

[![CI](https://github.com/wildedge/wildedge-python/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/wildedge/wildedge-python/actions/workflows/ci.yml)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://pypi.org/project/wildedge-sdk/)
[![Tested on Linux](https://img.shields.io/badge/tested%20on-linux-blue)](https://github.com/wildedge/wildedge-python/actions/workflows/ci.yml)
[![Tested on macOS](https://img.shields.io/badge/tested%20on-macOS-blue)](https://github.com/wildedge/wildedge-python/actions/workflows/ci.yml)
[![Tested on Windows](https://img.shields.io/badge/tested%20on-windows-blue)](https://github.com/wildedge/wildedge-python/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://codecov.io/gh/wildedge/wildedge-python/branch/main/graph/badge.svg)](https://codecov.io/gh/wildedge/wildedge-python)

On-device ML inference monitoring for Python. Tracks latency, errors, and model metadata. No inputs or outputs captured.

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

| Integration | Patches | Hub tracking | Example |
|---|---|---|---|
| `transformers` | `pipeline()`, `AutoModel.from_pretrained()` | `huggingface` | [transformers_example.py](examples/transformers_example.py) |
| `mlx` | `mlx_lm.load()`, `mlx_lm.generate()` | `huggingface` | [mlx_example.py](examples/mlx_example.py) |
| `timm` | `timm.create_model()` | `huggingface`, `torchhub` | [timm_example.py](examples/timm_example.py) |
| `gguf` | `llama_cpp.Llama.__init__` | `huggingface` | [gguf_example.py](examples/gguf_example.py) |
| `onnx` | `ort.InferenceSession` | `huggingface` | [onnx_example.py](examples/onnx_example.py) |
| `ultralytics` | `ultralytics.YOLO.__init__` | - | - |
| `tensorflow` | `tf.keras.models.load_model`, `tf.saved_model.load` | - | [tensorflow_example.py](examples/tensorflow_example.py) |
| `torch` | forward hooks via `client.load()` | `torchhub` | [pytorch_example.py](examples/pytorch_example.py) |
| `keras` | forward hooks via `client.load()` | - | [keras_example.py](examples/keras_example.py) |

For `torch` and `keras`, models are user-defined subclasses so there's no constructor to patch. Use `client.load()` to get load/unload tracking alongside inference:

```python
model = client.load(MyModel)
output = model(x)  # tracked automatically
```

## Manual tracking

Use `@wildedge.track` as a decorator or context manager when auto-instrumentation isn't available:

```python
handle = client.register_model(my_model)

@wildedge.track(handle)
def run(input):
    return my_model.predict(input)
```

## Configuration

| Parameter | Default | Env var | Description |
|---|---|---|---|
| `dsn` | - | `WILDEDGE_DSN` | `https://<secret>@ingest.wildedge.dev/<key>` |
| `app_version` | `None` | - | Your app's version string |
| `app_identity` | `<project_key>` | `WILDEDGE_APP_IDENTITY` | Namespace for offline persistence; set per-app in multi-process workloads |
| `debug` | `false` | `WILDEDGE_DEBUG` | Log events to console |
| `batch_size` | `10` | - | Events per transmission (1-100) |
| `flush_interval_sec` | `60` | - | Max seconds between flushes (1-3600) |
| `max_queue_size` | `200` | - | In-memory buffer limit (10-10000) |
| `enable_offline_persistence` | `true` | - | Persist unsent events to disk and replay on restart |
| `max_event_age_sec` | `900` | - | Max age before dead-lettering |
| `enable_dead_letter_persistence` | `false` | - | Persist dropped batches to disk |

## Privacy

WildEdge captures **no inputs or outputs**. Only metadata: latency, errors, model info, and download provenance. All inference runs locally; only telemetry is transmitted over HTTPS.

Report security issues to security@wildedge.dev.

## Links

- [Documentation](https://docs.wildedge.dev)
- [Compatibility Matrix](docs/compatibility.md)
- [Changelog](CHANGELOG.md)
- [License](LICENSE)
