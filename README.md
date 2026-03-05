# WildEdge Python SDK

[![CI](https://github.com/wildedge/wildedge-python/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/wildedge/wildedge-python/actions/workflows/ci.yml)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://pypi.org/project/wildedge-sdk/)
[![Tested on Linux](https://img.shields.io/badge/tested%20on-linux-blue)](https://github.com/wildedge/wildedge-python/actions/workflows/ci.yml)
[![Tested on macOS](https://img.shields.io/badge/tested%20on-macOS-blue)](https://github.com/wildedge/wildedge-python/actions/workflows/ci.yml)
[![Tested on Windows](https://img.shields.io/badge/tested%20on-windows-blue)](https://github.com/wildedge/wildedge-python/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://codecov.io/gh/wildedge/wildedge-python/branch/main/graph/badge.svg)](https://codecov.io/gh/wildedge/wildedge-python)

On-device ML inference monitoring for Python. Tracks latency, errors, and model metadata without capturing inputs or outputs.

## Install

```bash
uv add wildedge-sdk
```

## Quick start

```python
import wildedge

client = wildedge.WildEdge(
    dsn="...",  # or set WILDEDGE_DSN
)
```

## CLI wrapper

Use `wildedge run` to execute an existing Python entrypoint with WildEdge runtime initialization and integration instrumentation enabled before user code starts:

```bash
wildedge run --dsn "https://<secret>@ingest.wildedge.dev/<key>" -- python app.py
```

Module entrypoints are supported:

```bash
wildedge run -- python -m your_package.main --arg value
```

## Integrations

Call `client.instrument()` to activate auto-tracking for a supported library. Models created afterwards are registered and timed automatically with no changes to existing call sites.

See the `examples/` folder for complete working examples.

### Integration initialization

Initialize integrations at process startup, before model loading begins. Instrumentation patches are applied per process and should be installed before imports and constructor calls on instrumented libraries.

For high-priority paths, keep explicit registration with `client.load(...)` or `client.register_model(...)` as a fallback when model creation does not go through a patched API.
For an explicit fallback pattern, see `examples/gguf_gemma_manual_example.py`.

### PyTorch (custom models)

PyTorch models are user-defined subclasses, so there is no single constructor to patch. Use `client.load()` to time construction and track load/unload automatically; inference is tracked via forward hooks once the model is registered.

```python
model = client.load(MyModel)
output = model(x)  # tracked automatically
```

See `examples/pytorch_example.py` for a complete example.

### Keras (custom models)

Same pattern as PyTorch:

```python
model = client.load(MyKerasModel)
output = model(x)  # tracked automatically
```

See `examples/keras_example.py` for a complete example.

### ONNX Runtime

```python
import onnxruntime as ort

client.instrument("onnx")

session = ort.InferenceSession("yolov8n.onnx")
outputs = session.run(None, {"input": image})  # tracked automatically
```

See `examples/onnx_example.py` for a complete example.

### TensorFlow

```python
import tensorflow as tf

client.instrument("tensorflow")

model = tf.keras.models.load_model("model.keras")  # tracked automatically
output = model(batch, training=False)  # tracked automatically
```

See `examples/tensorflow_example.py` for a complete example.

### timm

```python
import timm

client.instrument("timm")

model = timm.create_model("resnet50", pretrained=True)
output = model(image_tensor)  # tracked automatically
```

See `examples/timm_example.py` for a complete example.

### GGUF / llama.cpp

```python
from llama_cpp import Llama

client.instrument("gguf")

llm = Llama("llama-3.2-1b.Q4_K_M.gguf", n_ctx=2048, n_gpu_layers=-1)
result = llm("What is the capital of France?")  # tracked automatically
```

See `examples/gguf_example.py` for a complete example.

## Limitations

- Currently supports only Python 3.10+ due to use of modern type annotations.
- Overhead: Less than 1% latency increase in internal benchmarks.
- For air-gapped environments, on-premise server installation is required.

## Manual tracking

Use `@wildedge.track` as a decorator or context manager when auto-instrumentation isn't available:

```python
handle = client.register_model(my_model)

# decorator
@wildedge.track(handle)
def run(input):
    return my_model.predict(input)

# context manager
with wildedge.track(handle):
    result = my_model.predict(input)
```

## Configuration

| Parameter | Default | Env var | Description |
|---|---|---|---|
| `dsn` | `-` | `WILDEDGE_DSN` | Required. `https://<secret>@ingest.wildedge.dev/<key>` |
| `app_version` | `None` | `-` | Optional. Your app's version string. |
| `debug` | `false` | `WILDEDGE_DEBUG` | Log events to console. |
| `batch_size` | `10` | `-` | Events per transmission (recommended: 1-100). |
| `flush_interval_sec` | `60` | `-` | Max seconds between flushes (recommended: 1-3600). |
| `max_queue_size` | `200` | `-` | In-memory buffer limit (recommended: 10-10000). |

## Testing

### Run tests locally

Install development dependencies and run the test suite:

```bash
uv sync --group dev
uv run pytest
```

### Run tests across Python versions

Use `tox` to run the test suite against all supported Python versions (3.10+):

```bash
uv sync --group dev
tox
```

Compatibility matrix details are documented in `docs/compatibility.md`.

Run the compatibility matrix locally with one command:

```bash
python3 scripts/run_compat_local.py
```

To fail when a dependency row is unsupported on your local platform, use:

```bash
python3 scripts/run_compat_local.py --strict-unsupported
```

## Security

WildEdge SDK privacy model:
- **No input/output capture**: Only metadata (latency, errors, model info) is collected.
- **Secure transmission**: Data is sent over HTTPS to WildEdge servers.
- **Local processing**: All inference happens locally; SDK only monitors performance.
- **DSN-based auth**: Uses project-specific secrets for authentication.

If you discover a security issue, please email security@wildedge.dev instead of creating a public issue.

## Links

- [Full Documentation](https://docs.wildedge.dev)
- [Website](https://wildedge.dev)
- [Compatibility Matrix](docs/compatibility.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [License](LICENSE)
