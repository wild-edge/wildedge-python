## Compatibility Matrix

Each integration is tested against the pinned library versions below: the
oldest supported release (`min`) and a recent one (`current`). The table is
generated from `scripts/compat_matrix.py`, which also defines the CI jobs.

### What the tests check

`tests/compat/test_<integration>_compat.py` loads a real model with the real
library and checks the events the SDK records: a `model_load` event, then an
`inference` event for the same model with the expected fields (token counts
for LLMs, input modality for vision models). Apart from one download for gguf, nothing
touches the network:

- openai and anthropic get canned responses through the SDK's own HTTP
  client (`MockTransport`).
- onnx loads a tiny model committed in `tests/compat/fixtures`. gguf
  downloads a 1.2 MB model on first use, checked against a pinned sha256.
- torch, timm, tensorflow, transformers, ultralytics and mlx build models from
  a config, with random weights.

A test never skips. A library that fails to install or import fails the job,
and a step checks that the installed versions match the pins. A known SDK
bug is marked `xfail(strict=True)` with its issue number, so the job fails
once the bug is fixed and the marker has to come off.

### When they run

On pull requests into `main` or a release branch, on pushes to them, nightly
against `devel`, and on manual runs. Every run covers all the combinations
below. Python 3.14 rows are canaries and don't fail the run. mlx runs on
macOS.

To run the full matrix locally: `python scripts/run_compat_local.py`.
`python scripts/compat_matrix.py table` prints the table below.

### Matrix

| Integration | Version set | Dependencies | Supported Python |
|---|---|---|---|
| `anthropic` | `min` | `anthropic==0.25.0` | 3.10, 3.11, 3.12, 3.13 |
| `anthropic` | `current` | `anthropic==1.15.0` | 3.10, 3.11, 3.12, 3.13 |
| `onnx` | `min` | `onnxruntime==1.18.1`, `numpy==1.26.4` | 3.10, 3.11, 3.12, 3.13, 3.14 |
| `onnx` | `min` (`py3.13` override) | `onnxruntime==1.20.1`, `numpy==2.1.3` | 3.13 |
| `onnx` | `current` | `onnxruntime==1.31.0`, `numpy==2.2.6` | 3.10, 3.11, 3.12, 3.13, 3.14 |
| `onnx` | `current` (`py3.10` override) | `onnxruntime==1.23.2`, `numpy==2.2.6` | 3.10 |
| `torch` | `min` | `torch==2.4.1`, `numpy==1.26.4` | 3.10, 3.11, 3.12, 3.13, 3.14 |
| `torch` | `min` (`py3.13` override) | `torch==2.6.0`, `numpy==2.1.3` | 3.13 |
| `torch` | `min` (`py3.14` override) | `torch==2.5.0`, `numpy==2.1.3` | 3.14 |
| `torch` | `current` | `torch==2.14.1`, `numpy==2.2.6` | 3.10, 3.11, 3.12, 3.13, 3.14 |
| `torch` | `current` (`py3.14` override) | `torch==2.14.1`, `numpy==2.5.4` | 3.14 |
| `timm` | `min` | `torch==2.4.1`, `torchvision==0.19.1`, `timm==1.0.11`, `numpy==1.26.4` | 3.10, 3.11, 3.12, 3.13, 3.14 |
| `timm` | `min` (`py3.13` override) | `torch==2.6.0`, `torchvision==0.21.0`, `timm==1.0.11`, `numpy==2.1.3` | 3.13 |
| `timm` | `min` (`py3.14` override) | `torch==2.5.0`, `torchvision==0.19.1`, `timm==1.0.11`, `numpy==2.1.3` | 3.14 |
| `timm` | `current` | `torch==2.14.1`, `torchvision==0.29.1`, `timm==1.0.30`, `numpy==2.2.6` | 3.10, 3.11, 3.12, 3.13, 3.14 |
| `timm` | `current` (`py3.14` override) | `torch==2.14.1`, `torchvision==0.29.1`, `timm==1.0.30`, `numpy==2.5.4` | 3.14 |
| `tensorflow` | `min` | `tensorflow==2.16.1`, `keras==3.3.3`, `numpy==1.26.4` | 3.10, 3.11, 3.12 |
| `tensorflow` | `current` | `tensorflow==2.21.0`, `keras==3.15.1`, `numpy==2.2.6` | 3.10, 3.11, 3.12 |
| `tensorflow` | `current` (`py3.10` override) | `tensorflow==2.21.0`, `keras==3.12.4`, `numpy==2.2.6` | 3.10 |
| `gguf` | `min` | `llama-cpp-python==0.2.90`, `numpy==1.26.4` | 3.10, 3.11, 3.12, 3.13 |
| `gguf` | `current` | `llama-cpp-python==0.3.36`, `numpy==2.2.6` | 3.10, 3.11, 3.12, 3.13 |
| `openai` | `min` | `openai==1.30.0` | 3.10, 3.11, 3.12, 3.13 |
| `openai` | `current` | `openai==3.35.1` | 3.10, 3.11, 3.12, 3.13 |
| `transformers` | `min` | `transformers==4.40.0`, `torch==2.4.1`, `numpy==1.26.4` | 3.10, 3.11, 3.12, 3.13 |
| `transformers` | `min` (`py3.13` override) | `transformers==4.45.0`, `torch==2.5.0`, `numpy==2.1.3` | 3.13 |
| `transformers` | `current` | `transformers==5.20.1`, `torch==2.14.1`, `numpy==2.2.6` | 3.10, 3.11, 3.12, 3.13 |
| `ultralytics` | `min` | `ultralytics==8.0.200`, `numpy==1.26.4` | 3.10, 3.11, 3.12, 3.13 |
| `ultralytics` | `current` | `ultralytics==8.4.182`, `numpy==2.2.6` | 3.10, 3.11, 3.12, 3.13 |
| `mlx` | `current` | `mlx==0.32.4`, `mlx-lm==0.32.2` | 3.12, 3.13 |
| `faster_whisper` | `current` | `faster-whisper==1.2.1`, `ctranslate2==4.8.2`, `numpy==2.2.6` | 3.12, 3.13 |
| `mlx_whisper` | `current` | `mlx-whisper==0.4.3`, `mlx==0.32.3`, `numpy==2.2.6` | 3.12, 3.13 |
