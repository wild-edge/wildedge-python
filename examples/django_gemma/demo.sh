#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${WILDEDGE_DSN:-}" ]]; then
  echo 'Set WILDEDGE_DSN first, e.g. export WILDEDGE_DSN="https://<secret>@ingest.wildedge.dev/<key>"' >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

uv sync

uv run wildedge doctor --integrations gguf --hubs huggingface

# wildedge run replaces this process (os.execle) so sitecustomize.py
# auto-installs the runtime before Django loads, patching Llama.__init__
# for automatic inference tracking.
#
# Server choice:
#   macOS  — waitress (thread-pool, no fork). Metal is initialised once in the
#             main process and shared safely across request threads.
#   Linux  — gunicorn (multi-process fork). Requires llama-cpp-python built
#             without Metal: CMAKE_ARGS="-DGGML_METAL=OFF" pip install llama-cpp-python
#             Then: wildedge run ... -- gunicorn gemmaapp.wsgi:application --config gunicorn.conf.py
if [[ "$(uname)" == "Darwin" ]]; then
  uv run wildedge run \
    --print-startup-report \
    --integrations gguf \
    --hubs huggingface \
    -- waitress-serve --port=8100 gemmaapp.wsgi:application
else
  uv run wildedge run \
    --print-startup-report \
    --integrations gguf \
    --hubs huggingface \
    -- gunicorn gemmaapp.wsgi:application --config gunicorn.conf.py
fi

# Test with:
#   curl -s -X POST http://localhost:8100/infer/ \
#     -H "Content-Type: application/json" \
#     -d '{"prompt": "What is on-device AI in one sentence?"}' | jq .
