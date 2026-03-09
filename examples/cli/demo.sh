#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${WILDEDGE_DSN:-}" ]]; then
  echo 'Set WILDEDGE_DSN first, e.g. export WILDEDGE_DSN="https://<secret>@ingest.wildedge.dev/<key>"' >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
EXAMPLE_SCRIPT="${PROJECT_DIR}/cli_wrapper_example.py"

cd "${PROJECT_DIR}"

uv sync

uv run wildedge doctor --integrations timm --hubs huggingface,torchhub

uv run wildedge run \
  --print-startup-report --integrations timm --hubs huggingface,torchhub -- \
  "${EXAMPLE_SCRIPT}"
