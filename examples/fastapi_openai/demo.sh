#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${WILDEDGE_DSN:-}" ]]; then
  echo 'Set WILDEDGE_DSN first, e.g. export WILDEDGE_DSN="https://<secret>@ingest.wildedge.dev/<key>"' >&2
  exit 1
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo 'Set OPENROUTER_API_KEY first, e.g. export OPENROUTER_API_KEY="sk-or-..."' >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

uv sync

uv run wildedge doctor --integrations openai

# wildedge run execs uvicorn with wildedge/autoload/ prepended to PYTHONPATH.
# sitecustomize.py bootstraps the runtime before the app loads, instrumenting
# the OpenAI client for automatic inference tracking.
#
# --reload spawns a fresh worker process via exec each time code changes.
# The module-level guard in sitecustomize.py ensures that worker bootstraps
# correctly (unlike the old os.environ guard, which propagated across exec
# and blocked the worker's init).
uv run wildedge run \
  --print-startup-report \
  --integrations openai \
  -- uvicorn app.main:app --reload --port 8000

# Test with:
#   curl -s -X POST http://localhost:8000/chat \
#     -H "Content-Type: application/json" \
#     -d '{"prompt": "What is on-device AI in one sentence?"}' | jq .
