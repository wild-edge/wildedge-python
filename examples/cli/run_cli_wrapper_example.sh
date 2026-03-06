#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${WILDEDGE_DSN:-}" ]]; then
  echo "WILDEDGE_DSN is required. Example:" >&2
  echo '  export WILDEDGE_DSN="https://<secret>@ingest.wildedge.dev/<key>"' >&2
  exit 1
fi

wildedge run --print-startup-report --integrations timm -- \
  python examples/cli/cli_wrapper_example.py
