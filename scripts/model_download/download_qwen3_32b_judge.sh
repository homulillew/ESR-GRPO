#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"

python "${SCRIPT_DIR}/download_hf_model.py" \
  --repo-id "Qwen/Qwen3-32B" \
  --output "${MODEL_ROOT}/Qwen3-32B" \
  "$@"
