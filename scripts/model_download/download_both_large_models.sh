#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/download_qwen3_30b_a3b_instruct.sh" "$@"
bash "${SCRIPT_DIR}/download_qwen3_32b_judge.sh" "$@"
