#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
python -P -m compileall -q src scripts tests
python -P -m pytest -q
python -P scripts/experiment1_pipeline/run_smoke_episode.py --output-dir outputs/smoke --overwrite
