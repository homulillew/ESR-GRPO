#!/usr/bin/env bash
set -euo pipefail

EXPER_DIR="$(cd "$(dirname "$0")" && pwd)"

: "${BCPLUS_DATASET:?set BCPLUS_DATASET to browsecomp_plus_decrypted.jsonl}"
: "${BCPLUS_SPLIT_MANIFEST:?set BCPLUS_SPLIT_MANIFEST to v1.json}"
: "${BCPLUS_QRELS:?set BCPLUS_QRELS to qrel_evidence.txt}"
: "${AREX_BASE_URL:?set AREX_BASE_URL, for example http://127.0.0.1:8002/v1}"
: "${AREX_RETRIEVAL_URL:?set AREX_RETRIEVAL_URL, for example http://127.0.0.1:8000}"

AREX_MODEL="${AREX_MODEL:-AREX-Turbo}"
AREX_API_KEY="${AREX_API_KEY:-EMPTY}"
AREX_OUTPUT_DIR="${AREX_OUTPUT_DIR:-${EXPER_DIR}/outputs/validation}"
AREX_SPLIT="${AREX_SPLIT:-validation}"
AREX_LIMIT="${AREX_LIMIT:-10}"
AREX_OFFSET="${AREX_OFFSET:-0}"
AREX_TOP_K="${AREX_TOP_K:-10}"
AREX_MAX_TURNS="${AREX_MAX_TURNS:-100}"
AREX_MAX_TOKENS="${AREX_MAX_TOKENS:-8192}"
AREX_RUN_SMOKE="${AREX_RUN_SMOKE:-1}"

mkdir -p "${AREX_OUTPUT_DIR}"

CORPUS_ARGS=()
if [[ -n "${BCPLUS_CORPUS_FILE:-}" ]]; then
  CORPUS_ARGS=(--corpus-file "${BCPLUS_CORPUS_FILE}")
fi

python "${EXPER_DIR}/preflight.py" \
  --dataset "${BCPLUS_DATASET}" \
  --split-manifest "${BCPLUS_SPLIT_MANIFEST}" \
  --qrels "${BCPLUS_QRELS}" \
  "${CORPUS_ARGS[@]}" \
  --model-base-url "${AREX_BASE_URL}" \
  --model "${AREX_MODEL}" \
  --api-key "${AREX_API_KEY}" \
  --retrieval-url "${AREX_RETRIEVAL_URL}" \
  --output "${AREX_OUTPUT_DIR}/preflight.json" \
  --strict

if [[ "${AREX_RUN_SMOKE}" == "1" ]]; then
  python "${EXPER_DIR}/run_arex_bcplus.py" \
    --dataset "${BCPLUS_DATASET}" \
    --split-manifest "${BCPLUS_SPLIT_MANIFEST}" \
    --split "${AREX_SPLIT}" \
    --limit 1 \
    --offset "${AREX_OFFSET}" \
    --output-dir "${AREX_OUTPUT_DIR}/smoke" \
    --model-base-url "${AREX_BASE_URL}" \
    --model "${AREX_MODEL}" \
    --api-key "${AREX_API_KEY}" \
    --retrieval-url "${AREX_RETRIEVAL_URL}" \
    "${CORPUS_ARGS[@]}" \
    --top-k "${AREX_TOP_K}" \
    --max-turns "${AREX_MAX_TURNS}" \
    --max-tokens "${AREX_MAX_TOKENS}" \
    --fail-fast
fi

python "${EXPER_DIR}/run_arex_bcplus.py" \
  --dataset "${BCPLUS_DATASET}" \
  --split-manifest "${BCPLUS_SPLIT_MANIFEST}" \
  --split "${AREX_SPLIT}" \
  --limit "${AREX_LIMIT}" \
  --offset "${AREX_OFFSET}" \
  --output-dir "${AREX_OUTPUT_DIR}" \
  --model-base-url "${AREX_BASE_URL}" \
  --model "${AREX_MODEL}" \
  --api-key "${AREX_API_KEY}" \
  --retrieval-url "${AREX_RETRIEVAL_URL}" \
  "${CORPUS_ARGS[@]}" \
  --top-k "${AREX_TOP_K}" \
  --max-turns "${AREX_MAX_TURNS}" \
  --max-tokens "${AREX_MAX_TOKENS}"

EVAL_ARGS=()
if [[ "${AREX_EXACT_MATCH_SMOKE:-0}" == "1" ]]; then
  EVAL_ARGS=(--exact-match-smoke)
else
  : "${AREX_JUDGE_BASE_URL:?set AREX_JUDGE_BASE_URL or set AREX_EXACT_MATCH_SMOKE=1}"
  EVAL_ARGS=(
    --judge-provider "${AREX_JUDGE_PROVIDER:-openai}"
    --judge-base-url "${AREX_JUDGE_BASE_URL}"
    --judge-model "${AREX_JUDGE_MODEL:-Qwen3-32B}"
    --judge-api-key "${AREX_JUDGE_API_KEY:-EMPTY}"
  )
fi

python "${EXPER_DIR}/evaluate.py" \
  --dataset "${BCPLUS_DATASET}" \
  --qrels "${BCPLUS_QRELS}" \
  --runs "${AREX_OUTPUT_DIR}/runs" \
  --output "${AREX_OUTPUT_DIR}/evaluation.json" \
  "${EVAL_ARGS[@]}"

