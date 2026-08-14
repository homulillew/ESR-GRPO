#!/usr/bin/env bash
set -euo pipefail

: "${ECHO_ROOT:?set ECHO_ROOT to the ECHO source directory}"
: "${BCPLUS_CORPUS_FILE:?set BCPLUS_CORPUS_FILE to corpus.parquet}"

BCPLUS_CACHE_FILE="${BCPLUS_CACHE_FILE:-./cache/browsecomp_bm25_cache.pkl}"
BCPLUS_PORT="${BCPLUS_PORT:-8000}"
BCPLUS_POOL_WORKERS="${BCPLUS_POOL_WORKERS:-8}"

mkdir -p "$(dirname "${BCPLUS_CACHE_FILE}")"

exec python "${ECHO_ROOT}/examples/sglang_multiturn/browsecomp_retrieval_server.py" \
  --mode bm25 \
  --corpus_file "${BCPLUS_CORPUS_FILE}" \
  --bm25_cache "${BCPLUS_CACHE_FILE}" \
  --pool_workers "${BCPLUS_POOL_WORKERS}" \
  --host 0.0.0.0 \
  --port "${BCPLUS_PORT}"

