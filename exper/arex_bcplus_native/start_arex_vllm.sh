#!/usr/bin/env bash
set -euo pipefail

: "${AREX_MODEL_PATH:?set AREX_MODEL_PATH to the local BAAI--AREX-Turbo directory}"

AREX_SERVED_MODEL="${AREX_SERVED_MODEL:-AREX-Turbo}"
AREX_PORT="${AREX_PORT:-8002}"
AREX_TP="${AREX_TP:-1}"
AREX_MAX_MODEL_LEN="${AREX_MAX_MODEL_LEN:-65536}"
AREX_GPU_MEMORY_UTILIZATION="${AREX_GPU_MEMORY_UTILIZATION:-0.9}"

# 注：--reasoning-parser qwen3 和 --language-model-only 是较新版本 vLLM 才有的参数。
# 镜像里的 vLLM 0.11.3 不支持，已移除。AREX baseline 直接解析 assistant content 中的
# XML 工具调用（见 protocol.py 的 extract_reasoning），不依赖 reasoning 分字段或多模态跳过。
exec vllm serve "${AREX_MODEL_PATH}" \
  --served-model-name "${AREX_SERVED_MODEL}" \
  --host 0.0.0.0 \
  --port "${AREX_PORT}" \
  --tensor-parallel-size "${AREX_TP}" \
  --max-model-len "${AREX_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${AREX_GPU_MEMORY_UTILIZATION}"

