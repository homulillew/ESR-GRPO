#!/usr/bin/env bash
# 方案 A：outputs 放在 $ASSET_ROOT/outputs/，与代码分离。
#
# 用法：
#   source setup_env.sh                 # 默认 ASSET_ROOT=/data/AgentSearchAssets
#   ASSET_ROOT=/other/path source setup_env.sh
#
# source 后即可直接启动服务和一键运行：
#   bash start_bcplus_bm25.sh &
#   CUDA_VISIBLE_DEVICES=0 bash start_arex_vllm.sh &
#   bash run_experiment.sh

# ASSET_ROOT 默认指向服务器存储根目录；如已设置则沿用。
# 服务器布局：$ASSET_ROOT/{ESR-GRPO,BrowseComp-Plus,models,caches,outputs}
ASSET_ROOT="${ASSET_ROOT:-/data1/ESR-GRPO}"

# 代码目录：ESR-GRPO 主项目（含 ECHO 训练框架子目录）。
ESR_ROOT="$ASSET_ROOT/ESR-GRPO"
ECHO_ROOT="$ESR_ROOT/ECHO"

# 数据目录：BrowseComp-Plus 题目、语料、切分、qrels、索引。
BCP_ROOT="$ASSET_ROOT/BrowseComp-Plus"
BCPLUS_DATASET="$BCP_ROOT/data/prepared/browsecomp_plus_decrypted.jsonl"
BCPLUS_SPLIT_MANIFEST="$BCP_ROOT/data/splits/v1.json"
BCPLUS_QRELS="$BCP_ROOT/topics-qrels/qrel_evidence.txt"
BCPLUS_CORPUS_FILE="$BCP_ROOT/data/prepared/corpus.parquet"

# 模型目录。
MODEL_ROOT="$ASSET_ROOT/models"

# 运行时缓存：BM25 服务首次启动时生成，删后可重建。
BCPLUS_CACHE_FILE="$ASSET_ROOT/caches/browsecomp_bm25_cache.pkl"

# 输出目录：方案 A，统一放在 assets 根下，与代码分离。
# 默认 validation_1，逐步扩大时改 AREX_LIMIT 和 AREX_OUTPUT_DIR。
AREX_OUTPUT_DIR="${AREX_OUTPUT_DIR:-$ASSET_ROOT/outputs/arex_bcplus_native/validation_1}"

# 服务端点：BM25 检索（8000）、AREX-Turbo vLLM（8002）、Judge（8003，可选）。
BCPLUS_PORT="${BCPLUS_PORT:-8000}"
AREX_PORT="${AREX_PORT:-8002}"

AREX_BASE_URL="http://127.0.0.1:${AREX_PORT}/v1"
AREX_MODEL="AREX-Turbo"
AREX_API_KEY="EMPTY"
AREX_RETRIEVAL_URL="http://127.0.0.1:${BCPLUS_PORT}"

# AREX-Turbo vLLM 参数。
AREX_MODEL_PATH="$MODEL_ROOT/BAAI--AREX-Turbo"
AREX_SERVED_MODEL="AREX-Turbo"
AREX_TP="${AREX_TP:-1}"
AREX_MAX_MODEL_LEN="${AREX_MAX_MODEL_LEN:-65536}"
AREX_GPU_MEMORY_UTILIZATION="${AREX_GPU_MEMORY_UTILIZATION:-0.9}"

# Judge：不下载 Qwen3-32B（~64GB），改用外部 API。支持两种格式：
#
# 1) OpenAI 兼容（GLM、智谱云、one-api 网关等）：
#    export AREX_JUDGE_PROVIDER=openai
#    export AREX_JUDGE_BASE_URL=https://open.bigmodel.cn/api/paas/v4
#    export AREX_JUDGE_API_KEY=你的key
#    export AREX_JUDGE_MODEL=glm-4
#
# 2) Anthropic 原生 API（Claude 网关）：
#    export AREX_JUDGE_PROVIDER=anthropic
#    export AREX_JUDGE_BASE_URL=https://api.anthropic.com
#    export AREX_JUDGE_API_KEY=你的anthropic key
#    export AREX_JUDGE_MODEL=claude-3-5-sonnet-20241022
#
# 未配置 base URL 或 API key 时，自动回退到精确匹配，只跑通评测链路，结果不作正式 Accuracy。
AREX_JUDGE_PROVIDER="${AREX_JUDGE_PROVIDER:-openai}"
AREX_JUDGE_BASE_URL="${AREX_JUDGE_BASE_URL:-}"
AREX_JUDGE_MODEL="${AREX_JUDGE_MODEL:-glm-4}"
AREX_JUDGE_API_KEY="${AREX_JUDGE_API_KEY:-}"
if [ -z "$AREX_JUDGE_BASE_URL" ] || [ -z "$AREX_JUDGE_API_KEY" ]; then
  AREX_EXACT_MATCH_SMOKE=1
  _JUDGE_MODE="精确匹配（未配置 Judge，结果不作正式 Accuracy）"
else
  AREX_EXACT_MATCH_SMOKE="${AREX_EXACT_MATCH_SMOKE:-0}"
  _JUDGE_MODE="$AREX_JUDGE_PROVIDER Judge: $AREX_JUDGE_MODEL"
fi

# 推理参数：沿用 AREX-Turbo 官方 inference 配置。
AREX_SPLIT="${AREX_SPLIT:-validation}"
AREX_LIMIT="${AREX_LIMIT:-1}"
AREX_OFFSET="${AREX_OFFSET:-0}"
AREX_TOP_K="${AREX_TOP_K:-10}"
AREX_MAX_TURNS="${AREX_MAX_TURNS:-100}"
AREX_MAX_TOKENS="${AREX_MAX_TOKENS:-8192}"
AREX_RUN_SMOKE="${AREX_RUN_SMOKE:-1}"

# 导出供 start_bcplus_bm25.sh / start_arex_vllm.sh / run_experiment.sh 读取。
export ASSET_ROOT ESR_ROOT ECHO_ROOT BCP_ROOT MODEL_ROOT
export BCPLUS_DATASET BCPLUS_SPLIT_MANIFEST BCPLUS_QRELS BCPLUS_CORPUS_FILE
export BCPLUS_CACHE_FILE BCPLUS_PORT
export AREX_MODEL_PATH AREX_SERVED_MODEL AREX_PORT AREX_TP
export AREX_MAX_MODEL_LEN AREX_GPU_MEMORY_UTILIZATION
export AREX_BASE_URL AREX_MODEL AREX_API_KEY AREX_RETRIEVAL_URL
export AREX_JUDGE_PROVIDER AREX_JUDGE_BASE_URL AREX_JUDGE_MODEL AREX_JUDGE_API_KEY AREX_EXACT_MATCH_SMOKE
export AREX_SPLIT AREX_LIMIT AREX_OFFSET AREX_TOP_K
export AREX_MAX_TURNS AREX_MAX_TOKENS AREX_RUN_SMOKE AREX_OUTPUT_DIR

# 切到实验目录，方便直接运行脚本。
cd "$ESR_ROOT/exper/arex_bcplus_native" || {
  echo "setup_env.sh: 实验目录不存在: $ESR_ROOT/exper/arex_bcplus_native" >&2
  return 1 2>/dev/null || exit 1
}

echo "ASSET_ROOT=$ASSET_ROOT"
echo "ESR_ROOT=$ESR_ROOT"
echo "ECHO_ROOT=$ECHO_ROOT"
echo "AREX_OUTPUT_DIR=$AREX_OUTPUT_DIR"
echo "Judge 模式: $_JUDGE_MODE"
echo "  (AREX_EXACT_MATCH_SMOKE=$AREX_EXACT_MATCH_SMOKE  1=精确匹配跑通链路,0=用真 Judge)"
echo "环境已就绪。启动服务："
echo "  bash start_bcplus_bm25.sh &"
echo "  CUDA_VISIBLE_DEVICES=0 bash start_arex_vllm.sh &"
echo "  bash run_experiment.sh"
