# AREX-Turbo 原生 BC-Plus 推理与评测

这个目录独立运行 AREX-Turbo 4B，不调用 ESR-GRPO 的 Evidence、TaskState、外部 Verifier 或动作信用代码。模型使用 AREX 官方 XML 工具协议：

```text
search → visit → update_context（按需）→ finish
```

AREX 官方 BrowseComp 提示还包含 `google_scholar`。BrowseComp-Plus 要求所有检索结果来自固定语料，因此本实验明确禁用该工具。除此之外，采样参数、XML 格式、研究流程和 `finish` 字段均沿用 AREX-Turbo 官方 inference 文件。

## 1. 文件说明

| 文件 | 作用 |
|---|---|
| `prompts.py` | AREX 原生提示和四工具 schema |
| `protocol.py` | 解析 `<tool_call>` XML，检查单工具调用和参数 |
| `clients.py` | 调用 AREX OpenAI 兼容服务和 ECHO BC-Plus 检索服务 |
| `runner.py` | 单题 Agent Loop，实现 `update_context` 上下文重建 |
| `run_arex_bcplus.py` | 按 split 逐题推理、保存轨迹、失败隔离和断点续跑 |
| `preflight.py` | 检查数据、split、检索服务和模型服务 |
| `evaluate.py` | 使用 BC-Plus 官方风格 Judge 提示计算 Accuracy、Evidence Recall 和 Calibration Error |
| `inspect_trajectory.py` | 把完整轨迹转换成便于人工阅读的时间线 |
| `start_arex_vllm.sh` | 启动 AREX-Turbo vLLM 服务 |
| `start_bcplus_bm25.sh` | 启动固定语料 BM25 服务 |
| `run_experiment.sh` | 预检、1 题 smoke、目标子集推理和评测 |

## 2. 轨迹中保存的内容

每道题保存：

- 官方 system prompt 和 user prompt。
- 每轮模型服务的完整原始 JSON。
- `reasoning_content`；服务没有单独返回时，从 `<think>` 中提取。
- 模型原始 XML。
- 解析后的工具名称和参数。
- 检索或访问结果。
- 每轮耗时。
- `update_context` 内容和上下文重建位置。
- 最终 answer、evidences 和 confidence。
- 全部检索 docid 和实际访问 docid。
- 异常类型、消息和 traceback。

输出目录：

```text
outputs/validation/
├── preflight.json
├── run_summary.json
├── trajectories/<query_id>.json
├── runs/<query_id>.json
├── failed/<query_id>.json
├── judge_cache/<query_id>.json
└── evaluation.json
```

只有 trajectory 状态为 `completed` 且 run 文件存在时，重跑才会跳过该题。失败题使用 `--retry-failed` 重试；`--overwrite` 会重新运行选中的全部题目。

## 3. 环境

推荐为 AREX 推理单独建立环境。模型配置使用 Qwen3.5，并记录了 Transformers 5.2.0；不要直接假设旧的 ECHO 训练环境能够加载它。

```bash
cd /data/AgentSearchAssets/code/ESR-GRPO/exper/arex_bcplus_native

python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -U vllm
```

BM25 服务使用 ECHO 脚本，还需要其依赖中的 pandas、rank-bm25、FastAPI 和 Uvicorn。

## 4. 设置路径

```bash
export ASSET_ROOT=/data/AgentSearchAssets
export ESR_ROOT="$ASSET_ROOT/code/ESR-GRPO"
export ECHO_ROOT="$ASSET_ROOT/code/ECHO"
export BCP_ROOT="$ASSET_ROOT/BrowseComp-Plus"
export MODEL_ROOT="$ASSET_ROOT/models"

export BCPLUS_DATASET="$BCP_ROOT/data/prepared/browsecomp_plus_decrypted.jsonl"
export BCPLUS_SPLIT_MANIFEST="$BCP_ROOT/data/splits/v1.json"
export BCPLUS_QRELS="$BCP_ROOT/topics-qrels/qrel_evidence.txt"
export BCPLUS_CORPUS_FILE="$BCP_ROOT/data/prepared/corpus.parquet"
```

如果代码目录不同，只修改这里的根目录。

## 5. 启动固定语料检索服务

```bash
cd "$ESR_ROOT/exper/arex_bcplus_native"
export BCPLUS_CACHE_FILE="$ASSET_ROOT/caches/browsecomp_bm25_cache.pkl"
export BCPLUS_PORT=8000

bash start_bcplus_bm25.sh 2>&1 | tee retrieval.log
```

第一次运行会构建 ECHO BM25 缓存。

检查：

```bash
curl -fsS http://127.0.0.1:8000/health
```

## 6. 启动 AREX-Turbo

官方命令使用 `--reasoning-parser qwen3` 和 `--language-model-only`。本实验直接解析 assistant content 中的 AREX XML，不需要 `--enable-auto-tool-choice` 或 tool-call parser。

```bash
export AREX_MODEL_PATH="$MODEL_ROOT/BAAI--AREX-Turbo"
export AREX_SERVED_MODEL=AREX-Turbo
export AREX_PORT=8002
export AREX_TP=1
export AREX_MAX_MODEL_LEN=65536

CUDA_VISIBLE_DEVICES=0 \
bash start_arex_vllm.sh 2>&1 | tee arex_vllm.log
```

模型支持 262144 token，但第一轮建议从 65536 开始，确认显存和吞吐后再提高。

检查：

```bash
curl -fsS http://127.0.0.1:8002/v1/models
```

## 7. Judge 服务

评测使用 OpenAI 兼容的 Judge 服务。`evaluate.py` 通过 `/chat/completions` 调用 Judge，任何兼容该接口的服务都能接入。

### 7.1 使用 GLM API（推荐，无需下载大模型）

Qwen3-32B 约 64 GB，仅在评测阶段使用，下载和占用显存成本较高。改用服务器自带的 GLM API（智谱云或本地部署）即可完成评测。

智谱云示例：

```bash
export AREX_JUDGE_BASE_URL=https://open.bigmodel.cn/api/paas/v4
export AREX_JUDGE_API_KEY=你的智谱 key
export AREX_JUDGE_MODEL=glm-4          # 或 glm-4-plus / glm-4-air
```

服务器本地部署的 GLM（vLLM）示例：

```bash
CUDA_VISIBLE_DEVICES=1,2 \
vllm serve "$MODEL_ROOT/glm-4" \
  --served-model-name glm-4 \
  --host 0.0.0.0 \
  --port 8003 \
  --tensor-parallel-size 2

export AREX_JUDGE_BASE_URL=http://127.0.0.1:8003/v1
export AREX_JUDGE_API_KEY=EMPTY
export AREX_JUDGE_MODEL=glm-4
```

`setup_env.sh` 会自动判断：配置了 `AREX_JUDGE_BASE_URL` 和 `AREX_JUDGE_API_KEY` 时使用真 Judge，未配置时自动回退到精确匹配（见 7.2）。

### 7.2 精确匹配回退

未配置 GLM 时，`setup_env.sh` 自动设置 `AREX_EXACT_MATCH_SMOKE=1`，用精确匹配跑通评测文件流，但该结果不能作为正式 Accuracy：

```bash
# 不 export Judge 变量，直接 source
source setup_env.sh    # 自动 AREX_EXACT_MATCH_SMOKE=1
```

### 7.3 Judge 模型差异说明

BrowseComp-Plus 官方使用 Qwen3-32B 作为 Judge。GLM-4 与 Qwen3-32B 对“语义等价”的判断标准不同，因此用 GLM 评测的结果不能与官方数字直接横向对比。对“实验一：跑通 benchmark”的目标足够；正式对比时，所有方法必须在相同 Judge 下重跑。

## 8. 一键运行

### 8.1 用 setup_env.sh（推荐）

`setup_env.sh` 固化了方案 A 的路径布局（outputs 放 `$ASSET_ROOT/outputs/`），并自动处理 Judge 回退。source 后直接运行：

```bash
cd "$ESR_ROOT/exper/arex_bcplus_native"

# 如需用 GLM Judge，在 source 前 export（见 §7.1）；不配则自动回退精确匹配
export AREX_JUDGE_BASE_URL=https://open.bigmodel.cn/api/paas/v4
export AREX_JUDGE_API_KEY=你的智谱 key
export AREX_JUDGE_MODEL=glm-4

source setup_env.sh

# 默认 AREX_LIMIT=1，先跑通一题
bash run_experiment.sh 2>&1 | tee "$AREX_OUTPUT_DIR/experiment.log"
```

逐步扩大：

```bash
export AREX_LIMIT=10  AREX_OUTPUT_DIR="$ASSET_ROOT/outputs/arex_bcplus_native/validation_10"
bash run_experiment.sh

export AREX_LIMIT=100 AREX_OUTPUT_DIR="$ASSET_ROOT/outputs/arex_bcplus_native/validation_100"
bash run_experiment.sh
```

当前研究切分 validation 共 124 题。全量验证集使用 `AREX_LIMIT=124`。

### 8.2 手动设置环境变量

如不使用 `setup_env.sh`，需手动 export 全部变量：

```bash
cd "$ESR_ROOT/exper/arex_bcplus_native"

export BCPLUS_DATASET="$BCP_ROOT/data/prepared/browsecomp_plus_decrypted.jsonl"
export BCPLUS_SPLIT_MANIFEST="$BCP_ROOT/data/splits/v1.json"
export BCPLUS_QRELS="$BCP_ROOT/topics-qrels/qrel_evidence.txt"
export BCPLUS_CORPUS_FILE="$BCP_ROOT/data/prepared/corpus.parquet"

export AREX_BASE_URL=http://127.0.0.1:8002/v1
export AREX_MODEL=AREX-Turbo
export AREX_API_KEY=EMPTY
export AREX_RETRIEVAL_URL=http://127.0.0.1:8000

# Judge：用 GLM（见 §7.1），或设 AREX_EXACT_MATCH_SMOKE=1 跑通链路
export AREX_JUDGE_BASE_URL=https://open.bigmodel.cn/api/paas/v4
export AREX_JUDGE_MODEL=glm-4
export AREX_JUDGE_API_KEY=你的智谱 key

export AREX_SPLIT=validation
export AREX_LIMIT=10
export AREX_OUTPUT_DIR="$ASSET_ROOT/outputs/arex_bcplus_native/validation_10"

mkdir -p "$AREX_OUTPUT_DIR"
bash run_experiment.sh 2>&1 | tee "$AREX_OUTPUT_DIR/experiment.log"
```

脚本先运行一题 smoke。只有该题成功调用 `finish` 后，才继续目标子集。

## 9. 分步运行

预检：

```bash
python preflight.py \
  --dataset "$BCPLUS_DATASET" \
  --split-manifest "$BCPLUS_SPLIT_MANIFEST" \
  --qrels "$BCPLUS_QRELS" \
  --corpus-file "$BCPLUS_CORPUS_FILE" \
  --model-base-url "$AREX_BASE_URL" \
  --retrieval-url "$AREX_RETRIEVAL_URL" \
  --output outputs/preflight.json \
  --strict
```

指定一题：

```bash
python run_arex_bcplus.py \
  --dataset "$BCPLUS_DATASET" \
  --split-manifest "$BCPLUS_SPLIT_MANIFEST" \
  --query-id '<query_id>' \
  --output-dir outputs/single \
  --model-base-url "$AREX_BASE_URL" \
  --retrieval-url "$AREX_RETRIEVAL_URL" \
  --corpus-file "$BCPLUS_CORPUS_FILE" \
  --fail-fast
```

重试失败题：

```bash
python run_arex_bcplus.py \
  --dataset "$BCPLUS_DATASET" \
  --split-manifest "$BCPLUS_SPLIT_MANIFEST" \
  --split validation \
  --limit 10 \
  --output-dir outputs/validation_10 \
  --model-base-url "$AREX_BASE_URL" \
  --retrieval-url "$AREX_RETRIEVAL_URL" \
  --corpus-file "$BCPLUS_CORPUS_FILE" \
  --retry-failed
```

评测：

```bash
python evaluate.py \
  --dataset "$BCPLUS_DATASET" \
  --qrels "$BCPLUS_QRELS" \
  --runs outputs/validation_10/runs \
  --judge-base-url "$AREX_JUDGE_BASE_URL" \
  --judge-model "$AREX_JUDGE_MODEL" \
  --output outputs/validation_10/evaluation.json
```

## 10. 查看轨迹

完整轨迹直接查看 `trajectories/<query_id>.json`。生成简洁时间线：

```bash
python inspect_trajectory.py \
  outputs/validation_10/trajectories/<query_id>.json \
  --include-tool-output \
  --output outputs/validation_10/<query_id>.inspection.json
```

人工检查重点：

- 第一轮是否调用 `search`。
- `visit` 是否只读取 search 返回的 URI。
- 是否尝试调用固定语料之外的工具。
- `update_context` 是否保留关键事实和 URI。
- `finish` 前是否明确核对约束。
- evidence URI 是否对应实际访问文档。
- answer 是否简洁，confidence 是否在 0% 至 100%。

## 11. 本地测试

测试不需要模型、网络和 GPU：

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

## 12. 结果解释边界

这个实验使用 AREX 原生状态动作和采样参数，但检索环境改为 BC-Plus 固定语料，并移除了 `google_scholar`。因此结果应报告为：

> AREX-Turbo 原生 Agent 协议在 BrowseComp-Plus 固定语料环境中的结果。

不要把该结果与 AREX 模型页上的公开网络 BrowseComp 数字直接等同。正式对比时，所有方法必须使用相同的 BC-Plus split、语料、检索器、top-k 和 Judge。
