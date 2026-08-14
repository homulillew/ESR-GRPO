# ESR-GRPO

ESR-GRPO（Evidence-State Routing GRPO）面向长程 Agent Search。代码保存原始证据、当前结论和待解决问题，并根据最终答案所依赖的证据与状态变更，把正的组内优势分配给相关完整动作。

```text
search → open_page → Evidence
                      ↓
                 update_state → TaskState
                      ↓
                 verify_answer ── needs_revision → gaps → 继续检索
                      ↓ supported
                 submit_answer
                      ↓
最终奖励 → GRPO 组内优势 → 动作关系回溯 → 策略 token mask
```

核心状态机、SQLite 账本、动作关系回溯和 CPU 测试均在本项目中独立实现。正式 rollout 与训练通过本地 ECHO/verl 源码运行；固定语料、qrels 和评测格式来自 BrowseComp-Plus。

## 1. 当前代码包含什么

- Evidence：保存完整页面正文、来源、内容哈希、创建动作和上游检索动作。保存后禁止修改和删除。
- TaskState：保存 Evidence finding 目录、当前答案、主要证据、gaps 和验证状态。每次修改都会追加新版本。
- ActionRecord：保存动作类型、合法性、状态版本、Evidence 关系和精确策略 token 范围。
- Agent 环境：实现 `search`、`open_page`、`read_evidence`、`update_state`、`verify_answer` 和 `submit_answer`。
- 动作级信用分配：回溯最终答案、最终证据和已解决 gaps 所依赖的动作。
- ECHO/verl 接口：提供 ESR 工具、最终奖励适配器、分段优势函数和源码钩子。
- BrowseComp-Plus 工具：创建研究切分、转换 parquet、生成 run 文件并计算 Accuracy、Evidence Recall 和 Calibration Error。
- 调试工具：覆盖检索、状态协议、动作信用、rollout metadata、训练日志和 GPU 训练前检查。

## 2. 运行环境

实验和训练统一在 Linux 上运行。Windows 版本只用于下载模型，避免先在 Windows 下载的大模型再次传输。

| 用途 | 环境要求 |
|---|---|
| CPU 单元测试和冒烟实验 | Linux、Python 3.10 |
| BrowseComp-Plus 数据和 Pipeline | Linux、Python 3.10、Java 21、ECHO 检索服务、OpenAI 兼容模型服务 |
| 4B 训练流程验证 | Linux、CUDA 12.8、4 张 GPU、ECHO/verl 环境 |
| 30B 正式训练 | Linux 多 GPU/多节点环境；TP、PP、CP 和 offload 需要按机器实测配置 |
| 模型下载 | Linux Bash 或 Windows PowerShell，Python 3.10+ |

建议在 ECHO 已验证的 Linux 镜像或 Conda 环境中安装训练依赖。不要在另一套 PyTorch/CUDA 环境中单独安装本项目后直接启动训练。

## 3. 目录结构

```text
ESR-GRPO/
├── src/esr_grpo/
│   ├── models.py              # Evidence、TaskState、ActionRecord
│   ├── store.py               # SQLite 追加式账本
│   ├── environment.py         # 动作执行和协议校验
│   ├── credit.py              # 动作关系回溯与 token mask
│   ├── diagnostics.py         # 轨迹、信用和 rollout 审计
│   ├── retrieval.py           # 内存检索器和 ECHO 检索客户端
│   ├── verification.py        # 在线 verify_answer
│   ├── judge.py               # 最终答案 Judge
│   ├── rollout.py             # Benchmark Agent Loop
│   ├── browsecomp.py          # 数据、qrels、run 和指标
│   ├── model_download.py      # 模型目录与分片校验
│   └── integrations/          # ECHO 工具、奖励和优势函数
├── configs/
│   ├── experiment1/           # Pipeline 配置
│   ├── experiment2/           # 状态、Evidence 和信用消融
│   ├── experiment3/           # AREX、GRPO、ECHO、ESR-GRPO 对照
│   └── echo/                  # ECHO 工具注册配置
├── scripts/
│   ├── common/                # 数据准备、ECHO 钩子、CPU 检查
│   ├── experiment1_pipeline/  # 推理、评测和 Pipeline 调试
│   ├── experiment2_analysis/  # 状态协议与信用分析
│   ├── experiment3_training/  # 训练入口和训练期诊断
│   └── model_download/        # Linux/Windows 模型下载入口
├── prompts/                   # ESR Agent 系统提示
├── tests/                     # 单元测试
└── docs/                      # 架构、实验、调试和 ECHO 集成说明
```

## 4. Linux 安装与首次验证

### 4.1 CPU 开发环境

```bash
cd /path/to/ESR-GRPO
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-core.txt
python -m pip install -e .
```

运行完整 CPU 检查：

```bash
bash scripts/common/run_all_cpu_checks.sh
```

该命令依次执行 Python 编译检查、单元测试和确定性冒烟 episode。冒烟输出中的以下字段必须成立：

```json
{
  "answer": "Vector Labs",
  "irrelevant_parallel_open_page_credited": false
}
```

单独运行冒烟实验：

```bash
python -P scripts/experiment1_pipeline/run_smoke_episode.py \
  --output-dir outputs/smoke \
  --overwrite
```

### 4.2 BrowseComp-Plus 依赖

```bash
python -m pip install -r requirements-browsecomp.txt
```

### 4.3 GPU 训练依赖

先安装并验证 ECHO/verl，再在同一 Linux 环境安装本项目。`requirements-train-cu128.txt` 记录本实验使用的 Python/CUDA 包版本。

```bash
python -m pip install -e .
python -m pip install -r requirements-train-cu128.txt
```

FlashAttention、Transformer Engine、Megatron Core 和 SGLang 必须与当前 CUDA、驱动和 ECHO 快照匹配。正式训练前必须运行第 9 节的训练预检。

## 5. 模型下载：Linux 和 Windows

实验使用两个 30B 级模型：

| 用途 | Hugging Face 仓库 | 默认目录名 |
|---|---|---|
| 正式策略模型 | `Qwen/Qwen3-30B-A3B-Instruct-2507` | `Qwen3-30B-A3B-Instruct-2507` |
| 最终答案 Judge | `Qwen/Qwen3-32B` | `Qwen3-32B` |

下载器会先把 `main` 解析为固定 commit，估算剩余磁盘空间，然后使用 `snapshot_download` 断点续传。下载完成后检查 `config.json`、safetensors 分片和索引引用，并写入 `download_manifest.json`。

### 5.1 Linux 下载

```bash
cd /path/to/ESR-GRPO
python3.10 -m venv .venv-download
source .venv-download/bin/activate
python -m pip install -e .
python -m pip install -r requirements-download.txt

export MODEL_ROOT=/data/models
bash scripts/model_download/download_qwen3_30b_a3b_instruct.sh
bash scripts/model_download/download_qwen3_32b_judge.sh
```

一次下载两个模型：

```bash
MODEL_ROOT=/data/models bash scripts/model_download/download_both_large_models.sh
```

### 5.2 Windows 下载

在 PowerShell 中执行：

```powershell
cd "C:\path\to\ESR-GRPO"
py -m venv .venv-download
.\.venv-download\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -r requirements-download.txt

.\scripts\model_download\download_qwen3_30b_a3b_instruct.ps1 -DestinationRoot D:\models
.\scripts\model_download\download_qwen3_32b_judge.ps1 -DestinationRoot D:\models
```

一次下载两个模型：

```powershell
.\scripts\model_download\download_both_large_models.ps1 -DestinationRoot D:\models
```

如果下载中断，重新运行同一命令即可。下载后执行完整文件哈希校验：

```bash
MODEL_ROOT=/data/models bash scripts/model_download/download_qwen3_30b_a3b_instruct.sh \
  --verify-only --verify-lfs-sha256
```

```powershell
.\scripts\model_download\download_qwen3_30b_a3b_instruct.ps1 `
  -DestinationRoot D:\models -VerifyOnly -VerifyLfsSha256
```

私有或限流仓库通过环境变量 `HF_TOKEN` 提供 token。更多参数见 [`scripts/model_download/README.md`](scripts/model_download/README.md)。

## 6. 数据准备

BrowseComp-Plus 官方发布只有一个包含 830 道题的 `test` split。本项目使用显式 manifest 保存研究切分，不把自定义切分称为官方训练集或验证集。

创建并验证切分：

```bash
esr-grpo make-split \
  --dataset data/browsecomp_plus_decrypted.jsonl \
  --output data/splits/v1.json \
  --seed 20260806

esr-grpo validate-split \
  --dataset data/browsecomp_plus_decrypted.jsonl \
  --manifest data/splits/v1.json
```

转换为 ECHO/verl parquet：

```bash
python -P scripts/common/prepare_browsecomp_data.py \
  --dataset data/browsecomp_plus_decrypted.jsonl \
  --split-manifest data/splits/v1.json \
  --system-prompt prompts/esr_system.txt \
  --output-dir data/processed
```

预期产物：

```text
data/processed/
├── train.esr.parquet
├── validation.esr.parquet
└── test.esr.parquet
```

## 7. 实验一：完整 Pipeline

### 7.1 运行前检查

先检查数据、切分、qrels、语料和服务端点：

```bash
python -P scripts/experiment1_pipeline/preflight.py \
  --dataset data/browsecomp_plus_decrypted.jsonl \
  --split-manifest data/splits/v1.json \
  --qrels /path/to/BrowseComp-Plus/topics-qrels/qrel_evidence.txt \
  --corpus-jsonl /path/to/corpus.jsonl \
  --retrieval-health-url http://127.0.0.1:8000/health \
  --policy-models-url http://127.0.0.1:8002/v1/models \
  --verifier-models-url http://127.0.0.1:8001/v1/models \
  --judge-models-url http://127.0.0.1:8003/v1/models \
  --output outputs/exp1/preflight.json \
  --strict
```

确认检索结果中的 docid 能读回完整正文：

```bash
python -P scripts/experiment1_pipeline/check_retrieval_roundtrip.py \
  --retrieval-url http://127.0.0.1:8000 \
  --query "example entity and event" \
  --top-k 5 \
  --open-count 3
```

### 7.2 Benchmark 推理

```bash
python -P scripts/experiment1_pipeline/run_benchmark.py \
  --dataset data/browsecomp_plus_decrypted.jsonl \
  --split-manifest data/splits/v1.json \
  --split validation \
  --limit 100 \
  --retrieval-url http://127.0.0.1:8000 \
  --model-base-url http://127.0.0.1:8002/v1 \
  --model BAAI/AREX-Turbo \
  --verifier-base-url http://127.0.0.1:8001/v1 \
  --verifier-model Qwen3-32B \
  --output-dir outputs/exp1/benchmark
```

每道题保存：

```text
outputs/exp1/benchmark/
├── stores/<query_id>.sqlite       # Evidence、TaskState 和动作账本
├── trajectory_<query_id>.json     # 模型消息与工具结果
└── runs/<query_id>.json           # BrowseComp-Plus run 格式
```

已存在的 SQLite 文件会被跳过，因此中断后可直接重跑。

### 7.3 最终评测

```bash
python -P scripts/experiment1_pipeline/evaluate_runs.py \
  --dataset data/browsecomp_plus_decrypted.jsonl \
  --qrels /path/to/BrowseComp-Plus/topics-qrels/qrel_evidence.txt \
  --runs outputs/exp1/benchmark/runs \
  --judge-base-url http://127.0.0.1:8003/v1 \
  --judge-model Qwen3-32B \
  --output outputs/exp1/evaluation.json
```

在线 `verify_answer` 只负责决定继续搜索还是允许提交。最终 Judge 比较提交答案和标准答案，并提供最终奖励和 Accuracy。

## 8. 实验二：状态与动作信用分析

汇总全部 SQLite 轨迹：

```bash
python -P scripts/experiment2_analysis/analyze_experiments.py \
  --stores outputs/exp1/benchmark/stores \
  --output outputs/exp2/analysis.json
```

逐条检查状态协议：

```bash
python -P scripts/experiment2_analysis/audit_state_protocol.py \
  --stores outputs/exp1/benchmark/stores \
  --output outputs/exp2/protocol_audit.json \
  --strict
```

查看单条轨迹为什么选择某些动作：

```bash
python -P scripts/experiment2_analysis/inspect_credit_trace.py \
  outputs/exp1/benchmark/stores/<query_id>.sqlite \
  --selected-only \
  --output outputs/exp2/<query_id>.credit.json
```

人工标签使用动作 ID：

```json
{"episode_id":"query-id","selected_action_ids":["a3","a7"],"annotator":"name","notes":""}
```

检查标签并计算 precision、recall 和 F1：

```bash
python -P scripts/experiment2_analysis/validate_credit_labels.py \
  --stores outputs/exp1/benchmark/stores \
  --labels data/annotations/credit_labels.jsonl \
  --output outputs/exp2/credit_label_audit.json \
  --strict
```

## 9. 实验三：ECHO/verl 训练

### 9.1 安装 ECHO 钩子

所有命令在 Linux 上执行。先设置 ECHO 源码目录：

```bash
export ECHO_ROOT=/path/to/ECHO
export PYTHONPATH="$PWD/src:$ECHO_ROOT:${PYTHONPATH:-}"
```

只检查当前 ECHO 快照是否仍能应用钩子：

```bash
python scripts/common/check_echo_compat.py --echo-root "$ECHO_ROOT"
python scripts/common/install_echo_hooks.py --echo-root "$ECHO_ROOT"
```

确认检查通过后显式写入钩子：

```bash
python scripts/common/install_echo_hooks.py --echo-root "$ECHO_ROOT" --apply
```

脚本修改 ECHO 的 `core_algos.py`、`ray_trainer.py` 和 `tool_agent_loop.py`，并在同目录创建 `.esr-grpo.bak` 备份。

### 9.2 训练前检查

```bash
export MODEL_PATH=/data/models/Qwen3.5-4B-Instruct
export TRAIN_FILE=$PWD/data/processed/train.esr.parquet
export VAL_FILE=$PWD/data/processed/validation.esr.parquet

python scripts/experiment3_training/preflight_training.py \
  --echo-root "$ECHO_ROOT" \
  --model-path "$MODEL_PATH" \
  --train-file "$TRAIN_FILE" \
  --val-file "$VAL_FILE" \
  --tool-config configs/echo/esr_tools.yaml \
  --output-dir outputs/exp3 \
  --required-gpus 4 \
  --strict
```

预检会检查 Python 包、CUDA GPU、模型分片、parquet 字段、ECHO 钩子、工具配置和输出目录。

### 9.3 单节点 4 GPU Pipeline 验证

```bash
export ESR_RETRIEVAL_URL=http://127.0.0.1:8000
export ESR_VERIFIER_BASE_URL=http://127.0.0.1:8001/v1
export ESR_VERIFIER_MODEL=Qwen3-32B
export ESR_VERIFIER_API_KEY=EMPTY
export ESR_STORE_DIR=$PWD/outputs/exp3/stores
export EXPERIMENT_NAME=qwen4b-bcp-esr-pipeline

bash scripts/experiment3_training/run_esr_4gpu.sh
```

`run_esr_4gpu.sh` 默认再次执行严格预检。它用于验证 rollout、最终奖励和训练更新能否连通，不代表 30B 正式训练资源配置。

### 9.4 公开基线入口

```bash
bash scripts/experiment3_training/run_baseline_4node.sh grpo
bash scripts/experiment3_training/run_baseline_4node.sh echo
```

该入口调用 ECHO 上游的 4 节点 × 8 GPU 脚本。AREX-Turbo 是推理基线，使用实验一的 `run_benchmark.py`。

## 10. 调试顺序

### 10.1 单条 episode

```bash
python scripts/experiment1_pipeline/inspect_episode.py \
  outputs/exp1/benchmark/stores/<query_id>.sqlite \
  --output outputs/debug/<query_id>.json
```

检查顺序：Evidence 是否完整保存、TaskState 是否覆盖所有 Evidence、gaps 是否只由验证器修改、最终提交是否合法、信用动作是否都有 token span。

### 10.2 ECHO rollout metadata

```bash
python scripts/experiment3_training/inspect_rollout_metadata.py \
  rollout_segments.jsonl \
  --output outputs/debug/rollout_audit.json \
  --strict
```

该脚本检查 mask 长度、非策略 token 信用、非最终分段奖励、最终分段数量、合法提交空 mask 和 overlong 奖励。

### 10.3 优势函数

```bash
python scripts/experiment3_training/debug_advantage.py
```

该脚本构造一组成功和失败 rollout，检查只有成功轨迹中被动作 mask 选中的 token 获得正优势。

### 10.4 训练日志

```bash
python scripts/experiment3_training/inspect_training_log.py \
  outputs/exp3/train.log \
  --output outputs/debug/training_log.json \
  --strict
```

输出包含 CUDA OOM、NCCL、Ray worker、Traceback、NaN/Inf 和关键训练指标摘要。

## 11. 关键环境变量

| 变量 | 作用 |
|---|---|
| `ECHO_ROOT` | 本地 ECHO/verl 源码目录 |
| `MODEL_PATH` | 策略模型本地目录 |
| `TRAIN_FILE`、`VAL_FILE` | ECHO parquet 路径 |
| `ESR_RETRIEVAL_URL` | 固定语料检索服务根地址 |
| `ESR_VERIFIER_BASE_URL` | 在线验证器的 OpenAI 兼容地址 |
| `ESR_VERIFIER_MODEL` | 在线验证器模型名 |
| `ESR_VERIFIER_API_KEY` | 在线验证器 API key；本地服务可用 `EMPTY` |
| `ESR_STORE_DIR` | 每条 rollout 的 SQLite 账本目录 |
| `HF_TOKEN` | Hugging Face 下载 token，可选 |
| `MODEL_ROOT` | Linux 模型下载根目录 |

训练入口的 batch size、响应长度、TP/PP/CP、offload 和保存频率可通过 `run_esr_4gpu.sh` 中列出的同名环境变量覆盖。

## 12. 实现约束

- `open_page` 在截断模型可见结果前保存完整页面正文。
- 新 Evidence 必须进入最新 TaskState 的 finding 目录。
- 修改已有 finding 前必须重新读取对应 Evidence，或仍处于首次打开页面的可见阶段。
- `update_state` 保留现有 gaps，并把验证状态重置为 `unverified`。
- 只有 `verify_answer` 可以创建或解决 gaps。
- `submit_answer` 要求最新状态已通过验证、gaps 为空且主要证据非空。
- `update_state`、`verify_answer` 和 `submit_answer` 串行执行；检索和只读动作允许并行。
- 最终奖励只写入最终 rollout 分段。
- 只有合法成功且组内优势为正的轨迹获得 ESR 信用。
- 一个动作的全部策略 token 整体进入或退出 mask；普通思考 token 不自动获得优势。

## 13. 测试状态与边界

当前 CPU 测试覆盖状态不可修改、Evidence 覆盖、finding 更新、gaps、提交条件、并行动作范围、结构信用、BrowseComp 指标、ECHO metadata 和模型分片校验。

本仓库不包含完整 ECHO、BrowseComp-Plus 语料、检索索引或模型权重。正式 GPU 训练仍需在目标 Linux 集群上验证显存、并行策略、服务吞吐和 ECHO 快照兼容性。

进一步阅读：

- [架构说明](docs/architecture.md)
- [实验执行说明](docs/experiments.md)
- [逐实验执行手册](docs/实验执行手册.md)
- [AREX-Turbo 原生 BC-Plus 推理与评测](exper/arex_bcplus_native/README.md)
- [调试清单](docs/debugging.md)
- [ECHO/verl 集成](docs/echo_integration.md)
- [模型下载说明](scripts/model_download/README.md)
