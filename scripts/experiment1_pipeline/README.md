# 实验一脚本

- `preflight.py`：检查数据、split manifest、qrels、语料和服务端点。
- `check_retrieval_roundtrip.py`：检查 search → open_page 的 docid 与正文读取。
- `run_smoke_episode.py`：离线状态与信用冒烟实验。
- `run_benchmark.py`：真实模型 Agent Loop。
- `evaluate_runs.py`：最终 Accuracy、Evidence Recall 和 Calibration Error。
- `inspect_episode.py`：逐动作检查单个 SQLite 账本。

推荐顺序：

```bash
python -P scripts/experiment1_pipeline/preflight.py --help
python -P scripts/experiment1_pipeline/check_retrieval_roundtrip.py --help
python -P scripts/experiment1_pipeline/run_smoke_episode.py --output-dir outputs/smoke --overwrite
python -P scripts/experiment1_pipeline/run_benchmark.py --help
python -P scripts/experiment1_pipeline/evaluate_runs.py --help
```
