# 实验三：训练与对照

本目录保存四条训练主线的启动入口及训练期调试工具。

- `run_esr_4gpu.sh`：单节点 4 GPU ESR-GRPO Pipeline 验证。
- `run_baseline_4node.sh`：多节点正式基线模板。
- `preflight_training.py`：训练前检查 parquet、模型目录、ECHO 钩子、GPU、依赖和服务。
- `inspect_rollout_metadata.py`：检查逐段 rollout 的信用 mask 和最终奖励。
- `inspect_training_log.py`：提取 OOM、NCCL/Ray 错误、非有限数值和关键指标。
- `debug_advantage.py`：CPU 上验证两条 rollout 的正优势和 mask 交集。

先运行：

```bash
python scripts/experiment3_training/preflight_training.py \
  --echo-root "$ECHO_ROOT" --model-path "$MODEL_PATH" \
  --train-file "$TRAIN_FILE" --val-file "$VAL_FILE" \
  --tool-config configs/echo/esr_tools.yaml \
  --output-dir outputs/exp3 --required-gpus 4 --strict
python scripts/experiment3_training/debug_advantage.py
```
