# 实验二：状态与信用分析

本目录保存实验二统计程序及其定向调试入口。

- `analyze_experiments.py`：汇总状态使用、非法动作、Evidence 回读和信用密度。
- `audit_state_protocol.py`：逐条检查 TaskState、Evidence、动作账本和提交条件。
- `inspect_credit_trace.py`：检查一条轨迹为什么选中某些动作，以及生成了哪些 token mask。
- `validate_credit_labels.py`：检查人工标签中的动作 ID，并计算结构路由的 precision、recall 和 F1。

示例：

```bash
python scripts/experiment2_analysis/audit_state_protocol.py --stores outputs/episodes --strict
python scripts/experiment2_analysis/inspect_credit_trace.py outputs/episodes/example.sqlite --selected-only
python scripts/experiment2_analysis/validate_credit_labels.py --stores outputs/episodes --labels data/credit_labels.jsonl --strict
```
