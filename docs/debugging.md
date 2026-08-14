# 调试清单

## CPU 协议层

```bash
bash scripts/common/run_all_cpu_checks.sh
```

测试失败时按顺序检查：

1. Evidence 是否在模型可见结果截断前写入 SQLite；
2. `update_state` 是否覆盖所有 Evidence；
3. 修改旧 finding 前是否有 `read_evidence`；
4. 验证后是否又创建了新 Evidence；
5. gaps 是否只由 `verify_answer` 改变；
6. 并行动作 token 范围是否独立；
7. 无关页面的动作 ID 是否仍进入最终信用集合。

单条或批量轨迹可以直接运行：

```bash
python scripts/experiment1_pipeline/inspect_episode.py outputs/smoke/episode.sqlite
python scripts/experiment2_analysis/audit_state_protocol.py --stores outputs/exp3/stores --recursive --strict
python scripts/experiment2_analysis/inspect_credit_trace.py outputs/exp3/stores/example.sqlite --selected-only
```

## ECHO 接口

```bash
python scripts/common/check_echo_compat.py --echo-root "$ECHO_ROOT"
python scripts/common/install_echo_hooks.py --echo-root "$ECHO_ROOT"
```

第二条命令不带 `--apply`，可用于 CI 检查源码标记是否仍匹配。

GPU 冒烟训练前随机抽取 10 个 SQLite 账本，运行：

```bash
esr-grpo analyze outputs/exp3/stores/*.sqlite --output outputs/exp3/debug.json
```

ECHO 分段元数据和训练日志使用独立入口检查：

```bash
python scripts/experiment3_training/inspect_rollout_metadata.py rollout_segments.jsonl --strict
python scripts/experiment3_training/inspect_training_log.py train.log --output outputs/exp3/log_audit.json --strict
```

重点观察：

- `legal_submit_rate` 是否接近零；
- `illegal_action_counts` 是否集中在某个接口；
- `avg_credit_token_density` 是否为 0 或接近 1；
- 成功组是否大量出现全零组内优势；
- final segment 是否确实携带 Judge 奖励；
- `esr_response_credit_mask` 长度是否与 padded response 对齐。

## 常见训练问题

`loss=0` 不一定是错误。二元奖励下，同组 rollout 全对或全错都会得到零 GRPO 优势。

掩码全零时，先检查合法提交和 Benchmark 成功，再检查动作 token span。不要直接把最终分段全部置为 1，这会把 ESR 退化为密集信用。

Judge 和 Verifier 不应共享同一个状态结果。Verifier supported 但 Judge 错误是正常 bad case，应保留用于协议分析。

30B 模型、32B Judge、embedding 服务和 rollout 同时占用 4 张 H800 时容易出现资源冲突。优先把 Judge 改为离线队列，或部署到独立节点。
