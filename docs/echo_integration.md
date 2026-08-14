# ECHO/verl 集成

## 1. 为什么需要三个钩子

ECHO 原实现以 `turn_id` 标记整轮响应，并默认给最终轨迹分段的全部策略 token 分配优势。ESR 需要：

1. 为每个并行工具调用提取独立 token 范围；
2. 把每个分段的 ESR 掩码传到 trainer；
3. 注册只使用 ESR 掩码的优势函数。

`scripts/common/install_echo_hooks.py` 对当前工作区 ECHO 快照安装这些接口。默认模式只检查标记，不写文件。`--apply` 写入前创建 `.esr-grpo.bak`。

## 2. Hermes 动作范围

`extract_hermes_action_spans` 在模型响应的 `<tool_call>...</tool_call>` 块上使用 tokenizer offset mapping。多个并行块按出现顺序映射到多个 FunctionCall。训练前必须抽查响应 token 解码后能恢复完全相同的块边界。

当前实现只把 Hermes 格式作为经过定义的训练格式。切换 `paper_fc`、`seed_oss` 或其他 parser 前，需要实现并测试对应边界提取器；不能把整轮响应范围复制给每个并行动作。

## 3. 原生工具

`configs/echo/esr_tools.yaml` 注册六个工具。它们通过 ECHO 的 `AgentData` 共享一个 `ESREnvironment`，账本默认写入 `ESR_STORE_DIR`。

环境变量：

- `ESR_RETRIEVAL_URL`：ECHO BrowseComp 检索服务根地址；
- `ESR_VERIFIER_BASE_URL`：OpenAI 兼容在线验证器；
- `ESR_VERIFIER_MODEL`：验证模型；
- `ESR_VERIFIER_API_KEY`：可选密钥；
- `ESR_STORE_DIR`：每个 rollout 的 SQLite 目录。

`submit_answer` 被 ToolAgentLoop 识别为终止动作。自定义奖励函数从合法提交元数据中读取答案，再调用 ECHO 的 BrowseComp Judge；没有合法提交时最终奖励为零。

## 4. 多段轨迹

rollout 完成后，适配器根据每个 `TrajectoryOutput.response_ids` 的长度生成分段掩码。Ray trainer 在逻辑 rollout 层读取最终分段奖励并计算组内优势，然后对每个分段应用自己的动作掩码。

动作元数据需要覆盖以下边界：

- 没有发生上下文重建；
- 一个动作跨越多个 token，但不能跨越两个轨迹分段；
- 同轮多个工具调用；
- 最终 submit 位于最后分段；
- 非最终分段最终奖励为零；
- overlong rollout 优势为零；
- 无合法提交或空结构掩码。

## 5. 当前资源入口

`scripts/experiment3_training/run_esr_4gpu.sh` 是单节点 4 GPU 的 4B Pipeline 配置。它使用 Megatron 数据并行、SGLang rollout 和上下文截断分段。正式 30B MoE 实验需要根据可用节点重新配置 TP/PP/CP 和独立 Judge 资源。

公开 ECHO 32B 脚本是 4 节点 × 8 GPU。比较论文训练成本时应保留这项差异。
