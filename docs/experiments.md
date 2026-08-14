# 实验执行说明

## 共同控制变量

所有方法必须固定：

- split manifest 及文件哈希；
- 固定语料、索引版本和检索模型；
- top-k、每篇 snippet 长度、open_page 长度和总工具预算；
- 策略模型初始 checkpoint、rollout 数量和采样参数；
- 最终 Judge checkpoint、提示、温度和解析规则。

如果 AREX-Turbo 原生协议与 ESR 工具不同，应同时报告“原生工具结果”和“统一工具结果”，不能只取更有利的一项。

## 实验一

### Benchmark 检查

`scripts/experiment1_pipeline/run_benchmark.py` 运行真实策略服务，逐题保存：

- `stores/<query_id>.sqlite`：Evidence、TaskState 和动作账本；
- `trajectory_<query_id>.json`：模型消息和工具结果；
- `runs/<query_id>.json`：BrowseComp-Plus 官方 run 格式。

检查项：数据字段完整、检索 docid 属于固定语料、Judge 解析错误率、Accuracy、Evidence Recall、校准误差和平均工具调用数。

### ESR Pipeline 检查

先运行 `scripts/experiment1_pipeline/run_smoke_episode.py`，再用 4B 模型运行 50–100 步训练。每个 batch 记录：

- 合法提交率和各动作非法率；
- Evidence 数量、TaskState 版本数和覆盖失败次数；
- 每个轨迹的奖励、组内优势、正优势比例；
- 被选动作数、动作掩码 token 密度和空掩码比例；
- policy loss、KL、clip fraction 和梯度范数。

当所有 rollout 的最终奖励相同时，GRPO 优势为零。应统计这种组，而不是把 loss 不下降直接归因于实现错误。

## 实验二

### Evidence 必要性

使用 `configs/experiment2/evidence_ablation.yaml`。`summary_only` 变体必须真正禁止原文恢复，不能仍在隐藏上下文中保留完整页面。

统计 `read_evidence` 调用次数、使用时机、重新读取后 finding 修改率和最终事实错误。只根据调用频率不能判断 Evidence 是否有用，还需比较去除原文后的答案质量。

### TaskState 表达能力

人工标签至少覆盖单证据、多证据并列、多跳、冲突和时间约束。记录无法自然表达的关系，并检查模型是否把结构信息塞入 answer 或 gap 文本。

### 协议绕过

检查非法提交、未覆盖 Evidence、占位 finding、虚构 Evidence ID、提交答案与 TaskState.answer 不一致、验证后新增 Evidence 仍直接提交等情况。

### 信用准确性

人工标签使用动作 ID，不用 turn ID。报告 precision、recall、F1、误报类型和漏报类型。结构回溯属于关系归因，实验标题和结论不要使用“因果准确率”。

## 实验三

四条主线配置位于 `configs/experiment3/`：

| 方法 | 状态管理 | 信用范围 |
|---|---|---|
| AREX-Turbo | AREX 原生或统一工具协议 | 推理基线 |
| 全轨迹 GRPO | 无 ESR 状态 | 全部策略 token |
| ECHO | action/finding 轮次摘要 | 最终分段和选中轮次 |
| ESR-GRPO | Evidence + TaskState | 回溯后的完整动作 |

正式结果至少报告 Accuracy、Evidence Recall、Calibration Error、平均工具调用数、合法提交率和训练成本。Citation 指标可以补充，但不能用 Citation Recall 替代 Evidence Recall。

## 数据使用边界

官方 830 题是单一 test 发布。若使用自定义 RL 切分，论文中必须明确说明这是“从官方 test 发布构造的研究切分”。最终测试题一旦用于方案调试、提示修改或 bad case 迭代，就不再是未触碰测试集。
