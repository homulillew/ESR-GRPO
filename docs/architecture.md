# 架构说明

## 1. 数据对象

`Evidence` 保存完整原文、来源、内容哈希、首次 `open_page` 动作和上游 `search` 动作。保存后禁止修改和删除。

`TaskState` 保存当前答案、全部 Evidence 的 finding 目录、主要证据 ID、待解决问题、验证状态和生成该版本的动作。每次 `update_state` 或 `verify_answer` 追加一个版本。

`ActionRecord` 保存独立 `action_id`、动作类型、合法性、状态版本、Evidence、gaps、上游动作和一个或多个 token 范围。同一轮并行调用共享 `turn_id`，但不共享 `action_id`。

SQLite 表使用触发器拒绝 Evidence、TaskState 和 ActionRecord 的 UPDATE/DELETE。提交答案作为只写一次的 episode 元数据保存。

## 2. 动作状态机

```text
                         ┌──────────── read_evidence ────────────┐
question → search → open_page → update_state → verify_answer    │
                                      │             │            │
                                      │        needs_revision ───┘
                                      │             │
                                      │         supported
                                      │             │
                                      └──────── submit_answer
```

`open_page` 的执行顺序是：

1. 检查 docid 是否来自指定的合法 search 动作；
2. 从固定语料读取完整正文；
3. 计算内容哈希并保存不可变 Evidence；
4. 最后截断模型可见的工具结果。

因此，即使当前工具结果触发上下文溢出，完整正文仍能通过 `read_evidence` 恢复。

`update_state` 把输入视为 finding patch：已有且未修改的 finding 自动保留，新 Evidence 必须提供 finding。修改旧 finding 时，原始 Evidence 必须刚由 `read_evidence` 或 `open_page` 放入可见上下文。该动作复制上一版本的 gaps，并把验证状态重置为 `unverified`。

`verify_answer` 读取主要 Evidence 的完整正文，调用独立 Verifier。它创建、保留或解决带稳定 `gap_id` 的待解决问题。

`submit_answer` 仅在最新状态覆盖全部 Evidence、答案非空、主要证据非空、gaps 为空且状态为 `supported` 时成功。

## 3. 信用回溯

信用路由先构造与奖励无关的结构关系，再应用最终奖励和组内优势门槛。

### 最终答案链

- 形成最终答案、最终主要证据集合和最终 finding 目录的最后一次 `update_state`；
- 生成最终 supported TaskState 的 `verify_answer`；
- 合法 `submit_answer`。

### 最终证据链

对最终 `supporting_evidence` 中每个 ID：

- 首次保存 Evidence 的 `open_page`；
- 该 Evidence 的上游 `search`；
- 写入最终 finding 的 `update_state`；
- 最后一次将其纳入主要证据的 `update_state`；
- 如果最终 finding 来自重新读取，则包含实际供该更新使用的 `read_evidence`。

### gap 解决链

对每次 `needs_revision` 验证，寻找后来显式解决相同 `gap_id` 的验证。只有区间内存在保留到最终状态的答案、finding、主要证据或最终 Evidence 时，才保留发现 gap、解决 gap 和中间相关动作。

三条链合并后按 `sequence_index` 排序。动作的所有 token span 整体置为 1；其他动作和工具观察置为 0。

最终 token 优势为：

$$
\widetilde A_{i,t}=M_{i,t}Z_i\max(A_i,0).
$$

其中 $Z_i$ 要求 Benchmark 成功、合法提交且轨迹没有被判定为无效。

## 4. 验证器与 Judge

Verifier 是在线工具，输入当前答案和主要 Evidence 原文，输出 `supported` 或 `needs_revision`。它只推动搜索状态。

Judge 在 episode 结束后比较提交答案和标准答案，输出最终奖励。训练不得把 Verifier 的 supported 状态直接当成答案正确奖励，否则策略可以通过操纵验证状态获得奖励。

## 5. 持久化与并发

并行 search/open/read 调用可以来自不同 worker thread。环境锁保证 SQLite 追加动作、Evidence 和状态的事务顺序稳定。并行结果各自拥有 token 范围，因此最终只采用其中一个 Evidence 时，其他调用不会因共享 turn 而得到信用。

