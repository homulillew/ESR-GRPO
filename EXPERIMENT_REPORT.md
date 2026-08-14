# BC-Plus 固定语料评测：AREX-Turbo vs ABSeeker-4B-RL 实验结果与分析

> 评测日期：2026-08-13 ~ 2026-08-14
> 数据集：BrowseComp-Plus（BC-Plus）固定语料，830 题 validation 全集
> Judge：Qwen3-32B（本地 vLLM，TP=2，关闭 thinking）
> 检索后端：BM25（端口 8000，100,195 文档固定语料）

---

## 1. 实验概述

在 BC-Plus 固定语料上对两个搜索 Agent 做端到端评测对比：

| | AREX-Turbo | ABSeeker-4B-RL |
|---|---|---|
| 模型规模 | — | 4B（Qwen3.5-4B 基座 + ABC 训练） |
| 原生检索后端 | 真实 web（Serper + Jina） | 真实 web（Serper + Jina） |
| 适配方式 | 替换为 BM25 检索 + corpus 全文 | 替换为 BM25 检索 + summary LLM 压缩 |
| Context 管理 | 原生支持 | **原生无 discard-all**（论文 55.3% 的关键） |
| 最大上下文 | — | 262,144 tokens |
| 最大交互轮数 | — | 200 turns |

两者原本都训练在真实 web 上，本次均适配到 BC-Plus 固定语料（BM25 检索 + 伪 URL `bcplus://document/<docid>`），用同一 Qwen3-32B judge 评测，保证可比性。

---

## 2. 主要结果

| 指标 | AREX-Turbo | ABSeeker-4B-RL |
|---|---|---|
| **Accuracy** | **71.89%**（596/830） | **40.34%**（334/829） |
| **Completion Rate** | **93.13%**（773/830） | **48.37%**（401/829） |
| Recall | 81.32% | — |
| Avg Tool Calls / 题 | **43.9** | **156.1** |
| Avg Wall Seconds / 题 | — | 359.7 |
| Judge Parse Errors | 1 | 1 |

> ABSeeker 处理 829 题（1 题缺失），AREX 处理 830 题。

**核心差距**：ABSeeker 的 Accuracy 比 AREX 低 31.55 个百分点，但根因不在"答得准不准"，而在"答不出来"——Completion Rate 48% vs 93%，超过一半的题根本没产出答案。

---

## 3. 失败模式分析（ABSeeker）

829 题中 428 题失败，按 `finish_reason` 分类：

| finish_reason | 数量 | 占比 | 含义 |
|---|---|---|---|
| `context_token_limit` | 285 | 34.4% | 上下文堆到上限（中位 226K）爆掉，未出答案 |
| `empty_response` | 138 | 16.6% | 跑满 200 轮，vLLM 在超大 context 上生成不出 token |
| `giveup` | 3 | 0.4% | 连续 3 次工具调用解析失败 |
| `tool_count_limit` | 2 | 0.2% | 跑满 200 轮仍无答案 |

**三类失败本质同源——上下文爆炸**：

- 失败题 context tokens 中位数 **220,937**，逼近 256K 上限；**52% 的失败题 ctx > 220K**。
- `empty_response` 中 **70% 的题 ctx > 180K**——是上下文过大导致模型生成退化，并非独立的失败模式。

---

## 4. 根因分析：模型在固定语料上"打转"

### 4.1 证据一：成功题也在打转

| | 成功题 | 失败题 |
|---|---|---|
| Avg tool calls | 155.5 | 156.7 |
| Median tool calls | **200**（跑满） | 162 |
| 跑满 200 轮才答出 | **59%**（236/401） | — |
| Context 中位数 | 187,958 | 220,937 |
| Context > 200K | **37%**（在爆边缘才答出） | 52% |

成功题平均要 **156 轮**（AREX 只要 44 轮），59% 跑满 200 轮上限才挤出答案，37% 的成功题 context 已超 200K——说明模型不是"搜到就答"，而是"反复试到撞墙前侥幸答对"。

### 4.2 证据二：search 远多于 visit

抽样 50 题统计工具调用比例：

| 工具 | 调用次数 | 占比 |
|---|---|---|
| search | 6,251 | **86.6%** |
| visit | 970 | **13.4%** |

search : visit ≈ **6.5 : 1**。模型绝大多数时间在反复发 search，很少点开文档读内容。

### 4.3 证据三：trace 显示反复换 query 而不 visit

以 qid=632（`context_token_limit` 失败）为例，181 轮全部是 search、**0 次 visit**，query 反复换近义说法：

```
[1] search "boxer bachelor's degree in accounting parent loves boxing"
[2] search "consecutive gold medals freestyle wrestling Pan American Games 2014"
[3] search "freestyle wrestling world championship gold 2014"
[4] search "battled depression as a teenager boxer accounting degree"
[5] search "freestyle wrestler accounting degree"
[6] search "bachelor's degree in accounting boxer"
[7] search "freestyle wrestler sports psychology degree"
[8] search "bachelor's degree in accounting boxer 2023"
[9] search "battled depression as a teenager boxer accounting degree sports psychology"
[10] search "freestyle tournament winner 2014"
...（共 181 次 search，0 次 visit，ctx 堆到 226K 爆掉）
```

模型在"换 query 重搜"和"visit 读文档"之间始终选择前者，从不收敛。

### 4.4 根本原因

1. **训练分布偏移**：ABSeeker 训练在真实 web 上，真实 web 里换 query 能命中不同页面，"多搜几次"是有效策略。但 BC-Plus 是**固定语料 + BM25**，同一意图的近义 query 召回文档高度重叠——反复 search 只会堆重复结果，不带来新信息。模型没学到"该 visit 去读"。

2. **原生无 context 管理**：论文明确指出，ABSeeker 无 context 管理时 BrowseComp 37.3%，加 discard-all 策略才到 55.3%。本次跑的是**原生无 context 管理配置**，256K 上下文被 search/visit 结果单向累积填满，没有回收机制，跑到 200K+ 必爆。这是 completion 只有 48% 的直接原因。

3. **search snippet 仍占空间**：即使 snippet 缩到 300 字符，每次 search 返回 10 条结果约 4000 字符，156 轮里若以 search 为主，累积仍可达 60 万字符量级，远超 256K token 上限。

---

## 5. 与论文报告值的对照

| 配置 | 论文 BrowseComp | 本次 BC-Plus |
|---|---|---|
| ABSeeker 无 context 管理 | 37.3% | **40.34%** |
| ABSeeker 有 discard-all | 55.3% | 未跑（待实现） |

本次 40.34% 与论文"无 context 管理"的 37.3% 基本吻合（BC-Plus 与 BrowseComp 题目分布略有差异），说明适配正确、未引入异常。**差距主要来自未实现 discard-all context 管理**。

---

## 6. 工程实现要点

### 6.1 双卡 4 实例架构（ABSeeker）

为提高双卡利用率，每张 GPU 同时跑 ABSeeker + Qwen3.5-4B summary LLM（同卡共存）：

| GPU | ABSeeker vLLM | Summary vLLM (Qwen3.5-4B) |
|---|---|---|
| GPU0 | 8002（util 0.72, ~58GB） | 8006（util 0.26, ~21GB） |
| GPU1 | 8004（util 0.72, ~58GB） | 8008（util 0.26, ~21GB） |

每卡 79GB < 81GB，实现双卡数据并行 + summary 走本卡零网络延迟、零外部 API 消耗。4 shard 并行，全量 830 题推理约 10.5 小时。

### 6.2 关键适配修复

1. **visit URL 变体解析**（关键）：模型会给 `bcplus://` 伪 URL 加前缀（`https://bcplus://...`、`https://r.jina.ai/http://bcplus://...` 等）。修复 `_resolve_docid` 用子串查找提取 docid，否则所有 visit 失败、模型跑满 200 轮无答案。

2. **summary LLM 压缩**：visit 不返回全文，用 Qwen3.5-4B 按 goal 压成 evidence/summary 摘要（~900 字符，压缩 27 倍），这是跑 200 轮不爆上下文的根本机制。

3. **search snippet 截断**：缩到 300 字符并跳过文档 YAML front matter，降低单次 search 占用。

### 6.3 Judge 配置

- Qwen3-32B 本地 vLLM，TP=2，max-model-len 32768，关闭 thinking（`chat_template_kwargs: {enable_thinking: false}`），避免 thinking 污染判定输出。
- evaluate.py 补 `--judge-disable-thinking` 参数 + clients.py 补 429/5xx 退避重试。
- 829 题 judge 约 25 分钟。

---

## 7. 结论与后续

### 结论

- **AREX-Turbo 在 BC-Plus 上显著优于 ABSeeker-4B-RL**：Accuracy 71.89% vs 40.34%，主要差距来自 Completion Rate（93% vs 48%）。
- ABSeeker 的瓶颈是**上下文管理缺失**导致大量题目爆上下文失败，而非答案判定准确性——成功题里 judge 判定正确率并不低。
- 模型在固定语料上存在**"反复 search 不 visit"的打转行为**，是训练分布偏移的表现。

### 后续改进方向

1. **实现 discard-all context 管理**（优先级最高）：上下文逼近上限时丢弃早期历史、保留关键证据。论文显示这是 ABSeeker 从 37.3% → 55.3% 的关键，预计能把 completion 从 48% 拉到 70%+，accuracy 随之提升。

2. **引导 visit 行为**：search 命中后引导模型 visit 文档读内容，减少无效 search 堆积。可在 prompt 或工具返回中强化"先 visit 再判断"的策略。

3. **query 去重/收敛提示**：检测到近义 query 重复 search 时，提示模型换策略（visit 已有结果或调整搜索方向）。

---

## 附录：产物路径

| 产物 | 路径 |
|---|---|
| AREX 结果 | `/data1/ESR-GRPO/outputs/arex_bcplus_native/validation_full_v4/` |
| AREX 评测 | `…/validation_full_v4/evaluation_qwen32b.json` |
| AREX 打包 | `/data1/ESR-GRPO/outputs/arex_bcplus_native/arex_830_full_v4.tar.gz`（924MB） |
| ABSeeker 结果 | `/data1/ESR-GRPO/outputs/abseeker_bcplus_native/validation_full_v1/` |
| ABSeeker 评测 | `…/validation_full_v1/evaluation_qwen32b.json` |
| ABSeeker 打包 | `/data1/ESR-GRPO/outputs/abseeker_bcplus_native/abseeker_830_full_v1.tar.gz`（346MB） |
| 数据集 | `/data1/ESR-GRPO/BrowseComp-Plus/data/prepared/browsecomp_plus_decrypted.jsonl` |
| 适配代码 | `/data1/ESR-GRPO/ESR-GRPO/exper/abseeker_bcplus_native/` |
