# LLM 时间线输出层 V3 交接说明

本文档用于交接 `newsline` 项目当前的 LLM 时间线输出层 V3。它可以作为后续继续开发、论文方法描述、答辩汇报或新对话恢复上下文的 handoff。

当前日期：2026-05-03

## 1. 当前结论

LLM 时间线输出层目前已经完成一轮可交付优化。就“LLM 层本身”而言，当前没有必须继续推进的结构性优化点。

已经解决的问题：

- 大 topic 下单条请求过多，运行时间过长。
- batch=4 时 LLM 输出 JSON 不稳定。
- LLM 输入字段过多，部分字段不需要模型判断。
- `needs_split` / `needs_merge` 现阶段尚无结构改写能力，但字段需要保留。
- `display_title` 需要保留 LLM 润色和纠偏能力。
- topic 字面匹配过强或过弱，容易误伤翻译别名或同形词。

当前定位：

```text
SBERT 层输出候选事件簇
-> 规则层进行风险标记和分流
-> LLM 只审查需要语义裁决的候选事件
-> 代码完成稳定排序、结构输出、JSON / MySQL 持久化
```

本层仍然不是全量 LLM 生成时间线，而是：

```text
规则骨架 + 压缩事件卡片 + 小批量 LLM 裁判 + 确定性排序与输出
```

## 2. V3 相比 V1 / V2 的核心变化

### 2.1 EventCard 输入压缩

代码位置：

```text
core/timeline_reasoning/models.py
```

`EventCard.to_llm_dict()` 现在只向 LLM 发送压缩后的事件卡片：

```text
event_id
topic
topic_profile
title
cluster_size
source_count
confidence
system_noise
time.start / time.end / time.anchor
risk_flags
quality_hints
evidence
noise_reason
```

不再发送：

```text
articles
member_news_ids
member_titles_sample
member_title_evidence 原始完整结构
quality_summary 原始完整结构
semantic_override_edge_count
graph_edge_count
news_id
```

其中：

- `evidence` 最多保留 4 条标题证据。
- 时间字段压缩为日期级别。
- `quality_summary` 被压缩为 `quality_hints`。
- `topic_profile` 被压缩为 `type / ambiguous / strict_entity`。

这样做的目标是减少 prompt 长度，提高 batch 稳定性，同时保留 LLM 判断 topic、噪声、标题、时间锚点所需的信息。

### 2.2 LLM 输出字段收敛

当前 prompt 版本：

```text
timeline_reasoning_v7
```

代码位置：

```text
core/timeline_reasoning/prompts.py
core/timeline_reasoning/llm_judge.py
```

LLM 当前只需要输出：

```text
event_id
keep_event
is_topic_relevant
final_is_noise
display_title
resolved_time_anchor
decision_reason
```

这些字段保留给 LLM，是因为它们需要语义判断：

| 字段 | 作用 |
|---|---|
| `keep_event` | 是否进入最终时间线 |
| `is_topic_relevant` | 是否与 topic 语义相关 |
| `final_is_noise` | 是否最终视为噪声 |
| `display_title` | 对 canonical title 做必要润色或纠偏 |
| `resolved_time_anchor` | 选择最终排序使用的时间锚点 |
| `decision_reason` | 简短解释，方便调试和论文说明 |

以下字段仍保留在数据结构和输出中，但不再让 LLM 判断：

| 字段 | 当前处理方式 |
|---|---|
| `needs_split` | 固定为 `false`，为未来结构拆分能力预留 |
| `needs_merge` | 固定为 `false`，为未来跨事件合并能力预留 |
| `split_reason` | 固定为 `null` |
| `merge_reason` | 固定为 `null` |
| `resolved_time_start` | 使用 EventCard 原始 `event_time_start` |
| `resolved_time_end` | 使用 EventCard 原始 `event_time_end` |
| `decision_confidence` | 本地由候选事件 confidence 派生 |
| `time_confidence` | 本地由是否存在 `resolved_time_anchor` 派生 |

注意：`display_title` 不是无用字段。它承担标题润色和纠偏能力，已经恢复为 LLM 输出字段。当前 prompt 要求：

```text
Return display_title only when it improves or corrects the input title. Otherwise return null.
```

最终代码逻辑为：

```text
display_title = LLM 返回值 or canonical_title
```

这样既保留标题优化能力，也避免模型每条都重复抄写 canonical title。

### 2.3 `needs_split` / `needs_merge` 只保留标记，不自动改结构

用户明确决定：当前毕业设计时间有限，`needs_split` / `needs_merge` 不做自动结构修改。

当前策略：

- 数据模型保留字段。
- JSON / MySQL 输出保留字段。
- LLM prompt 不再要求判断。
- LLM parser 即使收到旧模型输出的 split / merge 字段，也会忽略。
- 当前所有决策默认：

```text
needs_split = false
needs_merge = false
split_reason = null
merge_reason = null
```

未来如果要扩展，可以在此基础上增加：

- split candidate detection
- merge candidate detection
- 跨事件 pairwise merge 判断
- 自动拆分 / 合并后的 timeline 重建

但 V3 不做这些结构改写。

### 2.4 batch=4 可稳定运行

当前默认：

```text
llm_batch_size = 4
```

相关位置：

```text
code/script/run_timeline_reasoning.py
code/script/run_timeline_web_job.py
core/timeline_reasoning/pipeline.py
core/timeline_reasoning/llm_judge.py
```

之前 batch=4 容易触发 JSON 解析失败，然后 fallback split 到 batch=2。主要原因是 LLM 输出字段过多，4 条事件一起返回时 JSON 更长、更容易格式漂移。

V3 收敛输出 schema 后，Trump 样例已能稳定以 batch=4 跑完：

```text
batch 1/53 ... batch 53/53
无 split-a / split-b fallback
```

## 3. 当前模块职责

| 文件 | 职责 |
|---|---|
| `core/timeline_reasoning/models.py` | EventCard、EventDecision、TimelineRecord、TimelineReasoningResult 数据结构；LLM 输入压缩 |
| `core/timeline_reasoning/event_cards.py` | 从 SBERT EventNode 和 assignments 构造 EventCard，回挂文章证据 |
| `core/timeline_reasoning/topic_profile.py` | 构造 topic 画像，辅助实体 / 同形词 / 翻译别名判断 |
| `core/timeline_reasoning/filters.py` | 风险标记、规则分流、规则决策 |
| `core/timeline_reasoning/prompts.py` | LLM prompt，当前版本 `timeline_reasoning_v7` |
| `core/timeline_reasoning/llm_judge.py` | 调用 Ollama，解析 JSON，失败 fallback，batch split 保护 |
| `core/timeline_reasoning/ordering.py` | 根据最终时间字段稳定排序并生成 TimelineRecord |
| `core/timeline_reasoning/persistence.py` | 写入 MySQL 表和节点表 |
| `core/timeline_reasoning/pipeline.py` | LLM 时间线输出层主流程 |
| `code/script/run_timeline_reasoning.py` | 命令行运行入口 |
| `code/script/run_timeline_web_job.py` | Web job 运行入口 |
| `code/script/eval_timeline_reasoning.py` | 当前结果评估与风险摘要 |

## 4. 规则分流策略

代码位置：

```text
core/timeline_reasoning/filters.py
```

每个 EventCard 先被规则层标记风险，再分为：

```text
auto_accept
llm_review
rule_reject
```

`rule_reject` 只用于明显缺少必需字段的坏数据，例如：

```text
missing_event_id
missing_canonical_title
empty_cluster
```

`llm_review` 用于存在结构性风险的候选事件，例如：

```text
system_noise
missing_time
long_time_span
rolling_coverage
rolling_coverage_title
low_temporal_coherence
low_semantic_cohesion
large_cluster
semantic_override_edges
low_graph_density
high_duplicate_ratio
```

`low_confidence`、`low_source_support`、`translated_topic_alias_risk`、`ambiguous_topic_low_support` 当前更多作为诊断信号，不会单独导致 LLM review。

这样做是为了防止大 topic 下 LLM 调用数量爆炸，同时避免单来源但真实的事件被过早误杀。

## 5. LLM prompt 当前任务

当前 prompt 让 LLM 做 5 件事：

1. 判断候选事件是否是适合时间线的具体真实事件。
2. 判断它是否与 topic / topic_profile 相关。
3. 判断是否最终视为噪声。
4. 选择最可靠的 `resolved_time_anchor`。
5. 必要时生成更好的 `display_title`。

关键约束：

- 必须通用于任意 topic，不写 topic 特例。
- `system_noise`、`low_confidence`、`low_source_support` 等只是诊断信号，不是自动结论。
- translated alias 不是天然错误，只要明确指向同一实体就应保留。
- 对 proper noun / strict entity topic，候选事件中 topic 可以是 actor、speaker、target、counterparty、claim subject 等，不要求 topic 是唯一主体。
- 不能引入输入证据以外的新事实、日期或实体。
- 必须输出严格 JSON。

## 6. 运行方式

先激活虚拟环境：

```bash
cd /Users/hjfl/newsline
source .venv/bin/activate
```

运行最新 SBERT 发现结果上的 LLM 时间线层：

```bash
python code/script/run_timeline_reasoning.py \
  --topic "Trump" \
  --mode standard \
  --dry-run
```

指定 batch：

```bash
python code/script/run_timeline_reasoning.py \
  --topic "Trump" \
  --mode standard \
  --dry-run \
  --llm-batch-size 4
```

指定 SBERT run：

```bash
python code/script/run_timeline_reasoning.py \
  --topic "Trump" \
  --run-id "Trump_20260429_222422_80de0fbf" \
  --mode standard \
  --dry-run
```

调试小样本：

```bash
python code/script/run_timeline_reasoning.py \
  --topic "Apple" \
  --mode fast \
  --limit-events 20 \
  --dry-run
```

输出 JSON 默认位于：

```text
outputs/timeline/
```

## 7. 评估方式

运行：

```bash
python code/script/eval_timeline_reasoning.py \
  outputs/timeline/Trump_timeline_Trump_timeline_20260503_203415_dc295a2d.json \
  --suspicious-limit 200
```

评估脚本会输出：

- topic / run_id / prompt_version
- input event count
- timeline count
- decision source 分布
- keep / noise 分布
- needs_split / needs_merge 分布
- risk flags 统计
- suspicious kept events

注意：当前 `needs_split` / `needs_merge` 已经固定为 false，所以 eval 中类似：

```text
multi_article_structural_risk_without_split
```

更多是“结构风险提示”，不是当前版本的错误结论。后续如果长期保留 split/merge 只标记不执行，建议把 eval 文案改得更温和。

## 8. 当前样例结果

### 8.1 Apple fast

早期 fast 样例：

```text
input_event_count: 19
review_event_count: 4
accepted_event_count: 19
rejected_event_count: 0
```

说明 fast 模式下主要依赖规则，仅抽少量高风险事件给 LLM。

### 8.2 Apple standard

standard 样例曾达到：

```text
input_event_count: 19
review_event_count: 19
accepted_event_count: 18
rejected_event_count: 1
```

后续路由优化后，standard 不再把所有低支持事件都送 LLM，而是更重视结构性风险。

### 8.3 Trump standard V3

最新 V3 样例：

```text
topic: Trump
discovery_run_id: Trump_20260429_222422_80de0fbf
reasoning_run_id: Trump_timeline_20260503_203415_dc295a2d
prompt_version: timeline_reasoning_v7
mode: standard
input_event_count: 2016
review_event_count: 209
accepted_event_count: 2000
rejected_event_count: 16
decision_source:
  rule: 1807
  llm: 209
```

对比旧版：

```text
旧版 v3:
review_event_count: 408
accepted_event_count: 1935
rejected_event_count: 81

新版 v7:
review_event_count: 209
accepted_event_count: 2000
rejected_event_count: 16
```

解释：

- LLM 审查量减少约一半。
- batch 从单条调用变为 batch=4，实际 LLM 请求数从约 408 次下降到约 53 次。
- 输出 schema 收敛后，batch=4 已能正常跑完。
- 当前策略偏高召回，保留较多事件；这是设计选择，不是 LLM JSON 稳定性问题。

## 9. 当前已知取舍

### 9.1 时间线仍可能很长

Trump 这种大 topic 最终保留 2000 条事件。这个问题目前不在 LLM 层继续处理。

可选后续方向：

```text
presentation_score
timeline_rank
默认前端展示 top N
按月份 / 子主题折叠
```

但用户当前明确决定：topN 暂不处理。

### 9.2 standard 模式偏高召回

当前 standard 模式不会因为单独的低置信度、低来源支持、翻译别名风险就强制 LLM review。

优点：

- 减少 LLM 调用。
- 避免单来源真实事件被误删。
- 大 topic 能稳定跑完。

缺点：

- 最终时间线会保留更多低支持事件。
- 噪声过滤偏保守。

当前毕业设计阶段，这个取舍是合理的：宁可保留候选并提供风险标记，也不在 LLM 层过度删除。

### 9.3 LLM 对 topic relevance 仍可能偏严格

个别 Trump 样例中，LLM 会把“Trump 只是发言者 / 电话参与者 / 外交介入者”的事件判为不相关。

当前 prompt 已经说明：

```text
topic-relevant when the named entity is a clear actor, speaker, decision-maker, target, counterparty, subject of a concrete claim, or otherwise materially involved
```

如果后续继续优化，可以针对这一条通用约束微调，但不建议写 Trump 特例。

## 10. 数据库与输出

主要 MySQL 表：

```text
timeline_reasoning_runs
timeline_reasoning_decisions
timeline_nodes
timeline_node_articles
```

JSON 输出结构：

```text
summary
timeline
decisions
decision_contexts
output_paths
```

其中：

- `timeline` 是最终保留并排序后的事件节点。
- `decisions` 保存所有输入事件的决策，包括被拒事件。
- `decision_contexts` 保存 EventCard 原始上下文，方便复盘。
- 每个 timeline node 会回挂完整 articles，前端可以展开查看。

## 11. 测试

当前相关测试：

```bash
python3 -m unittest tests.test_timeline_reasoning
python3 -m unittest tests.test_imports tests.test_event_discovery tests.test_timeline_reasoning
```

当前已通过：

```text
Ran 33 tests
OK
```

测试覆盖重点：

- 从 event discovery 结果读取 EventNode。
- 构造 EventCard 并压缩 LLM 输入。
- 规则路由与风险标记。
- LLM JSON parser 去除 think block。
- 显式 null 时间保留。
- `display_title` 由 LLM 可选纠偏。
- `resolved_time_start/end`、置信度等字段本地派生，不让 LLM 决定。
- pipeline dry-run JSON 输出。

## 12. 后续建议

当前不建议继续大改 LLM 层。若之后还有时间，优先级如下：

1. 更新 eval 文案，让 split/merge 相关提示更符合“字段预留、不执行结构改写”的现状。
2. 增加少量人工 gold sample，对 rejected / suspicious kept 做人工标签。
3. 如需改善前端体验，再做 presentation score / topN / 折叠展示，但不要混入 LLM 决策层。
4. 若未来实现 split/merge，再重新启用 LLM 标记或设计 pairwise 结构修复模块。

当前版本可以作为毕业设计 LLM 时间线输出层的稳定版本继续使用。
