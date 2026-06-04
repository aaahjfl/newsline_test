# LLM 层 v1 总结

当前日期：2026-04-22

## 1. 设计目标

LLM 层 v1 的目标不是重新发现事件，也不是把全部新闻直接交给大模型生成时间线，而是在已有候选事件簇基础上完成一层轻量、可控、可追溯的“决断”。

本层主要解决四件事：

1. 判断候选事件是否应该进入最终时间线。
2. 判断候选事件是否与用户 topic 相关。
3. 判断候选事件是否是最终噪声。
4. 在已有时间字段基础上选择最终展示时间，并生成稳定排序。

因此，本层定位为：

```text
规则骨架 + 轻量大模型裁判 + 结构化落库
```

而不是：

```text
全量 LLM 时间线生成器
```

这样的设计有几个好处：

- 降低 LLM 上下文压力。
- 降低本地模型运行负载。
- 避免纯生成式时间线带来的不可控和幻觉。
- 保留完整溯源链，方便前端展示和论文解释。
- 后续可逐步优化 prompt、规则和局部排序，不影响整体架构。

## 2. 核心设计思路

### 2.1 LLM 输入要轻

每个候选事件簇不会把全部新闻都送给 LLM。

LLM 输入使用轻量事件卡片 `EventCard`，主要包含：

```text
event_id
topic
canonical_title
cluster_size
source_count
confidence
system_is_noise
noise_reason
event_time_start
event_time_end
event_time_anchor
risk_flags
member_titles_sample
```

完整新闻列表 `articles` 不进入 LLM 输入。

原因：

- 本地 `qwen3.5:9b` 上下文和速度有限。
- 大量文章会造成 prompt 过长。
- LLM 第一版只需要做事件级裁判，不需要读完整证据链。

### 2.2 展示输出要完整

虽然 LLM 输入轻量，但最终 JSON 和 MySQL 输出会把事件簇下的全部新闻重新挂回时间线节点。

即：

```text
LLM 输入：canonical_title + 轻量字段 + 少量成员标题样本
最终输出：时间线节点 + 簇内全部新闻 articles
```

这样前端可以默认展示简洁时间线，点击节点后展开全部新闻来源、标题、日期和 URL。

### 2.3 规则先分流，LLM 再裁判

v1 没有让所有事件都无条件进入 LLM，而是先通过规则分为三类：

```text
auto_accept
llm_review
rule_reject
```

含义：

| 分流结果 | 含义 |
|---|---|
| `auto_accept` | 风险较低，规则直接保留 |
| `llm_review` | 存在风险，交给 LLM 判断 |
| `rule_reject` | 极少数坏数据，规则直接剔除 |

这样做是为了减少 LLM 调用次数，同时避免过早删除真实事件。

### 2.4 system_is_noise 只是参考信号

`system_is_noise` 会进入 LLM 输入，但不会作为最终结论。

当前 prompt 明确告诉模型：

```text
system_is_noise is an upstream reference signal, not a final verdict. You may overturn it.
```

因此：

- `system_is_noise=true` 会触发 LLM review。
- LLM 可以保留该事件。
- LLM 也可以判定 `final_is_noise=true`。

最终噪声判断以 LLM 层输出的 `final_is_noise` 为准。

### 2.5 全局排序由代码稳定完成

v1 没有让 LLM 直接生成全局时间线顺序。

LLM 只负责选择或修正事件时间字段：

```text
resolved_time_start
resolved_time_end
resolved_time_anchor
```

最终排序由代码按确定性规则完成：

```text
resolved_time_anchor
resolved_time_start
resolved_time_end
event_time_anchor
event_time_start
event_time_end
event_id
```

这样可以保证：

- 排序结果稳定可复现。
- 方便调试。
- 方便后续评测 Kendall's tau。
- 符合“规则骨架 + LLM 裁判”的论文叙述。

## 3. 整体数据流程

### 3.1 输入

LLM 层读取已有数据库表：

```text
event_discovery_events
event_discovery_assignments
event_discovery_graph
```

其中：

- `event_discovery_events` 提供候选事件簇级字段。
- `event_discovery_assignments` 提供事件簇与原始新闻的映射。
- `event_discovery_graph` 提供轻量图链接诊断信息，如 `semantic_override` 边数量。

### 3.2 构造 EventCard

代码位置：

```text
core/timeline_reasoning/event_cards.py
```

每个候选事件被转换为一个 `EventCard`。

`EventCard` 同时服务两个目标：

1. 给 LLM 的轻量输入。
2. 后续回挂完整文章列表。

其中 `to_llm_dict()` 会排除完整 `articles`，避免上下文过长。

### 3.3 规则分流与风险标记

代码位置：

```text
core/timeline_reasoning/filters.py
```

当前风险标记包括：

```text
missing_event_id
missing_canonical_title
empty_cluster
system_noise
low_confidence
medium_confidence
missing_time
long_time_span
large_cluster
low_source_support
rolling_coverage_title
semantic_override_edges
```

这些标记会进入：

- LLM 输入
- JSON 输出
- MySQL 输出

用于调试、前端筛选和论文解释。

### 3.4 LLM 决断

代码位置：

```text
core/timeline_reasoning/llm_judge.py
core/timeline_reasoning/prompts.py
```

当前模型：

```text
qwen3.5:9b
```

当前 prompt 版本：

```text
timeline_reasoning_v1
```

LLM 输出 `EventDecision`：

```text
event_id
decision_source
keep_event
is_topic_relevant
final_is_noise
needs_split
needs_merge
display_title
resolved_time_start
resolved_time_end
resolved_time_anchor
decision_confidence
time_confidence
decision_reason
raw_response_json
```

### 3.5 Fallback

如果本地 LLM 请求失败、超时或返回无法解析的 JSON，v1 不会中断整个 pipeline。

它会生成：

```text
decision_source = "llm_fallback"
```

并用保守规则给出兜底决断。

这样可以保证系统整体跑通，适合初版集成和展示。

### 3.6 构建最终时间线

代码位置：

```text
core/timeline_reasoning/ordering.py
```

只有 `keep_event=true` 的事件会进入最终 `timeline`。

每个最终时间线节点是一个 `TimelineRecord`，其中包含：

- 事件基本字段
- 原始时间字段
- resolved 时间字段
- 决断字段
- 风险标记
- 簇内完整 `articles`

### 3.7 输出 JSON

无论是否 `--dry-run`，都会输出 JSON。

目录：

```text
outputs/timeline/
```

JSON 是完整结果，包含：

```text
summary
timeline
decisions
decision_contexts
```

其中：

- `timeline` 适合前端展示。
- `decisions` 适合调试每个事件的决断。
- `decision_contexts` 适合分析 LLM 输入上下文。

### 3.8 写入 MySQL

如果运行时不加 `--dry-run`，会写入四张表：

```text
timeline_reasoning_runs
timeline_event_decisions
timeline_nodes
timeline_node_articles
```

含义：

| 表名 | 意义 |
|---|---|
| `timeline_reasoning_runs` | 每次 LLM 层运行记录 |
| `timeline_event_decisions` | 每个事件的完整决断记录 |
| `timeline_nodes` | 最终进入时间线的节点 |
| `timeline_node_articles` | 每个节点展开后的原始新闻 |

## 4. 当前运行方式

调试推荐：

```bash
.venv/bin/python code/script/run_timeline_reasoning.py \
  --topic "Apple" \
  --mode fast \
  --limit-events 20 \
  --dry-run \
  --llm-batch-size 1 \
  --llm-timeout-seconds 60
```

正式写 MySQL：

```bash
.venv/bin/python code/script/run_timeline_reasoning.py \
  --topic "Apple" \
  --mode fast \
  --limit-events 20 \
  --llm-batch-size 1 \
  --llm-timeout-seconds 60
```

## 5. 当前 v1 的特点

### 已完成

- 能读取候选事件簇。
- 能构造轻量 LLM 输入。
- 能规则分流。
- 能调用本地 Ollama `qwen3.5:9b`。
- 能解析 LLM JSON 输出。
- 能处理 `<think>...</think>` 可见输出。
- 能处理 LLM 超时 fallback。
- 能生成最终时间线顺序。
- 能输出完整 JSON。
- 能写入 MySQL 四张表。
- 能保留每个时间线节点下的全部原始新闻。

### 当前偏保守

v1 目标是先跑通系统闭环，所以很多地方设计偏保守：

- 默认 `llm_batch_size=1`，保证稳定但速度慢。
- LLM 判断失败时 fallback，不中断系统。
- 不自动拆分或合并事件，只打标。
- 不做复杂局部时序推理。
- 不让 LLM 生成全局时间线。

## 6. 待改进点

以下只列 LLM 层自身待优化点，不涉及上游事件发现层。

### 6.1 Prompt 的 topic relevance 判断需要加强

当前 v1 对 topic 相关性的判断还偏宽松。

例如多义词或普通词命中时，LLM 可能只看到标题中出现了 topic 字符串，就认为相关。

后续应加强通用判断：

```text
不要仅凭字面重合判断 topic relevance。
如果 topic 是多义词，应根据 canonical_title 和 member_titles_sample 判断它是否符合用户查询的主要语义。
```

注意：不要针对具体 topic 写特例。

### 6.2 Prompt 的簇一致性判断需要加强

当前 prompt 虽然要求判断 `needs_split`，但执行不够强。

后续应强调：

```text
如果 member_titles_sample 中出现不同主体、不同诉讼、不同产品、不同日期或明显不同事件，
即使 keep_event=true，也必须 needs_split=true。
```

这仍然是通用规则，不针对某个 topic。

### 6.3 LLM 输入样本需要更有代表性

当前 `member_titles_sample` 只是简单采样。

后续可以优化：

- 优先采不同日期的标题。
- 优先采不同 source 的标题。
- 大簇采更分散的样本。
- 对时间跨度较长的簇，采最早、中间、最晚标题。

这样 LLM 更容易发现混杂事件。

### 6.4 fast / standard 分流规则需要调优

当前 fast 模式在某些 topic 下仍可能把很多事件送进 LLM。

后续可以优化：

- 对高 confidence、时间完整、标题明确的 singleton 直接 `auto_accept`。
- 降低 `low_source_support` 在 fast 模式中的权重。
- 区分“需要 LLM 判断噪声”和“只需要规则排序”的事件。

目标：

```text
fast 模式更快
standard 模式更稳
full 模式用于小样本实验
```

### 6.5 LLM 输出字段可以进一步压缩

当前 LLM 输出字段较完整，但对本地模型来说仍有一定负担。

后续可考虑：

- 让 LLM 只输出变化字段。
- 对 rule accept 事件不生成 LLM 格式字段。
- 把 `decision_reason` 限制得更短。
- 对时间无冲突事件不要求 LLM 输出完整 resolved_time。

### 6.6 增加结果评测脚本

建议新增一个 LLM 层评测脚本，用于快速检查 JSON 结果。

可以统计：

```text
timeline_count
decision_source 分布
final_is_noise 数量
needs_split 数量
needs_merge 数量
fallback 数量
低 decision_confidence 事件
cluster_size > 1 且 needs_split=false 的事件
```

这会比人工翻 JSON 更高效。

### 6.7 增加人工审阅友好输出

可以额外生成一份 Markdown 或 HTML 报告：

```text
时间线节点
展示标题
日期
决断理由
簇内新闻
可疑标记
```

这样更适合给导师快速看效果。

### 6.8 后续再考虑自动拆分/合并

v1 只打：

```text
needs_split
needs_merge
```

暂不自动拆分或合并。

后续如果要做自动拆分，建议单独作为 v2 子任务，不要直接塞进当前决断流程。

原因：

- 自动拆分涉及重新生成 event_id。
- 需要重新分配 articles。
- 会影响前端节点和数据库结构。
- 容易引入不可控错误。

## 7. v1 总结

LLM 层 v1 已经完成从候选事件簇到结构化时间线的完整闭环。

它的核心设计是：

```text
轻量输入
规则分流
通用 prompt
LLM 决断
确定性排序
完整溯源输出
```

当前最重要的优化方向不是重写架构，而是在现有架构上逐步增强：

1. topic relevance 判断
2. needs_split 判断
3. LLM 输入采样
4. 分流规则
5. 结果评测和可视化报告

整体上，v1 已经具备可运行、可展示、可落库、可扩展的基础，可以作为后续 LLM 层优化的起点。
