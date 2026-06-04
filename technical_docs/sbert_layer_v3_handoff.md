# SBERT 事件发现层 V3 交接说明

本文档用于交接 `newsline` 项目当前的 SBERT / embedding 事件发现层 V3。它可以作为后续 LLM 决断层、论文方法描述、答辩汇报或新对话继续开发的上下文。

当前日期：2026-04-29

## 1. 本层定位

SBERT 层负责把数据库中的新闻标题从“单条新闻”组织成“候选事件簇”。

它不是最终事实裁判，也不直接生成最终时间线。本层输出的是结构化、可追溯、带诊断信号的候选事件集合，供下一层 LLM 判断：

- 该簇是否是一个真实具体事件
- 该簇是否与用户 topic 相关
- 簇内标题是否描述同一个现实事件
- 是否需要拆分、合并或降噪
- 哪些事件可进入最终时间线

当前整体流程：

```text
用户输入 topic
-> LLM 生成多语种 topic alias
-> SQL LIKE 召回候选新闻标题
-> Python 标题匹配过滤
-> 标题辅助归一化与报道形态识别
-> 折叠归一化重复标题
-> Qwen3-Embedding-4B 标题向量化
-> 图链接聚类与大簇二次切分
-> 小簇合并 pass
-> 构建 EventNode、risk_flags、quality_metrics
-> 写入 JSON 和 MySQL
-> LLM 层按 topic/run_id 读取并决断
```
一句话概括：

```text
SBERT 层负责高召回、可解释、不过度自信的候选事件发现；LLM 层负责最终语义决断和时间线组织。
```

## 2. V3 相比 V2 的核心变化

V2 已经完成了从 DBSCAN 到图链接聚类的升级。V3 主要解决两个问题：

- 大 topic 下仍存在 rolling/live 类大簇粘连
- singleton 和 size=1/2 小簇过多，结果偏碎

V3 新增了四类通用机制，不包含任何 Trump、Apple 等 topic 特例。

### 2.1 标题辅助归一化

新增模块：

```text
core/event_discovery/title_features.py
```

标题归一化不是替换展示标题，也不是生成事件标题。它只生成一个辅助字段 `normalized_title`，用于：

- 折叠媒体包装造成的重复标题
- 计算 duplicate ratio
- 辅助识别 rolling/live 标题
- 让 confidence 不再被重复标题虚高

`canonical_title` 仍然是聚类后选出的原始标题，负责展示和代表事件。

示例：

```text
LIVE: Trump signs bill to end longest US government shutdown | Donald Trump News
Trump signs bill to end longest US government shutdown
```

归一化后会更接近同一个核心标题，但输出时仍保留原始新闻标题和 URL。

### 2.2 rolling/live 报道形态识别

V3 将以下标题形态视为通用风险，而不是某个 topic 的特例：

```text
live
updates
latest
timeline
breaking
as it happened
rolling
直播
快讯
```

这类标题通常不是单一事件，而是滚动报道容器。处理策略：

- 不直接删除
- 不硬编码 topic
- 加 `risk_flags=["rolling_coverage"]`
- 聚类连边时使用更严格时间窗口
- 降低 confidence
- 交给 LLM 层重点 review

rolling coverage 参与连边时的策略：

```python
ROLLING_TIME_WINDOW_DAYS = 3.0
ROLLING_OVERRIDE_SIMILARITY_THRESHOLD = 0.97
```

普通标题仍使用原图链接逻辑：

```python
TIME_WINDOW_DAYS = 30.0
OVERRIDE_SIMILARITY_THRESHOLD = 0.92
```

### 2.3 confidence 拆分为质量指标

V2 的 confidence 是一个启发式总分，容易出现两个问题：

- 大簇因 cluster size 加分而过度自信
- singleton 置信度固定，无法区分真实单篇事件和召回噪声

V3 保留对外字段 `confidence`，但内部新增 `quality_metrics`：

| 字段 | 含义 |
|---|---|
| `semantic_cohesion` | 簇内标题平均语义相似度 |
| `temporal_coherence` | 时间一致性 |
| `support_score` | 基于唯一标题数的支持度 |
| `graph_density` | 簇内图边密度 |
| `duplicate_ratio` | 重复标题占比 |
| `unique_title_count` | 归一化后的唯一标题数 |
| `article_count` | 原始新闻数量 |
| `time_span_days` | 事件时间跨度 |
| `clustered_title_count` | 实际进入 embedding 聚类的标题数 |

同时新增事件级 `risk_flags`：

| flag | 含义 |
|---|---|
| `rolling_coverage` | 标题像滚动报道或 live page |
| `analysis_or_explainer` | 标题像分析、评论、解释性文章 |
| `long_time_span` | 簇内时间跨度过长 |
| `high_duplicate_ratio` | 重复标题比例较高 |
| `low_graph_density` | 图结构稀疏 |
| `low_temporal_coherence` | 时间一致性较差 |

LLM 层不会被迫读取所有质量指标。它主要接收：

```text
confidence + risk_flags + member_titles_sample
```

因此 V3 增强了规则层解释能力，但没有显著增加 LLM 上下文压力。

### 2.4 小簇合并 pass

V3 增加了小簇合并，目标是减少 singleton 和 size=2 小碎片。

位置：

```text
core/event_discovery/clustering.py
```

合并原则非常保守：

- 只允许初始 size <= 2 的小组件参与
- 合并后最大 size <= 5
- rolling coverage 组件不参与合并
- 有时间时，合并后时间跨度必须 <= 7 天
- 缺时间时，相似度阈值更高

当前参数：

```python
SMALL_CLUSTER_MERGE_SOURCE_MAX_SIZE = 2
SMALL_CLUSTER_MERGE_RESULT_MAX_SIZE = 5
SMALL_CLUSTER_MERGE_TIME_WINDOW_DAYS = 7.0
SMALL_CLUSTER_MERGE_AVG_SIMILARITY = 0.86
SMALL_CLUSTER_MERGE_MAX_SIMILARITY = 0.90
SMALL_CLUSTER_MERGE_MISSING_TIME_AVG_SIMILARITY = 0.90
SMALL_CLUSTER_MERGE_MISSING_TIME_MAX_SIMILARITY = 0.94
```

合并产生的图边会写入 graph：

```text
edge_reason = "small_cluster_merge"
```

这个 pass 的目标不是强行降低事件数，而是在不制造大簇的前提下，把明显属于同一事件的小碎片合并回来。

## 3. 主要模块

核心代码：

| 文件 | 职责 |
|---|---|
| `core/event_discovery/topic_expansion.py` | 生成多语种 topic alias |
| `core/event_discovery/pipeline.py` | 主流程，负责召回、过滤、标题特征、聚类、输出和落库 |
| `core/event_discovery/title_features.py` | 标题辅助归一化与标题风险识别 |
| `core/event_discovery/encoder.py` | 加载 `Qwen/Qwen3-Embedding-4B` 并生成标题向量 |
| `core/event_discovery/clustering.py` | 图链接聚类、大簇二次切分、小簇合并 |
| `core/event_discovery/event_builder.py` | 构建 EventNode、confidence、risk_flags、quality_metrics |
| `core/schemas.py` | 定义 NewsItem、EventCluster、EventNode、EventDiscoveryResult |
| `core/timeline_reasoning/pipeline.py` | LLM 层读取 SBERT 结果 |
| `core/timeline_reasoning/event_cards.py` | 将 SBERT 结果压缩为 LLM 输入卡片 |

命令行入口：

```text
code/script/run_event_discovery.py
code/script/eval_event_discovery.py
```

## 4. 运行方式

项目根目录：

```bash
cd /Users/hjfl/newsline
```

运行事件发现：

```bash
.venv/bin/python code/script/run_event_discovery.py --topic "Trump"
```

小样本测试：

```bash
.venv/bin/python code/script/run_event_discovery.py --topic "Apple" --limit 200
```

注意：`--limit` 会限制参与聚类的候选新闻数量，不只是限制终端显示。正式全量实验不要加 `--limit`。

评测最新 run：

```bash
.venv/bin/python code/script/eval_event_discovery.py --topic "Trump" --top-k 15
```

评测指定 run：

```bash
.venv/bin/python code/script/eval_event_discovery.py \
  --topic "Trump" \
  --run-id "Trump_20260429_222422_80de0fbf" \
  --top-k 15
```

## 5. 输入数据

SBERT 层从 MySQL 表 `parser_newsdata` 读取新闻，只读不改。

主要字段：

| 字段 | 用途 |
|---|---|
| `id` | 原始新闻 ID |
| `title` | 新闻标题，用于召回、过滤、embedding |
| `source` | 来源，用于 source_count 和溯源 |
| `url` | 原文链接 |
| `standard_timestamp` | 新闻发布时间备用 |
| `event_timestamp` | spaCy 层解析出的核心事件时间，进入本层后叫 `event_time_anchor` |
| `event_time_start` | 事件时间范围起点 |
| `event_time_end` | 事件时间范围终点 |
| `time_granularity` | 时间粒度 |
| `is_noise` | parser 层噪声标记，目前不作为 SBERT 过滤条件 |

注意：

```text
parser_newsdata.is_noise 不作为 SBERT 层过滤条件。
```

SBERT 层会输出自己的 `system_is_noise`，供 LLM 层参考。

## 6. Topic 多语种召回

当前使用本地 Ollama 模型：

```text
qwen3.5:9b
```

配置位置：

```text
configs/pipeline_config.py
configs/model_config.py
```

主要语言：

```text
en, zh-cn, es, ko, fr, ru, uk, sw
```

topic alias 只用于召回，不直接参与最终语义决断。

alias notes：

| note | 含义 |
|---|---|
| `possible_translated_named_entity` | 专名被翻译成非原文形式，可能有歧义 |
| `very_short_alias` | alias 很短，召回可能较宽 |

## 7. 召回与过滤

SQL 召回：

```sql
WHERE title LIKE %alias_1%
   OR title LIKE %alias_2%
   OR ...
```

Python 层再过滤：

- CJK alias 使用包含匹配
- 拉丁字母/数字 alias 使用词边界匹配
- 避免 `Apple` 命中 `grapples` 这类子串

输出计数：

| 字段 | 含义 |
|---|---|
| `candidate_count` | SQL LIKE 召回出的候选新闻数 |
| `filtered_count` | Python 标题过滤后进入聚类流程的新闻数 |

## 8. Embedding 设计

当前模型：

```text
Qwen/Qwen3-Embedding-4B
```

只对标题做 embedding，不对正文做 embedding。

原因：

- 当前数据库主要以标题为事件发现入口
- 标题通常包含事件核心信息
- 正文 embedding 会显著增加内存与计算成本
- 正文可能引入背景噪声

当前 embedding prompt 位于：

```text
core/event_discovery/encoder.py
```

核心要求：

```text
目标是新闻标题与新闻标题之间的事件一致性比较。
只有当两条标题描述同一现实世界中的单一具体事件时，向量才应高度相似。
同一人物、机构、品牌、地点或宽泛主题相关，但不是同一具体事件时，应降低相似度。
```

## 9. 图链接聚类

基本逻辑：

```text
标题 embedding
-> 计算 cosine similarity
-> 满足相似度阈值和时间约束才连边
-> 极高相似度可 override 普通时间约束
-> 连通分量形成候选事件簇
-> 大簇或低凝聚度簇递归提高阈值二次切分
-> 小簇合并 pass 修复过碎问题
```

主要图边类型：

| edge_reason | 含义 |
|---|---|
| `semantic_only` | 缺少可用时间，只按语义连边 |
| `semantic_and_time` | 语义相似且时间差在窗口内 |
| `semantic_override` | 时间差较大但语义极高 |
| `small_cluster_merge` | V3 小簇合并产生的解释边 |

## 10. EventNode 输出

聚类后生成标准 `EventNode`。

核心字段：

| 字段 | 含义 |
|---|---|
| `event_id` | 事件 ID，格式为 `{run_id}:{topic}_event_XXX` |
| `topic` | 输入 topic |
| `member_news_ids` | 簇内原始新闻 ID |
| `cluster_size` | 簇内原始新闻数量 |
| `canonical_title` | 原始标题中选出的代表标题，不是 LLM 摘要 |
| `representative_news_id` | 代表标题对应的新闻 ID |
| `event_time_start` | 簇内最早事件时间起点 |
| `event_time_end` | 簇内最晚事件时间终点 |
| `event_time_anchor` | 簇内事件时间锚点中位数 |
| `source_count` | 不同来源数量 |
| `confidence` | V3 质量指标加权后的启发式分数 |
| `system_is_noise` | SBERT 层疑似噪声标记 |
| `noise_reason` | 噪声原因 |
| `risk_flags` | 事件风险标记 |
| `quality_metrics` | 事件质量诊断指标 |

`canonical_title` 选择方式仍是：

```text
计算簇内每条标题与其他成员的平均相似度，选择平均相似度最高的一条原始标题。
```

## 11. JSON 和 MySQL 输出

JSON 输出：

```text
outputs/clustered/{topic}_events.json
outputs/clustered/{topic}_assignments.json
outputs/clustered/{topic}_graph.json
```

MySQL 表：

```text
event_discovery_events
event_discovery_assignments
event_discovery_graph
```

V3 对 `event_discovery_events` 增加：

```text
risk_flags LONGTEXT
quality_metrics LONGTEXT
```

`ensure_event_discovery_schema()` 会在运行时自动补列。

## 12. LLM 层读取方式

推荐通过：

```python
from core.timeline_reasoning.pipeline import (
    load_event_nodes_for_timeline,
    load_event_assignments_for_timeline,
)

run_id, events = load_event_nodes_for_timeline("Trump")
assignments = load_event_assignments_for_timeline(run_id)
```

指定 run：

```python
run_id, events = load_event_nodes_for_timeline(
    "Trump",
    run_id="Trump_20260429_222422_80de0fbf",
)
```

LLM 层会将 EventNode 转为 EventCard。V3 的 `risk_flags` 会进入 EventCard，并影响规则分流：

- `rolling_coverage`
- `long_time_span`
- `low_confidence`
- `system_noise`
- `semantic_override_edges`
- `large_cluster`

LLM 输入仍保持轻量，不直接塞入完整 `quality_metrics`。

## 13. 当前 Trump 压力测试结果

最近一次 V3 测试：

```text
run_id: Trump_20260429_222422_80de0fbf
candidate_count: 3278
filtered_count: 3223
event_count: 2016
assignment_count: 3223
singleton_count: 1453
singleton_ratio: 72.07%
max_cluster_size: 23
graph_edges: 1310
```

图边分布：

```json
{
  "semantic_and_time": 946,
  "semantic_override": 120,
  "small_cluster_merge": 244
}
```

对比 V2 / V3 阶段效果：

| 阶段 | event_count | singleton_count | singleton_ratio | max_cluster_size | graph_edges |
|---|---:|---:|---:|---:|---:|
| V2 基线 | 2276 | 1805 | 79.31% | 39 | 1747 |
| V3 第一阶段：rolling/quality | 2261 | 1783 | 78.86% | 23 | 1065 |
| V3 第二阶段：小簇合并 | 2016 | 1453 | 72.07% | 23 | 1310 |

结论：

- 大簇粘连明显改善，最大簇从 39 降到 23
- 小簇过碎明显改善，singleton 从 1805 降到 1453
- 小簇合并没有重新制造大簇，max_cluster_size 仍为 23
- `small_cluster_merge=244`，说明小簇合并机制有效
- rolling/live、长时间跨度、低时间一致性事件会被打 risk flag，交给 LLM 层复核

## 14. 当前仍需注意的问题

### 14.1 rolling coverage 大簇仍存在

例如：

```text
Iran war live...
LIVE: Trump signs bill...
```

这类标题本质上是滚动报道容器，不一定是单一事件。V3 不建议在 SBERT 层继续硬拆太多，而是：

- SBERT 层标记 `rolling_coverage`
- 降低 confidence
- LLM 层判断 `needs_split`
- 最终时间线层决定是否保留、拆分或降噪

### 14.2 singleton 比例仍然较高

V3 后 Trump singleton ratio 约为 72%。这仍偏高，但比 V2 明显改善。

不建议为了降低数字强行合并，因为新闻标题数据里确实存在大量单篇报道事件。后续若要继续优化，应建立人工标注集，而不是只调无监督阈值。

### 14.3 需要多 topic sanity check

Trump 是压力测试，但不应只针对 Trump 调参。

建议再跑：

```bash
.venv/bin/python code/script/run_event_discovery.py --topic "Apple"
.venv/bin/python code/script/run_event_discovery.py --topic "Fed"
.venv/bin/python code/script/run_event_discovery.py --topic "iPhone"
```

重点检查：

- 是否仍有异常大簇
- singleton 是否合理下降
- `small_cluster_merge` 是否过多
- `rolling_coverage` 是否只标记报道形态，而不是误伤普通事件

## 15. 当前阶段判断

SBERT 层 V3 已经完成当前阶段优化目标：

- 多语种 topic alias 召回
- 标题候选过滤
- 标题辅助归一化
- rolling/live 报道形态识别
- Qwen3-Embedding-4B 标题向量化
- 图链接聚类
- 大簇二次切分
- 小簇合并
- confidence 拆分与风险标记
- JSON / MySQL 输出
- LLM 层读取接口
- 无监督诊断脚本

后续主线建议：

```text
SBERT 层功能冻结
-> 用 3-4 个 topic 做 sanity check
-> 增强 eval 脚本输出 risk_flags 和 quality_metrics
-> 将当前 V3 设计写入论文方法章节
-> 主要精力转向 LLM 决断层和最终时间线质量
```
