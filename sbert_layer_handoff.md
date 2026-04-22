# SBERT 事件发现层交接说明

本文档用于把当前 `newsline` 项目中的 SBERT / embedding 事件发现层交接给下一层 LLM 决断层。下一层新开对话时，可以直接把本文档作为上下文使用。

当前日期：2026-04-21

## 1. 本层定位

SBERT 层的职责是把数据库中的新闻标题从“单条新闻”组织成“候选事件簇”。

它不负责最终事实判断，也不负责最终时间线推理。本层输出的是一个结构化的、可追溯的候选事件集合，供下一层 LLM 继续判断：

- 这个簇是否真的是一个事件
- 这个簇是否和用户 topic 同义/同指
- 簇内新闻是否都描述同一个具体事件
- 是否需要拆分、合并、降噪
- 事件之间如何组织成时间线

当前整体流程是：

```text
用户输入 topic
-> LLM 生成多语种 topic alias
-> SQL LIKE 召回候选新闻标题
-> Python 标题匹配过滤
-> Qwen3-Embedding-4B 标题向量化
-> 图链接聚类
-> 构建事件节点 EventNode
-> 写入 JSON 和 MySQL
-> 下一层 LLM 按 topic/run_id 读取并决断
```

## 2. 运行入口

项目根目录：

```bash
cd /Users/hjfl/newsline
```

运行事件发现：

```bash
.venv/bin/python code/script/run_event_discovery.py --topic "Trump"
```

小样本测试可以加 `--limit`：

```bash
.venv/bin/python code/script/run_event_discovery.py --topic "Apple" --limit 200
```

注意：正式全量跑 topic 时不建议加 `--limit`。`--limit` 会限制参与聚类的候选新闻数量，不只是限制终端显示。

评测/检查当前 SBERT 输出：

```bash
.venv/bin/python code/script/eval_event_discovery.py --topic "Trump" --top-k 15
```

指定某一次运行：

```bash
.venv/bin/python code/script/eval_event_discovery.py --topic "Trump" --run-id "Trump_20260421_104016_39d84a87"
```

## 3. 主要模块

核心代码位置：

```text
core/event_discovery/topic_expansion.py
core/event_discovery/pipeline.py
core/event_discovery/encoder.py
core/event_discovery/clustering.py
core/event_discovery/event_builder.py
core/schemas.py
```

命令行脚本：

```text
code/script/run_event_discovery.py
code/script/eval_event_discovery.py
```

下一层读取接口：

```text
core/timeline_reasoning/pipeline.py
```

主要职责：

| 模块 | 职责 |
|---|---|
| `topic_expansion.py` | 调用本地 Ollama `qwen3.5:9b`，把 topic 扩展成多语种 alias |
| `pipeline.py` | 主流程，负责召回、过滤、聚类、JSON 输出和 MySQL 落库 |
| `encoder.py` | 加载 `Qwen/Qwen3-Embedding-4B`，对候选标题做 embedding |
| `clustering.py` | 图链接聚类，包含相似度阈值、时间约束、大簇二次切分 |
| `event_builder.py` | 把聚类结果转换成标准事件节点 |
| `schemas.py` | 定义 `NewsItem`、`EventNode`、`EventDiscoveryResult` 等数据结构 |
| `timeline_reasoning/pipeline.py` | 下一层读取 `event_discovery_events` / `event_discovery_assignments` 的接口 |

## 4. 输入数据

SBERT 层从 MySQL 表 `parser_newsdata` 读取新闻，只读不改。

主要使用字段：

| 字段 | 用途 |
|---|---|
| `id` | 原始新闻 ID |
| `title` | 新闻标题，用于召回、过滤、embedding |
| `source` | 来源，用于事件 source_count 和追溯 |
| `url` | 原文链接，给下一层/前端追溯 |
| `standard_timestamp` | 发布时间备用 |
| `event_timestamp` | spaCy 层解析出的新闻核心事件时间，进入本层后叫 `event_time_anchor` |
| `event_time_start` | spaCy 层解析出的事件时间范围起点 |
| `event_time_end` | spaCy 层解析出的事件时间范围终点 |
| `time_granularity` | 时间粒度 |
| `is_noise` | parser 层噪声标记，目前 SBERT 层不把它作为过滤条件 |

注意：`parser_newsdata.is_noise` 不作为 SBERT 层过滤条件。本层保留输入新闻，自己输出 `system_is_noise` 作为后续 LLM 的参考。

## 5. Topic 多语种召回

当前数据库标题语言中，超过 2% 的主要语言配置为：

```text
en, zh-cn, es, ko, fr, ru, uk, sw
```

配置在：

```text
configs/pipeline_config.py
```

关键配置：

```python
"topic_expansion_langs": ["en", "zh-cn", "es", "ko", "fr", "ru", "uk", "sw"],
"topic_alias_backend": "ollama",
"topic_alias_ollama_url": "http://localhost:11434/api/generate",
"topic_alias_ollama_think": False,
"topic_alias_per_language_limit": 4,
"topic_alias_total_limit": 40,
```

实际使用本地 Ollama 模型：

```text
qwen3.5:9b
```

Topic alias 生成规则：

- LLM 只翻译/转写 topic，不翻译新闻标题。
- 每种目标语言生成 1 到 4 个 alias。
- 专名、品牌、人物、机构等应保留常见原文写法。
- 不使用 Apple 这类个例黑名单。
- topic 层尽量负责扩召回，不做最终语义裁决。
- 可疑 alias 会保留，但在 `topic_alias_details.notes` 中标记。

当前通用 notes：

| note | 含义 |
|---|---|
| `possible_translated_named_entity` | 专名被翻译成非原文形式，可能存在歧义 |
| `very_short_alias` | alias 很短，可能召回较宽 |

例如 `Apple` 可能输出：

```json
{
  "text": "苹果",
  "lang": "zh-cn",
  "priority": "strong",
  "notes": ["possible_translated_named_entity", "very_short_alias"]
}
```

这类 alias 仍会进入 SQL 召回。下一层 LLM 可以根据 notes 判断是否降低信任或剔除相关事件。

## 6. 召回和过滤

SQL 召回方式：

```sql
WHERE title LIKE %alias_1%
   OR title LIKE %alias_2%
   OR ...
```

召回后会在 Python 层再次过滤标题：

- CJK alias 使用包含匹配。
- 拉丁字母/数字 alias 使用词边界匹配。
- 避免 `Apple` 误命中 `grapples` 这类子串。

输出计数：

| 字段 | 含义 |
|---|---|
| `candidate_count` | SQL LIKE 召回出的候选新闻数量 |
| `filtered_count` | Python 标题过滤后进入 embedding/聚类的新闻数量 |

## 7. Embedding 设计

当前 embedding 模型：

```text
Qwen/Qwen3-Embedding-4B
```

只对标题做 embedding，不对正文做 embedding。

原因：

- 当前数据库主要以标题为事件发现入口。
- 标题通常包含事件核心信息。
- 正文 embedding 会显著增加内存和计算成本。
- 正文可能引入更多背景噪声。

当前 prompt 位于：

```text
core/event_discovery/encoder.py
```

核心含义：

```text
为新闻标题生成事件级聚类向量。
目标是标题和标题之间的事件一致性比较，不是 topic-query 和标题之间的检索。
只有当两条标题描述同一现实世界中的单一具体事件时才高度相似。
同一人物、机构、品牌、地点或宽泛主题相关，但不是同一具体事件时，应降低相似度。
忽略媒体立场、措辞差异、标题格式差异和多语种表达差异。
```

这点很重要：当前召回靠 LLM alias + SQL，SBERT embedding 只负责候选标题之间的事件级聚类。

## 8. 图链接聚类

当前已经从 MVP 的 DBSCAN 改为图链接聚类。

基本逻辑：

```text
标题 embedding
-> 两两 cosine similarity
-> 相似度达到阈值才连边
-> 时间差过大则默认不连边
-> 极高语义相似度可以 override 时间约束
-> 连通分量形成候选事件簇
-> 对过大或低凝聚度分量递归提高阈值二次切分
```

主要参数在：

```text
core/event_discovery/clustering.py
```

当前参数：

```python
SIMILARITY_THRESHOLD = 0.80
TIME_WINDOW_DAYS = 30.0
OVERRIDE_SIMILARITY_THRESHOLD = 0.92
OVERSIZED_COMPONENT_LIMIT = 120
COHESION_REFINEMENT_MIN_SIZE = 6
MIN_COMPONENT_EDGE_DENSITY = 0.35
MIN_COMPONENT_AVG_SIMILARITY = 0.84
REFINEMENT_STEP = 0.03
MAX_REFINEMENT_THRESHOLD = 0.95
```

图边 `edge_reason` 有三类：

| edge_reason | 含义 |
|---|---|
| `semantic_only` | 没有可用时间，只按语义相似度连边 |
| `semantic_and_time` | 语义相似且时间差在窗口内 |
| `semantic_override` | 时间差超过窗口，但语义相似度达到 override 阈值 |

下一层 LLM 可重点关注：

- 大簇
- 时间跨度很长的簇
- `semantic_override` 边较多的簇
- `LIVE` / rolling coverage 标题形成的簇

## 9. 事件节点构建

聚类后会生成标准事件对象 `EventNode`。

字段定义在：

```text
core/schemas.py
```

核心字段：

| 字段 | 含义 |
|---|---|
| `event_id` | 事件 ID，格式为 `{run_id}:{topic}_event_XXX` |
| `topic` | 用户输入 topic |
| `member_news_ids` | 簇内新闻 ID 列表 |
| `cluster_size` | 簇大小 |
| `canonical_title` | 代表性标题，从原始标题中选出，不是 LLM 摘要 |
| `representative_news_id` | canonical_title 对应的新闻 ID |
| `event_time_start` | 簇内新闻时间范围最早起点 |
| `event_time_end` | 簇内新闻时间范围最晚终点 |
| `event_time_anchor` | 簇内新闻事件时间锚点的中位时间 |
| `source_count` | 簇内不同来源数量 |
| `confidence` | 启发式质量分数，不是概率 |
| `system_is_noise` | SBERT 层的疑似噪声标记 |
| `noise_reason` | 噪声原因 |

`canonical_title` 选择方式：

```text
计算簇内每条标题与其他成员的平均相似度
选择平均相似度最高的一条原始标题作为代表标题
```

## 10. Confidence 和 Noise

`confidence` 是启发式质量分数，不是机器学习概率。

当前综合三类信息：

```text
语义相似度：55%
簇规模：25%
时间一致性：20%
```

当前噪声规则很保守：

```text
if confidence < 0.55:
    system_is_noise = True
    noise_reason = "low_cluster_confidence"
else:
    system_is_noise = False
```

重要变化：

- 单标题 singleton 不再自动标记为 noise。
- singleton 会作为单新闻事件保留给下一层。
- SBERT 层不删除疑似噪声，只打标。

这样做是为了避免误删真实但报道较少的重要事件。

## 11. JSON 输出

每次运行会输出三个 JSON：

```text
outputs/clustered/{topic}_events.json
outputs/clustered/{topic}_assignments.json
outputs/clustered/{topic}_graph.json
```

### events.json

用于事件级阅读。

顶层字段：

```json
{
  "topic": "Trump",
  "run_id": "Trump_20260421_104016_39d84a87",
  "topic_aliases": [],
  "topic_alias_details": [],
  "candidate_count": 3278,
  "filtered_count": 3223,
  "events": []
}
```

### assignments.json

用于从事件追溯到原始新闻。

顶层字段：

```json
{
  "topic": "Trump",
  "run_id": "...",
  "topic_aliases": [],
  "topic_alias_details": [],
  "candidate_count": 3278,
  "filtered_count": 3223,
  "assignments": []
}
```

每条 assignment 包含：

```json
{
  "news_id": "gdelt_xxx",
  "event_id": "Trump_...:Trump_event_001",
  "title": "...",
  "source": "...",
  "url": "...",
  "event_time_anchor": "2025-09-01 00:00:00",
  "system_is_noise": false,
  "noise_reason": null,
  "cluster_size": 3,
  "canonical_title": "...",
  "run_id": "..."
}
```

### graph.json

用于调试聚类边。

每条边包含：

```json
{
  "left_news_id": "...",
  "right_news_id": "...",
  "left_event_id": "...",
  "right_event_id": "...",
  "similarity": 0.946402,
  "time_gap_days": 3.0,
  "edge_reason": "semantic_and_time",
  "run_id": "..."
}
```

## 12. MySQL 输出表

SBERT 层写入三张表：

```text
event_discovery_events
event_discovery_assignments
event_discovery_graph
```

每次运行生成新的 `run_id`，不会覆盖旧结果。

### event_discovery_events

下一层 LLM 的主要事件级输入表。

字段：

| 字段 | 类型/含义 |
|---|---|
| `id` | 自增主键 |
| `run_id` | 本次运行 ID |
| `event_id` | 事件 ID，run 内唯一 |
| `topic` | 输入 topic |
| `cluster_size` | 簇大小 |
| `canonical_title` | 代表标题 |
| `representative_news_id` | 代表标题对应新闻 ID |
| `member_news_ids` | JSON 字符串，簇内新闻 ID |
| `event_time_start` | 事件时间范围起点 |
| `event_time_end` | 事件时间范围终点 |
| `event_time_anchor` | 事件时间锚点 |
| `source_count` | 来源数量 |
| `confidence` | SBERT 质量分数 |
| `system_is_noise` | SBERT 疑似噪声 |
| `noise_reason` | 噪声原因 |
| `generated_at` | 写入时间 |

索引：

```text
UNIQUE (run_id, event_id)
KEY run_id
KEY topic
KEY event_time_anchor
```

### event_discovery_assignments

事件和新闻的映射表。下一层需要查看簇内成员时用它。

字段：

| 字段 | 含义 |
|---|---|
| `run_id` | 本次运行 ID |
| `topic` | topic |
| `event_id` | 事件 ID |
| `news_id` | 原始新闻 ID |
| `title` | 原始新闻标题 |
| `source` | 来源 |
| `url` | 原文链接 |
| `event_time_anchor` | 该新闻自己的事件时间 |
| `cluster_size` | 所属事件簇大小 |
| `canonical_title` | 所属事件代表标题 |
| `system_is_noise` | 所属事件噪声标记 |
| `noise_reason` | 噪声原因 |
| `generated_at` | 写入时间 |

### event_discovery_graph

图链接边表。主要用于调试或下一层解释聚类依据。

字段：

| 字段 | 含义 |
|---|---|
| `run_id` | 本次运行 ID |
| `topic` | topic |
| `left_news_id` | 左新闻 ID |
| `right_news_id` | 右新闻 ID |
| `left_event_id` | 左新闻所属事件 |
| `right_event_id` | 右新闻所属事件 |
| `similarity` | embedding 相似度 |
| `time_gap_days` | 两条新闻事件时间差 |
| `edge_reason` | 连边原因 |
| `generated_at` | 写入时间 |

## 13. 下一层推荐读取接口

下一层 LLM 建议优先用：

```text
core/timeline_reasoning/pipeline.py
```

已有函数：

```python
get_latest_event_discovery_run_id(topic)
load_event_nodes_for_timeline(topic, run_id=None)
load_event_assignments_for_timeline(run_id)
build_initial_timeline(event_nodes)
run_timeline_reasoning(topic_or_event_nodes, run_id=None)
```

典型用法：

```python
from core.timeline_reasoning.pipeline import (
    load_event_nodes_for_timeline,
    load_event_assignments_for_timeline,
)

run_id, events = load_event_nodes_for_timeline("Trump")
assignments = load_event_assignments_for_timeline(run_id)
```

如果只传 topic，会自动取该 topic 最新 run。

如果要复现某一次运行，应显式传 `run_id`：

```python
run_id, events = load_event_nodes_for_timeline(
    "Trump",
    run_id="Trump_20260421_104016_39d84a87",
)
```

## 14. 下一层 LLM 推荐输入

不建议一次性把所有 events 全塞进 LLM。对于 Trump 这类 topic，事件可能有几千个。

推荐下一层采用分批/分阶段策略。

### 单个事件决断输入

对每个事件簇，建议提供：

```text
topic
run_id
event_id
canonical_title
cluster_size
event_time_start / event_time_end / event_time_anchor
confidence
system_is_noise / noise_reason
member_titles
member_news_ids
source/url 可选
```

LLM 对单个事件可以输出：

```json
{
  "event_id": "...",
  "is_valid_event": true,
  "is_topic_relevant": true,
  "needs_split": false,
  "needs_merge": false,
  "llm_is_noise": false,
  "normalized_event_title": "...",
  "event_time_anchor": "...",
  "reason": "..."
}
```

### 大簇重点检查

优先让 LLM 检查：

- `cluster_size >= 10`
- `event_time_end - event_time_start > 45 days`
- canonical_title 含 `LIVE` / `live` / `timeline` / `updates`
- confidence 较低的大簇
- graph 中 `semantic_override` 边较多的簇

这些指标现在可以用评测脚本快速发现：

```bash
.venv/bin/python code/script/eval_event_discovery.py --topic "Trump" --top-k 20
```

报告输出在：

```text
outputs/reports/{topic}_event_discovery_eval_{run_id}.json
```

### 时间线构建输入

等单事件决断完成后，建议再把保留下来的事件按时间排序，做时间线推理。

初始排序规则已经有：

```python
build_initial_timeline(event_nodes)
```

它按以下优先级排序：

```text
event_time_anchor
event_time_start
event_time_end
event_id
```

## 15. 当前已观察到的质量特点

### Apple

`Apple` 是歧义 topic，暴露出公司 Apple 和水果苹果的冲突。

当前策略：

- topic 层不做 Apple 特例黑名单。
- `苹果`、`苹果公司`、`苹果电脑` 会进入召回。
- 可疑 alias 会在 `topic_alias_details.notes` 中标记。
- 水果相关事件由下一层 LLM 根据 topic 语义和事件内容判断是否剔除。

### Trump

`Trump` 是高频人物 topic，整体召回和多语种 alias 表现较好。

观察到的典型情况：

- 大量 singleton 属于正常现象，因为许多新闻只出现一次。
- 中小簇多数是具体事件。
- 大簇可能来自 live/rolling coverage 或同一议题下的多阶段报道。
- 大簇应交给 LLM 判断是否拆分。

最近一次 Trump 运行示例：

```text
candidate_count: 3278
filtered_count: 3223
event_count: 2276
singleton_count: 1805
max_cluster_size: 39
graph_edges: 1747
```

评测脚本会自动标出大簇、长时间跨度簇、live/rolling 标题簇。

## 16. 评测脚本

脚本：

```text
code/script/eval_event_discovery.py
```

运行：

```bash
.venv/bin/python code/script/eval_event_discovery.py --topic "Trump" --top-k 15
```

可调参数：

```bash
--top-k 20
--large-cluster-min 8
--long-span-days 30
--sample-titles 5
```

输出内容：

- topic alias 数量
- alias detail 数量
- event_count
- assignment_count
- singleton_count / singleton_ratio
- avg_confidence
- cluster_size_distribution
- graph_summary
- diagnostic_event_count
- top_events
- diagnostic_events

报告路径：

```text
outputs/reports/{topic}_event_discovery_eval_{run_id}.json
```

## 17. 当前测试情况

最近已通过：

```bash
.venv/bin/python -m unittest tests.test_event_discovery tests.test_imports tests.test_timeline_reasoning tests.test_active_capabilities
```

结果：

```text
Ran 22 tests
OK
```

## 18. 给下一层 LLM 的设计建议

当前导师要求先做整体系统，因此下一层不必一开始追求最优。建议先做一个可跑通的 LLM 决断层：

1. 输入 `topic` 和 `run_id`。
2. 从 MySQL 读取 events 和 assignments。
3. 对每个事件簇做轻量判断：
   - 是否相关
   - 是否噪声
   - 是否同一具体事件
   - 是否疑似需要拆分
4. 暂时先不真的自动拆分复杂簇，只打标。
5. 对保留事件按时间排序，生成初版 timeline。
6. 输出 JSON，并保留原始 `event_id` / `news_id` / `url` 可追溯。

第一版 LLM 层建议不要一次性处理几千个事件。可以先支持：

- `--limit-events`
- 只处理 `cluster_size >= 2`
- 或只处理评测脚本标出的 `diagnostic_events`
- singleton 先直接保留或只做 topic relevance 判断

## 19. 一句话总结

当前 SBERT 层已经完成从 topic 到候选事件簇的正式流程：它使用本地 LLM 做多语种 topic alias 扩展，用 SQL 召回候选标题，用 Qwen3-Embedding-4B 进行标题级事件相似度建模，用图链接算法生成候选事件簇，并把事件、新闻归属和图边同时写入 JSON 与 MySQL。下一层 LLM 可以直接按 `topic/run_id` 读取事件和成员新闻，在此基础上做事件有效性判断、噪声决断、疑似拆分/合并标记和时间线生成。
