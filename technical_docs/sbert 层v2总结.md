# SBERT 层 V2 总结

本文档用于总结当前 `newsline` 项目中 SBERT / embedding 事件发现层 V2 的阶段性工作。相比 MVP 阶段，本层已经从单脚本实验逐步整理为一个较完整的事件发现模块，可以作为后续 LLM 决断层和时间线生成层的前置输入。

## 1. 本层目标

SBERT 层的目标是从数据库中的新闻标题中发现“候选事件簇”。

它的输入是用户给定的 `topic`，以及 MySQL 中已经由前置 parser / spaCy 层处理过的新闻数据。

它的输出不是最终时间线，而是供下一层 LLM 判断的结构化候选事件：

- 每个事件包含哪些新闻
- 每个事件的代表标题是什么
- 每个事件的时间锚点是什么
- 每个事件的聚类置信度如何
- 哪些事件可能是噪声
- 哪些新闻之间存在图链接边

因此，本层可以理解为：

```text
从新闻标题中进行事件候选生成，为下一层 LLM 决断提供结构化输入。
```

## 2. V2 相比 MVP 的主要改进

### 2.1 从 DBSCAN 改为图链接聚类

MVP 阶段使用 DBSCAN 聚类。DBSCAN 的问题是：

- 对 `eps` 参数非常敏感。
- 大 topic 下容易出现聚类过粗或过碎。
- DBSCAN 自带 noise 机制，容易把低密度但真实存在的单条事件直接丢掉。
- 难以解释两条新闻为什么被分到同一簇。

V2 改为图链接聚类：

```text
标题 embedding
-> 两两计算相似度
-> 相似度超过阈值才连边
-> 时间差过大则默认不连边
-> 极高相似度可以 override 时间约束
-> 连通分量形成事件簇
```

这种方式的优点是：

- 每条边都有相似度和时间差，可以解释。
- 可以保留图边用于调试。
- singleton 不会被自动删除。
- 更容易针对大 topic 做二次切分。
- 适合作为 LLM 层的可解释前置结果。

### 2.2 增加大簇二次切分机制

图链接聚类存在链式串联问题：

```text
A 像 B，B 像 C，C 像 D
即使 A 和 D 并不像，也可能被放进同一连通分量。
```

这在 `Trump` 这类大 topic 上尤其明显。V2 增加了二次切分机制：

- 对超大连通分量提高相似度阈值重新切分。
- 对中等大小但内部凝聚度较低的分量继续切分。
- 使用边密度和平均相似度判断是否需要细分。

目前这显著缓解了几千条新闻被串成一个大簇的问题。

### 2.3 引入 LLM topic 多语种 alias 扩展

数据库中新闻标题是多语种的，仅用原始 topic 做 SQL 召回会漏掉大量非英文标题。

V2 引入本地 Ollama 模型：

```text
qwen3.5:9b
```

用于将用户输入的 topic 扩展为多语种 alias。当前覆盖数据库中占比超过 2% 的主要语言：

```text
en, zh-cn, es, ko, fr, ru, uk, sw
```

例如 `Trump` 可以扩展为：

```text
Trump
Donald Trump
Donald J. Trump
特朗普
川普
唐纳德·特朗普
트럼프
도널드 트럼프
Трамп
Дональд Трамп
```

topic alias 仅用于召回，不直接参与标题翻译。

### 2.4 去掉面向个例的人工黑名单

在测试 `Apple` 时，曾经出现公司 Apple 和水果 apple 的歧义。早期尝试过针对 `Apple` 做手工清洗，但这种方式泛化性差，不适合作为正式系统设计。

V2 已经去掉面向个例的黑名单，改为通用策略：

- topic 层负责尽量扩召回。
- 不在 topic 翻译阶段做强语义裁决。
- 可疑 alias 保留，但写入 `topic_alias_details.notes`。
- 后续 LLM 层根据事件内容再判断是否相关或是否噪声。

当前通用 notes 包括：

| note | 含义 |
|---|---|
| `possible_translated_named_entity` | 专名被翻译成非原文形式，可能有歧义 |
| `very_short_alias` | alias 很短，可能召回较宽 |

### 2.5 调整 SBERT embedding prompt

之前为了尝试全库 embedding 召回，曾经使用过偏 topic 检索的 embedding prompt。

V2 当前流程已经改为：

```text
LLM topic alias + SQL 召回
-> SBERT 只在候选标题内部做事件聚类
```

因此 embedding prompt 已经改回事件级聚类导向：

```text
目标是比较新闻标题与新闻标题，而不是比较主题查询与新闻标题。
只有当两条标题描述同一现实世界中的单一具体事件时，向量才应高度相似。
同一人物、机构、品牌、地点或宽泛主题相关，但不是同一具体事件时，应降低相似度。
```

这样可以减少“同 topic 但不同事件”的标题被过度拉近。

### 2.6 不再把 singleton 直接标记为 noise

MVP 中 DBSCAN 会天然把离群点标为 noise。V2 中图链接聚类会把没有边的新闻保留为 singleton 事件。

当前策略是：

- singleton 保留为候选事件。
- 不因 `cluster_size == 1` 直接标记为 noise。
- 只有当 `confidence < 0.55` 时才标记为疑似噪声。

这样可以避免误删那些只有一篇报道、但可能真实重要的事件。

### 2.7 增加 JSON 输出和 MySQL 落库

V2 不再只是终端输出或临时实验结果，而是将结果写入结构化文件和数据库。

JSON 输出：

```text
outputs/clustered/{topic}_events.json
outputs/clustered/{topic}_assignments.json
outputs/clustered/{topic}_graph.json
```

MySQL 输出表：

```text
event_discovery_events
event_discovery_assignments
event_discovery_graph
```

这样后续 LLM 层和前端层都可以按 `topic` 或 `run_id` 读取结果。

### 2.8 增加评测和诊断脚本

V2 增加了评测脚本：

```bash
.venv/bin/python code/script/eval_event_discovery.py --topic "Trump" --top-k 15
```

该脚本用于快速检查：

- event 数量
- assignment 数量
- singleton 比例
- cluster size 分布
- 图边数量
- 大簇
- 长时间跨度簇
- live / rolling coverage 簇
- 大簇低置信度风险

它不是严格监督评测，因为目前还没有人工标注集，但已经可以帮助快速发现聚类过粗、过碎和大簇粘连问题。

## 3. 当前整体流程逻辑

当前 SBERT 层的正式流程如下。

### 3.1 输入 topic

用户输入一个 topic，例如：

```text
Trump
Apple
Fed
```

命令：

```bash
.venv/bin/python code/script/run_event_discovery.py --topic "Trump"
```

### 3.2 LLM 生成多语种 alias

系统调用本地 Ollama `qwen3.5:9b`，将 topic 扩展成多语种 alias。

输出分为两类：

```text
topic_aliases
topic_alias_details
```

其中：

- `topic_aliases` 是 SQL 实际使用的去重后 alias 文本。
- `topic_alias_details` 保留每种语言生成了什么 alias，以及可疑 notes。

### 3.3 SQL 召回候选新闻

使用多语种 alias 对 `parser_newsdata.title` 做 SQL LIKE 召回：

```sql
WHERE title LIKE %alias_1%
   OR title LIKE %alias_2%
   OR ...
```

此时得到 `candidate_count`。

### 3.4 Python 层标题过滤

SQL LIKE 召回后再做标题过滤：

- 英文等拉丁字母 alias 使用词边界匹配。
- CJK alias 使用包含匹配。
- 避免 `Apple` 命中 `grapples` 这类子串。

过滤后得到 `filtered_count`。

### 3.5 标题 embedding

对过滤后的标题调用：

```text
Qwen/Qwen3-Embedding-4B
```

只对标题做 embedding，不处理正文。

### 3.6 图链接聚类

计算候选标题之间的相似度矩阵，并根据相似度和时间约束连边。

连边规则大致为：

```text
similarity >= 0.80
且时间差 <= 30 天
```

如果相似度极高：

```text
similarity >= 0.92
```

则可以突破时间窗口限制。

之后用连通分量形成事件簇，并对大簇或低凝聚度簇进行二次切分。

### 3.7 构建 EventNode

每个聚类结果会转换为标准事件节点。

主要字段：

| 字段 | 含义 |
|---|---|
| `event_id` | 事件 ID |
| `topic` | 输入 topic |
| `member_news_ids` | 簇内新闻 ID |
| `cluster_size` | 簇大小 |
| `canonical_title` | 代表标题 |
| `representative_news_id` | 代表标题对应新闻 ID |
| `event_time_start` | 事件时间范围起点 |
| `event_time_end` | 事件时间范围终点 |
| `event_time_anchor` | 事件时间锚点 |
| `source_count` | 新闻来源数量 |
| `confidence` | 启发式置信度 |
| `system_is_noise` | SBERT 层疑似噪声 |
| `noise_reason` | 噪声原因 |

### 3.8 输出结果

结果会同时写入 JSON 和 MySQL。

JSON 用于人工检查和调试。

MySQL 用于下一层 LLM 决断层读取。

## 4. 当前测试现象

### 4.1 Apple

`Apple` 暴露出典型的实体歧义问题：

- Apple 公司
- 水果 apple / 苹果
- Apple 作为文化标题中的普通词

当前系统不在 topic 层硬删这些结果，而是通过 alias notes 和事件内容交给下一层判断。

这符合当前架构：

```text
SBERT 层负责召回和聚类。
LLM 层负责语义决断。
```

### 4.2 Trump

`Trump` 是高频人物 topic，召回量较大。

最近测试结果大致为：

```text
candidate_count: 3278
filtered_count: 3223
event_count: 2276
singleton_count: 1805
max_cluster_size: 39
graph_edges: 1747
```

观察：

- alias 扩展效果较好。
- 大量 singleton 属于合理现象。
- 中小簇多数看起来是具体事件。
- 大簇主要来自 live/rolling coverage 或同一议题多阶段报道。

这说明当前 SBERT 层已经可以产生可用的候选事件，但大 topic 下仍需要 LLM 层做大簇核查。

## 5. 当前已经具备的接口能力

下一层可以通过：

```text
core/timeline_reasoning/pipeline.py
```

读取 SBERT 层结果。

已有接口：

```python
get_latest_event_discovery_run_id(topic)
load_event_nodes_for_timeline(topic, run_id=None)
load_event_assignments_for_timeline(run_id)
build_initial_timeline(event_nodes)
run_timeline_reasoning(topic_or_event_nodes, run_id=None)
```

推荐下一层使用方式：

```python
from core.timeline_reasoning.pipeline import (
    load_event_nodes_for_timeline,
    load_event_assignments_for_timeline,
)

run_id, events = load_event_nodes_for_timeline("Trump")
assignments = load_event_assignments_for_timeline(run_id)
```

如果要固定某一次结果：

```python
run_id, events = load_event_nodes_for_timeline(
    "Trump",
    run_id="Trump_20260421_104016_39d84a87",
)
```

## 6. 当前仍需优化的地方

导师当前要求是先整体做出来，因此以下内容可以作为后续优化方向，不必阻塞下一层 LLM。

### 6.1 大 topic 下的大簇粘连

`Trump` 这类 topic 中，live coverage 或长期议题容易形成大簇。

后续可以优化：

- 更强的时间一致性约束
- 对 live / timeline / updates 标题特殊处理
- 对大簇交给 LLM 判断是否拆分
- 根据 graph 边结构做更细粒度的社区发现

### 6.2 Topic 语义歧义

`Apple` 这类 topic 可能同时指代公司和水果。

当前做法是保留召回，让 LLM 层判断。

后续可以优化：

- 让用户输入更明确的 topic，例如 `Apple Inc.`
- 增加 topic disambiguation 步骤
- 让 LLM 根据 alias notes 判断 topic 语义
- 对明显不同义项的事件打 `llm_is_noise`

### 6.3 召回仍依赖标题 LIKE

当前召回主要依赖 SQL LIKE。

优点是简单、可控、内存压力低。

缺点是：

- 召回依赖 alias 质量。
- 无法召回标题中没有直接出现 alias 的相关新闻。
- 多语种和别名覆盖仍可能不完整。

后续可以探索：

- title 全库 embedding 索引作为补充召回
- BM25 / fulltext index
- alias + embedding hybrid recall

但当前为了控制内存和复杂度，先保留 SQL LIKE 路线是合理的。

### 6.4 缺少人工标注评测集

当前评测脚本主要是无监督诊断。

后续如果要写论文实验，可以建立一小批人工标注样本：

- 同一事件标题对
- 不同事件但同 topic 标题对
- 噪声标题
- 事件簇 gold label

这样可以计算：

- pairwise precision / recall
- cluster purity
- B-cubed precision / recall
- noise detection accuracy

### 6.5 置信度仍是启发式

当前 `confidence` 由语义相似度、簇规模和时间一致性加权得到。

它不是概率，也不是监督训练出来的分数。

后续可以：

- 调整权重
- 针对大簇降低置信度
- 引入 graph density
- 引入标题语言、来源数量等特征
- 让 LLM 给出二次置信度

### 6.6 JSON 和 DB 信息仍可继续统一

当前 JSON 中包含 alias details、candidate_count、filtered_count 等调试信息。

MySQL 结果表主要保存事件、assignment 和 graph。

后续可以考虑增加单独的 run metadata 表，保存：

- run_id
- topic
- topic_aliases
- topic_alias_details
- candidate_count
- filtered_count
- config snapshot
- generated_at

这会让后续复现实验更方便。

## 7. 下一步建议

当前最合理的下一步不是继续打磨 SBERT，而是先做 LLM 决断层的 MVP。

建议下一层先实现：

```text
读取某个 topic/run_id 的 EventNode 和 assignments
-> 对每个事件簇做 LLM 判断
-> 输出是否保留、是否噪声、是否疑似需要拆分
-> 生成初版时间线 JSON
```

第一版可以先简化：

- singleton 先保留或只做 topic relevance 判断。
- 大簇优先交给 LLM 检查。
- 先不自动拆分事件，只标记 `needs_split`。
- 先不做复杂 merge，只标记 `needs_merge`。
- 保留原始 `event_id`、`news_id`、`url`，保证可追溯。

推荐 LLM 层输出字段：

```json
{
  "event_id": "...",
  "is_valid_event": true,
  "is_topic_relevant": true,
  "llm_is_noise": false,
  "needs_split": false,
  "needs_merge": false,
  "normalized_event_title": "...",
  "event_time_anchor": "...",
  "reason": "..."
}
```

## 8. 总体结论

SBERT 层 V2 已经完成了从实验脚本到正式功能层的升级。

目前它已经具备：

- 多语种 topic alias 召回
- 标题候选过滤
- Qwen3-Embedding-4B 标题向量化
- 图链接事件聚类
- 大簇二次切分
- 事件级结构化输出
- 新闻到事件的 assignment
- 图边调试信息
- MySQL 落库
- JSON 输出
- 评测诊断脚本
- 下一层读取接口

当前它已经足够支撑下一层 LLM 决断层开发。

后续优化重点应放在：

- 大 topic 的大簇拆分
- topic 语义歧义处理
- 更稳健的召回策略
- 人工标注评测
- LLM 与 SBERT 结果的协同决断

一句话总结：

```text
SBERT 层 V2 已经能把数据库新闻按 topic 召回并组织成可追溯的候选事件簇；它负责提供结构化、可解释的初步事件发现结果，下一层 LLM 负责最终语义决断和时间线组织。
```
