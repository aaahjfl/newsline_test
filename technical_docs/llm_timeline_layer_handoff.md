# LLM 时间线决断层交接说明

本文档用于把当前 `newsline` 项目中的 LLM 时间线决断层交接给下一轮优化使用。

当前日期：2026-04-22

## 1. 本层定位

LLM 时间线决断层位于 SBERT / embedding 事件发现层之后。

SBERT 层已经把原始新闻标题组织成候选事件簇，本层不重新做事件发现，也不直接处理海量原始新闻。本层的职责是：

- 读取 SBERT 层输出的候选事件簇
- 用规则先对事件簇做分流和风险标记
- 将轻量化事件卡片送给本地 LLM 判断
- 判断事件是否保留、是否相关、是否噪声、是否疑似需要拆分/合并
- 选择或修正最终时间字段
- 按确定性规则生成最终时间线顺序
- 把时间线节点和簇内全部新闻写入 JSON / MySQL

核心原则：

```text
SBERT 层负责发现候选事件簇
LLM 层负责裁判和补充决断
最终全局排序由代码稳定生成
```

LLM 不是一次性生成完整时间线，而是作为“轻量时序裁判”参与局部判断。

## 2. 当前整体流程

```text
用户输入 topic
-> 读取 event_discovery_events
-> 读取 event_discovery_assignments
-> 读取 event_discovery_graph 轻量诊断信息
-> 构造 EventCard
-> 规则分流：auto_accept / llm_review / rule_reject
-> 对 llm_review 事件调用 qwen3.5:9b
-> 得到 EventDecision
-> 用规则确定最终时间线 order_index
-> 回挂簇内全部原始新闻 articles
-> 输出完整 JSON
-> 非 dry-run 时写入 MySQL 四张 timeline_* 表
```

## 3. 运行入口

项目根目录：

```bash
cd /Users/hjfl/newsline
```

建议先激活虚拟环境：

```bash
source /Users/hjfl/newsline/.venv/bin/activate
```

运行 LLM 时间线决断层：

```bash
.venv/bin/python code/script/run_timeline_reasoning.py --topic "Apple" --mode fast --limit-events 20 --dry-run --llm-batch-size 1 --llm-timeout-seconds 60
```

正式写入 MySQL 时去掉 `--dry-run`：

```bash
.venv/bin/python code/script/run_timeline_reasoning.py --topic "Apple" --mode fast --limit-events 20 --llm-batch-size 1 --llm-timeout-seconds 60
```

指定某一次 SBERT 事件发现运行：

```bash
.venv/bin/python code/script/run_timeline_reasoning.py \
  --topic "Apple" \
  --run-id "Apple_20260421_103229_832fa474" \
  --mode fast \
  --limit-events 20 \
  --dry-run
```

## 4. CLI 参数

脚本位置：

```text
code/script/run_timeline_reasoning.py
```

参数：

| 参数 | 含义 |
|---|---|
| `--topic` | 要生成时间线的主题 |
| `--run-id` | 指定 SBERT 层 run_id；不传则自动取该 topic 最新一次 |
| `--mode` | 决断模式：`fast` / `standard` / `full` |
| `--limit-events` | 调试时限制处理事件数 |
| `--dry-run` | 只输出 JSON，不写 MySQL |
| `--llm-batch-size` | 每次发给 LLM 的事件数；当前默认 1，优先稳定 |
| `--llm-timeout-seconds` | 单次 LLM 请求超时时间 |

## 5. 三种运行模式

### fast

只把高风险事件送给 LLM，低风险事件走规则保留。

适合：

- 小样本调试
- 快速看 JSON 输出结构
- 避免本地模型负载过高

### standard

更积极地把中等风险事件送给 LLM。

适合：

- 初版系统演示
- 对 topic relevance 和噪声判断要求更高的场景

### full

所有事件都送给 LLM。

适合：

- 小数据实验
- 论文对照实验
- 不适合直接跑大 topic 全量

## 6. 主要代码模块

核心目录：

```text
core/timeline_reasoning/
```

| 文件 | 职责 |
|---|---|
| `pipeline.py` | 主流程入口；读取 SBERT 输出、调用分流/LLM/排序/落表 |
| `models.py` | 定义 `EventCard`、`EventDecision`、`TimelineRecord`、`TimelineReasoningResult` |
| `event_cards.py` | 将 `EventNode + assignments` 转为轻量事件卡片 |
| `filters.py` | 规则分流和风险标记 |
| `prompts.py` | 通用 LLM prompt |
| `llm_judge.py` | 调用 Ollama、解析 JSON、处理超时和 fallback |
| `ordering.py` | 生成最终稳定时间线顺序 |
| `persistence.py` | MySQL 建表和写入 |

相关通用 LLM 客户端：

```text
core/llm/ollama_client.py
```

配置：

```text
configs/model_config.py
configs/pipeline_config.py
```

## 7. LLM 输入设计

LLM 不直接接收整个簇的全部新闻。

每个事件先被压缩成 `EventCard`。默认输入给 LLM 的字段包括：

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

注意：

```text
articles 不进入 LLM 输入
```

这样做是为了：

- 降低上下文长度
- 避免本地模型过慢或超时
- 让 LLM 聚焦于事件级裁判

但是最终输出的 timeline 节点会重新挂回完整 `articles`，用于前端展开和证据溯源。

## 8. system_is_noise 的处理

`system_is_noise` 会进入 LLM 输入。

它的含义是：

```text
SBERT 层基于聚类质量给出的疑似噪声标记
```

它不是最终结论。

Prompt 中明确说明：

```text
system_is_noise is an upstream reference signal, not a final verdict. You may overturn it.
```

当前策略：

- `system_is_noise = true` 会触发 `llm_review`
- 不会被规则层直接硬删
- LLM 可保留，也可判为 `final_is_noise = true`

## 9. 规则分流

分流函数：

```text
core/timeline_reasoning/filters.py
route_event_card()
```

三种结果：

| 路由 | 含义 |
|---|---|
| `auto_accept` | 规则认为风险较低，直接保留 |
| `llm_review` | 存在风险，交给 LLM 判断 |
| `rule_reject` | 极少数坏数据，规则直接剔除 |

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

其中：

- `fast` 模式只 review 更高风险的事件
- `standard` 模式会 review 更多中风险事件
- `full` 模式所有事件都 review

## 10. LLM Prompt 设计

Prompt 文件：

```text
core/timeline_reasoning/prompts.py
```

版本：

```text
timeline_reasoning_v1
```

Prompt 是通用的，不针对任何具体 topic、人物、公司、国家或事件写特例。

核心要求：

- 判断候选是否是具体现实事件
- 判断是否与 topic 相关
- 判断是否最终作为噪声
- 从输入时间字段中选择最可信时间
- 如果输入不支持精确日期，不允许编造
- 输出严格 JSON
- `system_is_noise` 只是参考，不是最终结论
- singleton 不应仅因来源少而被自动剔除
- rolling coverage 或混合事件应标记 `needs_split=true`

输出形状：

```json
{
  "decisions": [
    {
      "event_id": "string",
      "keep_event": true,
      "is_topic_relevant": true,
      "final_is_noise": false,
      "needs_split": false,
      "needs_merge": false,
      "display_title": "string or null",
      "resolved_time_start": "YYYY-MM-DD HH:MM:SS or null",
      "resolved_time_end": "YYYY-MM-DD HH:MM:SS or null",
      "resolved_time_anchor": "YYYY-MM-DD HH:MM:SS or null",
      "decision_confidence": 0.0,
      "time_confidence": 0.0,
      "decision_reason": "short reason"
    }
  ]
}
```

## 11. Ollama 调用

当前模型：

```text
qwen3.5:9b
```

配置位置：

```text
configs/model_config.py
```

LLM 层 Ollama URL：

```text
configs/pipeline_config.py
timeline_reasoning_ollama_url = "http://127.0.0.1:11434/api/generate"
```

注意：LLM 层现在默认使用 `127.0.0.1`，不是 `localhost`。

原因：

```text
在 macOS + 代理 / VPN / 虚拟网卡环境下，localhost 有时可能被代理设置或解析行为影响。
```

请求前会先访问：

```text
http://127.0.0.1:11434/api/tags
```

做健康检查。如果连不上，会直接 fallback，不再长时间卡住。

当前调用设置：

```text
stream = false
think = false
num_ctx = 8192
num_predict = 1024
temperature = 0.0
```

如果模型仍输出可见 `<think>...</think>`，解析前会自动剥离。

## 12. 超时与 fallback

文件：

```text
core/timeline_reasoning/llm_judge.py
```

当前策略：

- 默认 `llm_batch_size = 1`
- batch > 1 时，如果超时，会自动拆成更小批次
- 单条事件仍超时或解析失败时，不中断整次运行
- 会生成 `decision_source = "llm_fallback"` 的兜底决断

兜底逻辑：

- 如果事件带 `system_is_noise`
- 或是 rolling coverage
- 或缺失核心字段

则更倾向于 `final_is_noise = true`。

否则保守保留，但置信度较低。

## 13. 最终排序

排序文件：

```text
core/timeline_reasoning/ordering.py
```

最终全局顺序不由 LLM 直接生成，而由代码稳定排序。

排序优先级：

```text
resolved_time_anchor
resolved_time_start
resolved_time_end
event_time_anchor
event_time_start
event_time_end
event_id
```

这样做的原因：

- 排序可复现
- 方便调试
- 方便后续计算 Kendall's tau
- 符合“规则骨架 + LLM 裁判”的设计

## 14. JSON 输出

无论是否 `--dry-run`，都会输出 JSON。

目录：

```text
/Users/hjfl/newsline/outputs/timeline/
```

本次示例输出：

```text
/Users/hjfl/newsline/outputs/timeline/Apple_timeline_Apple_timeline_20260421_235520_bda1b1d1.json
```

JSON 顶层字段：

```text
topic
discovery_run_id
reasoning_run_id
model_name
mode
prompt_version
generated_at
status
summary
timeline
decisions
decision_contexts
output_paths
```

### summary

运行统计：

```text
input_event_count
review_event_count
accepted_event_count
rejected_event_count
```

### timeline

最终时间线节点，前端展示主要读取这里。

每个节点包含：

```text
reasoning_run_id
discovery_run_id
topic
event_id
order_index
canonical_title
display_title
event_time_start
event_time_end
event_time_anchor
resolved_time_start
resolved_time_end
resolved_time_anchor
display_date
cluster_size
source_count
member_news_ids
confidence
system_is_noise
noise_reason
decision_source
keep_event
is_topic_relevant
final_is_noise
needs_split
needs_merge
decision_confidence
time_confidence
decision_reason
risk_flags
articles
```

其中 `articles` 是簇内全部原始新闻：

```text
news_id
title
source
url
event_time_anchor
cluster_size
canonical_title
system_is_noise
noise_reason
```

### decisions

每个事件的决断记录，包括被保留和被剔除的事件。

### decision_contexts

每个事件决断时使用的上下文，包括完整 EventCard 信息。

这个字段主要用于：

- 调试
- prompt 优化
- 论文解释

## 15. MySQL 输出表

如果运行时使用了 `--dry-run`：

```text
不会写 MySQL
只写 JSON
```

正式运行去掉 `--dry-run` 后，会写入四张表：

```text
timeline_reasoning_runs
timeline_event_decisions
timeline_nodes
timeline_node_articles
```

### timeline_reasoning_runs

意义：

```text
记录每一次 LLM 时间线决断运行
```

一行对应一次运行。

主要字段：

```text
reasoning_run_id
discovery_run_id
topic
model_name
mode
prompt_version
input_event_count
review_event_count
accepted_event_count
rejected_event_count
status
config_json
generated_at
```

用途：

```sql
SELECT *
FROM timeline_reasoning_runs
WHERE topic = 'Apple'
ORDER BY generated_at DESC
LIMIT 5;
```

### timeline_event_decisions

意义：

```text
保存每个事件的完整决断记录
```

包括：

- 规则保留的事件
- LLM 判断的事件
- LLM fallback 的事件
- 被剔除的事件

主要字段：

```text
reasoning_run_id
discovery_run_id
topic
event_id
canonical_title
event_time_start
event_time_end
event_time_anchor
cluster_size
source_count
confidence
system_is_noise
noise_reason
risk_flags
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
generated_at
```

用途：

```text
调试和论文分析，查看每个事件为什么被保留、剔除或 fallback。
```

### timeline_nodes

意义：

```text
最终时间线节点表
```

只保存最终进入时间线的事件，也就是 `keep_event = true` 的事件。

主要字段：

```text
reasoning_run_id
discovery_run_id
topic
event_id
order_index
canonical_title
display_title
event_time_start
event_time_end
event_time_anchor
resolved_time_start
resolved_time_end
resolved_time_anchor
display_date
cluster_size
source_count
member_news_ids
confidence
system_is_noise
noise_reason
decision_source
keep_event
is_topic_relevant
final_is_noise
needs_split
needs_merge
decision_confidence
time_confidence
decision_reason
risk_flags
generated_at
```

用途：

```text
前端主时间线主要读取这张表。
```

典型查询：

```sql
SELECT *
FROM timeline_nodes
WHERE reasoning_run_id = 'Apple_timeline_20260421_235520_bda1b1d1'
ORDER BY order_index;
```

### timeline_node_articles

意义：

```text
每个时间线节点下面挂载的原始新闻明细
```

也就是“canonical_title / display_title 后面展开显示的簇内全部新闻”。

主要字段：

```text
reasoning_run_id
discovery_run_id
topic
event_id
news_id
title
source
url
event_time_anchor
cluster_size
canonical_title
system_is_noise
noise_reason
sort_index
generated_at
```

用途：

```text
前端点击时间线节点时，读取这张表展示原始新闻证据链。
```

典型查询：

```sql
SELECT title, source, url, event_time_anchor
FROM timeline_node_articles
WHERE reasoning_run_id = 'Apple_timeline_20260421_235520_bda1b1d1'
  AND event_id = 'Apple_20260421_103229_832fa474:Apple_event_008'
ORDER BY sort_index;
```

## 16. 与 SBERT 层表的关系

LLM 层读取以下 SBERT 层表：

```text
event_discovery_events
event_discovery_assignments
event_discovery_graph
```

含义：

| 表 | 含义 |
|---|---|
| `event_discovery_events` | SBERT 层候选事件簇 |
| `event_discovery_assignments` | 事件簇与原始新闻的映射 |
| `event_discovery_graph` | 图链接边，主要用于风险判断和调试 |

LLM 层输出：

```text
timeline_reasoning_runs
timeline_event_decisions
timeline_nodes
timeline_node_articles
outputs/timeline/*.json
```

## 17. 查看 JSON 效果的命令

查看最新 JSON：

```bash
ls -t outputs/timeline/*.json | head -5
```

查看 summary：

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
path = max(Path("outputs/timeline").glob("*.json"), key=lambda p: p.stat().st_mtime)
data = json.loads(path.read_text(encoding="utf-8"))
print(path)
print(json.dumps(data["summary"], ensure_ascii=False, indent=2))
PY
```

查看时间线标题：

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
path = max(Path("outputs/timeline").glob("*.json"), key=lambda p: p.stat().st_mtime)
data = json.loads(path.read_text(encoding="utf-8"))
for item in data["timeline"]:
    print(item["order_index"], item["display_date"], item["display_title"])
PY
```

查看时间线节点及展开新闻：

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
path = max(Path("outputs/timeline").glob("*.json"), key=lambda p: p.stat().st_mtime)
data = json.loads(path.read_text(encoding="utf-8"))

for item in data["timeline"]:
    print("\\n#", item["order_index"], item["display_date"], item["display_title"])
    print("decision:", item["decision_source"], "noise:", item["final_is_noise"], "split:", item["needs_split"])
    print("reason:", item["decision_reason"])
    print("articles:", len(item["articles"]))
    for article in item["articles"]:
        print("  -", article.get("source"), "|", article.get("event_time_anchor"), "|", article.get("title"))
PY
```

## 18. 当前测试情况

已通过：

```bash
.venv/bin/python -m unittest tests.test_timeline_reasoning tests.test_imports
```

结果：

```text
Ran 10 tests
OK
```

此前完整核心测试也通过：

```bash
.venv/bin/python -m unittest tests.test_event_discovery tests.test_imports tests.test_timeline_reasoning tests.test_active_capabilities
```

结果：

```text
Ran 25 tests
OK
```

## 19. 当前一次 Apple dry-run 示例

运行命令：

```bash
.venv/bin/python code/script/run_timeline_reasoning.py \
  --topic "Apple" \
  --mode fast \
  --limit-events 20 \
  --dry-run \
  --llm-batch-size 1 \
  --llm-timeout-seconds 60
```

输出：

```text
topic: Apple
discovery_run_id: Apple_20260421_103229_832fa474
reasoning_run_id: Apple_timeline_20260421_235520_bda1b1d1
model_name: qwen3.5:9b
mode: fast
input_event_count: 19
review_event_count: 19
accepted_event_count: 19
rejected_event_count: 0
```

JSON：

```text
/Users/hjfl/newsline/outputs/timeline/Apple_timeline_Apple_timeline_20260421_235520_bda1b1d1.json
```

## 20. 已观察到的问题

### 1. Apple topic 歧义

示例：

```text
Chappell Roan Apple dance, and other Primavera moments
```

这个标题中的 Apple 可能不是 Apple 公司，而是歌曲、文化事件或普通词命中。

当前 LLM 没有剔除它。

原因：

- 当前 prompt 对 topic 歧义的要求还不够强
- LLM 可能把字符串命中当成相关

优化建议：

```text
Prompt 增加通用歧义判断：
如果 topic 是多义词，不要仅凭字面重合判断相关。
应根据 canonical_title 和 member_titles_sample 判断它是否与用户查询的主实体或主语义一致。
```

注意：不要写 Apple 特例，保持通用。

### 2. 混簇没有被充分标记 needs_split

示例：

```text
Musk sues Apple, OpenAI over alleged AI competition suppression
```

该簇中混入了：

```text
Apple sued over use of copyrighted books to train Apple Intelligence
Apple sued by authors over use of books in AI training
```

这些和 Musk 起诉 Apple/OpenAI 不是同一个具体事件。

当前 LLM 没有标记 `needs_split=true`。

原因：

- LLM 主要看 canonical_title 和少量样本
- prompt 对“同一具体事件”的约束仍需加强

优化建议：

```text
如果 member_titles_sample 中出现不同诉讼、不同产品、不同主体、不同时间或明显不同事件，
即使 keep_event=true，也必须 needs_split=true。
```

### 3. fast 模式下 review_event_count 仍可能偏高

Apple 示例中：

```text
input_event_count: 19
review_event_count: 19
```

说明当前 fast 模式仍然把这批事件全部送给了 LLM。

原因可能是：

- 这些事件大多是 singleton
- confidence 或 source_count 触发了风险标记

优化建议：

- 调整 `filters.py` 中 fast 模式规则
- 对 singleton 高质量标题可先 rule accept
- 降低 `low_source_support` 对 fast 模式的影响

### 4. LLM 性能较慢

当前本地模型：

```text
qwen3.5:9b
```

单条 batch 稳定但慢。

优化建议：

- 先保持 `--llm-batch-size 1` 保证稳定
- 本地模型确认稳定后再尝试 `--llm-batch-size 2`
- 减少 prompt 文本
- 缩短输出字段或只让 LLM 输出必要字段
- 对低风险事件走规则，不送 LLM

## 21. 下一轮优化优先级

建议按以下顺序优化：

1. 强化 prompt 的 topic relevance 判断，解决多义词 / 字符串误命中。
2. 强化 prompt 的 cluster consistency 判断，要求混簇标记 `needs_split=true`。
3. 调整 fast / standard 分流规则，减少不必要 LLM 调用。
4. 优化 `member_titles_sample` 采样策略，让大簇样本更能暴露混簇。
5. 增加一个评测脚本，统计：
   - `decision_source` 分布
   - `needs_split` 数量
   - `final_is_noise` 数量
   - `cluster_size > 1` 的问题簇
6. 后续再做自动拆分 / 合并，不建议下一步立刻做。

## 22. 一句话总结

当前 LLM 时间线决断层已经完成第一版闭环：它能从 SBERT 层读取候选事件簇，构造轻量 EventCard，基于规则和 qwen3.5:9b 做事件保留、相关性、噪声、时间字段和拆分风险判断，按确定性规则生成最终时间线，并输出完整 JSON；非 dry-run 时会写入四张 MySQL timeline 表。下一轮优化重点应放在通用 prompt 的 topic 歧义判断、混簇识别和减少不必要 LLM 调用上。
