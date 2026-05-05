# NewsLine

NewsLine 是一个面向多语种新闻标题的事件时间线重构系统，对应毕业设计《基于 SBERT 与轻量大模型的新闻时序重构技术研究》的工程实现。系统以 MySQL 中的新闻标题、来源、链接和标准化时间字段为输入，围绕用户给定 topic 完成候选新闻召回、事件发现、事件裁判、时间线持久化和 Web 可视化展示。

系统采用分层流水线：

```text
MySQL news data
-> spaCy preprocessing and event-time parsing
-> multilingual topic alias expansion
-> Qwen embedding title encoding
-> graph-link event discovery
-> lightweight LLM timeline reasoning
-> MySQL timeline persistence
-> FastAPI static web display
```

该架构将候选事件发现、事件语义裁判和最终展示解耦。embedding 层负责将语义相近的新闻标题组织为候选事件簇；LLM 层负责处理低置信度、长时间跨度、滚动报道和 topic 相关性不确定的事件；数据库层保存事件簇、图边、模型决策、最终时间线节点和文章溯源信息。

## Current Status

当前项目已完成端到端原型：

- 使用 spaCy 处理新闻文本与事件时间字段。
- 使用 MySQL 存储解析后新闻、事件发现结果和最终时间线结果。
- 使用 `Qwen/Qwen3-Embedding-4B` 生成新闻标题向量。
- 使用 topic alias 扩展提升多语种标题召回能力。
- 使用 embedding 相似度图链接方法生成候选事件簇，并记录图边、时间约束和聚类质量指标。
- 使用 `risk_flags` 和 `quality_metrics` 标记事件簇质量、时间一致性、重复率和潜在噪声。
- 使用本地 Ollama `qwen3.5:9b` 对不确定事件进行结构化裁判。
- 使用 FastAPI 提供任务创建、状态轮询、结果读取、历史结果复用和静态页面服务。
- 使用原生 HTML/CSS/JavaScript 实现 topic 输入、模式选择、日期筛选、生成进度、横向时间线和节点详情展示。
- 使用 `pytest` 覆盖事件发现、时间线推理和模块导入等核心逻辑。

## Repository Layout

```text
newsline/
├── configs/                  # model, database, path and pipeline configuration
├── database/                 # MySQL connection helpers and schema notes
├── data_pipeline/            # scraping, normalization and spaCy processing entry points
├── core/
│   ├── event_discovery/      # embedding encoding and graph-link event discovery
│   ├── llm/                  # local Ollama client abstraction
│   └── timeline_reasoning/   # rule routing, LLM judging and timeline persistence
├── code/script/              # CLI scripts, evaluation scripts and web job runner
├── services/                 # FastAPI timeline API
├── frontend/static/          # static HTML/CSS/JS frontend
├── outputs/                  # local generated outputs
├── tests/                    # unit tests and experiments
├── TECHNICAL_REPORT.md       # project technical report
└── README.md
```

## Technology Stack

### Languages

- Python 3.14.x: backend service, NLP processing, event discovery, LLM orchestration and persistence.
- JavaScript: frontend interaction, polling and timeline rendering.
- HTML / CSS: frontend page structure and visual style.
- SQL: MySQL schema, query and persistence logic.

### Backend and Data

- FastAPI: HTTP API and static frontend service.
- Pydantic: request validation.
- PyMySQL: MySQL access.
- MySQL: persistent storage for parsed news, event discovery outputs and timeline outputs.
- NumPy / scikit-learn: vector matrix and similarity processing.
- sentence-transformers / Transformers: embedding model loading and inference.
- Ollama HTTP API: local lightweight LLM inference.

### NLP and Models

- spaCy: text processing and event-time parsing entry point.
- `Qwen/Qwen3-Embedding-4B`: title embedding model.
- `qwen3.5:9b`: topic alias generation and event reasoning model.
- `facebook/nllb-200-distilled-600M`: topic translation support.

## Data Model

The active pipeline reads parsed news records from MySQL. The expected news fields include:

- `id`
- `title`
- `source`
- `url`
- `standard_timestamp`
- `event_timestamp`
- `event_time_start`
- `event_time_end`
- `time_granularity`
- `is_noise`

Core result tables include:

- `event_discovery_events`
- `event_discovery_assignments`
- `event_discovery_graph`
- `timeline_reasoning_runs`
- `timeline_event_decisions`
- `timeline_nodes`
- `timeline_node_articles`

`database/schema.sql` records the main schema definitions. The persistence modules also contain runtime schema checks for result tables.

## Method Overview

### 1. Preprocessing and Time Parsing

The preprocessing layer normalizes news data and extracts event-time fields. The active formal entry point is:

```text
data_pipeline/processors/spacy_pipeline.py
```

This layer provides language-aware processing, base-time normalization and title-level event-time extraction for downstream event discovery.

### 2. Event Discovery

The event discovery layer receives a topic and executes the following steps:

1. Generate multilingual topic aliases.
2. Retrieve candidate news titles from MySQL.
3. Apply title-level relevance filtering.
4. Normalize and deduplicate titles.
5. Encode titles with `Qwen/Qwen3-Embedding-4B`.
6. Compute title similarity matrix.
7. Construct a graph using semantic similarity and event-time constraints.
8. Convert connected components into candidate event clusters.
9. Refine low-cohesion or oversized components.
10. Persist event nodes, article assignments and graph edges.

Main files:

- `core/event_discovery/pipeline.py`
- `core/event_discovery/topic_expansion.py`
- `core/event_discovery/encoder.py`
- `core/event_discovery/clustering.py`
- `core/event_discovery/event_builder.py`
- `core/event_discovery/title_features.py`

### 3. Timeline Reasoning

The timeline reasoning layer transforms candidate event clusters into compact `EventCard` objects. Each card contains representative title, article evidence, event time fields, cluster statistics, confidence score, `risk_flags` and `quality_metrics`.

Rule routing divides events into three categories:

- `auto_accept`: low-risk events accepted by deterministic rules.
- `llm_review`: uncertain events reviewed by the local LLM.
- `rule_reject`: structurally invalid events rejected by deterministic rules.

The LLM returns structured decisions, including topic relevance, final noise status, display title, time anchor, split / merge hint and confidence. Final timeline ordering is computed deterministically by code.

Main files:

- `core/timeline_reasoning/models.py`
- `core/timeline_reasoning/event_cards.py`
- `core/timeline_reasoning/filters.py`
- `core/timeline_reasoning/llm_judge.py`
- `core/timeline_reasoning/ordering.py`
- `core/timeline_reasoning/persistence.py`
- `core/timeline_reasoning/pipeline.py`
- `core/timeline_reasoning/topic_profile.py`

### 4. Web Display

The display layer is served by FastAPI and implemented with static HTML/CSS/JavaScript.

Frontend capabilities:

- topic input;
- `fast` / `standard` / `full` reasoning mode;
- dataset date range selector;
- force-regenerate switch;
- recent timeline result list;
- job progress and cancellation;
- horizontal interactive timeline;
- article hover preview;
- node detail drawer with source articles and model decision metadata.

Main files:

- `services/timeline_api.py`
- `code/script/run_timeline_web_job.py`
- `frontend/static/index.html`
- `frontend/static/styles.css`
- `frontend/static/app.js`

## Environment Setup

Install dependencies:

```bash
cd /Users/hjfl/newsline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure MySQL connection:

```text
configs/db_config.py
```

Prepare the local LLM service:

```bash
ollama pull qwen3.5:9b
ollama serve
```

## Running the Pipeline

Run event discovery:

```bash
.venv/bin/python code/script/run_event_discovery.py --topic "Trump"
```

Run event discovery with a date range:

```bash
.venv/bin/python code/script/run_event_discovery.py \
  --topic "Apple" \
  --start-date 2025-06-01 \
  --end-date 2026-04-01
```

Run timeline reasoning:

```bash
.venv/bin/python code/script/run_timeline_reasoning.py \
  --topic "Apple" \
  --mode standard \
  --llm-batch-size 4 \
  --llm-timeout-seconds 300
```

Run the web-facing full job:

```bash
.venv/bin/python code/script/run_timeline_web_job.py \
  --topic "Trump" \
  --mode standard \
  --start-date 2025-06-01 \
  --end-date 2026-04-01
```

## Running the Web Frontend

Start the FastAPI service:

```bash
cd /Users/hjfl/newsline
source .venv/bin/activate
uvicorn services.timeline_api:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Important API routes:

```text
GET  /api/health
POST /api/timeline/jobs
GET  /api/timeline/jobs/{job_id}/status
POST /api/timeline/jobs/{job_id}/cancel
GET  /api/timeline/jobs/{job_id}/result
GET  /api/timeline/results/{reasoning_run_id}
GET  /api/timeline/recent?limit=6
```

## Outputs

Event discovery JSON outputs:

```text
outputs/clustered/{topic}_events.json
outputs/clustered/{topic}_assignments.json
outputs/clustered/{topic}_graph.json
```

Timeline reasoning JSON outputs:

```text
outputs/timeline/
```

The Web frontend reads formal results from MySQL:

```text
timeline_reasoning_runs
timeline_nodes
timeline_node_articles
```

## Tests

Run the current test suite:

```bash
.venv/bin/python -m pytest
```

## Project Documents

- `TECHNICAL_REPORT.md`
- `sbert_layer_v3_handoff.md`
- `llm_layer_v3_handoff.md`
- `frontend_display_layer_v2_handoff.md`
- `cross_language_retrieval_schemes.md`

## Final Result

The current repository contains a runnable end-to-end prototype. It can read parsed news records from MySQL, discover topic-related candidate events through multilingual recall and embedding graph linkage, apply lightweight LLM reasoning to uncertain events, persist traceable timeline results, and provide a browser interface for timeline generation and inspection.
