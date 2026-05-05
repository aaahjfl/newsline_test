# NewsLine

NewsLine 是一个面向多语种新闻标题的事件时间线重构系统，也是毕业设计《基于 SBERT 与轻量大模型的新闻时序重构技术研究》的工程原型。系统从 MySQL 中读取新闻标题、来源、链接和标准化时间字段，围绕用户输入的 topic 自动完成候选新闻召回、事件发现、LLM 事件裁判、时间线落库和 Web 可视化展示。

项目当前不是“直接让大模型生成一条时间线”，而是采用可追溯的分层流水线：

```text
news data in MySQL
-> spaCy-based preprocessing / event time parsing
-> multilingual topic alias expansion
-> Qwen embedding title encoding
-> graph-link event discovery
-> lightweight LLM timeline reasoning
-> MySQL timeline persistence
-> FastAPI + static frontend display
```

核心设计目标是把高召回的候选事件发现和低成本的语义裁判拆开：SBERT / embedding 层负责把相似新闻标题组织成可检查的候选事件簇，LLM 层只处理不确定事件，最终结果保留事件簇、新闻溯源、风险标记、模型裁判结果和数据库记录。

## Current Status

当前主流程已经形成闭环：

- 使用 spaCy 主线完成多语种新闻文本处理和事件时间解析，HeidelTime 相关代码仅作为历史兼容参考保留。
- 使用本地 MySQL 存储原始新闻、解析后新闻、事件发现结果和最终时间线结果。
- 使用 `Qwen/Qwen3-Embedding-4B` 对新闻标题进行稠密向量化。
- 使用多语种 topic alias 扩展增强跨语言召回，包含 Ollama topic alias 与 NLLB 翻译辅助能力。
- 使用图链接聚类替代早期 DBSCAN 方案，保留 singleton，记录相似度边，并结合时间窗口约束控制错误合并。
- 对大簇、滚动报道、长时间跨度、低置信度、低图密度等情况生成 `risk_flags` 和 `quality_metrics`。
- 使用本地 Ollama `qwen3.5:9b` 作为轻量 LLM 裁判，对不确定事件进行保留、降噪、相关性、标题和时间锚点判断。
- 使用 FastAPI 提供 Web API，用静态 HTML/CSS/JavaScript 实现交互式前端。
- 前端支持 topic 输入、fast / standard / full 模式、日期范围筛选、历史结果复用、重新生成、最近时间线记录、横向时间线、节点 hover 预览和详情抽屉。
- 已有单元测试覆盖事件发现、LLM 决策路由、导入能力和核心数据模型。

## Repository Layout

```text
newsline/
├── configs/                  # model, database, path and pipeline configuration
├── database/                 # MySQL connection helpers and schema notes
├── data_pipeline/            # scraping, normalization and spaCy time parsing entry points
├── core/
│   ├── event_discovery/      # embedding title encoding and graph-link event discovery
│   ├── llm/                  # local Ollama client abstraction
│   └── timeline_reasoning/   # rule + LLM timeline reasoning layer
├── code/script/              # runnable CLI, evaluation and web job scripts
├── services/                 # FastAPI timeline API
├── frontend/static/          # plain HTML/CSS/JS frontend
├── outputs/                  # local JSON outputs and generated artifacts
├── tests/                    # unit tests and experiments
├── archive_mvp/              # archived MVP-era experiments
├── TECHNICAL_REPORT.md       # project technical report
└── README.md
```

## Technology Stack

### Development Language

- Python 3.14.x for backend, NLP pipeline, event discovery, LLM orchestration and persistence.
- JavaScript, HTML and CSS for the frontend display layer.
- SQL for MySQL schema and query logic.

### Backend and Data Stack

- FastAPI: timeline job API and static frontend service.
- Pydantic: API request validation.
- PyMySQL: MySQL access.
- MySQL: persistent storage for news data, event discovery outputs and timeline outputs.
- NumPy / scikit-learn / sentence-transformers / Transformers: embedding and numerical processing.
- Ollama HTTP API: local lightweight LLM inference.

### NLP and Model Stack

- spaCy: active text processing and time parsing pipeline.
- `Qwen/Qwen3-Embedding-4B`: current title embedding model.
- `qwen3.5:9b`: current local LLM model for topic alias generation and timeline reasoning.
- `facebook/nllb-200-distilled-600M`: topic translation support.

The original proposal mentioned HeidelTime, DBSCAN, Qwen3-8B and Streamlit. The current implementation has evolved to spaCy, graph-link clustering, `qwen3.5:9b` through Ollama, and FastAPI + static frontend.

## Data Model

The active pipeline assumes the parsed news table contains fields such as:

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

`database/schema.sql` contains the current schema notes and base table definitions. Some tables are also ensured at runtime by the persistence layer to keep older local databases compatible.

## Method Overview

### 1. Preprocessing and Time Parsing

The current active preprocessing route is spaCy-based. It normalizes news titles, detects language, extracts or standardizes event time fields, and writes parsed records into MySQL. The formal entry point is:

```text
data_pipeline/processors/spacy_pipeline.py
```

The older HeidelTime-era files are retained in `code/data_pipeline/` for comparison and compatibility, but they are no longer the main technical route.

### 2. Event Discovery Layer

The event discovery layer receives a topic and retrieves candidate news titles from MySQL. It expands the topic into multilingual aliases, filters candidates, deduplicates normalized titles, encodes titles with the embedding model, and builds a graph:

- graph nodes are candidate news titles;
- graph edges connect semantically similar titles;
- time window constraints suppress links between distant reports;
- very high similarity can override time distance;
- connected components become candidate event clusters;
- oversized or low-cohesion components are refined by raising the similarity threshold;
- small high-confidence components can be merged conservatively.

This graph-link method replaces the early DBSCAN prototype. It is easier to debug because edges, edge reasons, component density and similarity values can be stored and inspected.

Main files:

- `core/event_discovery/pipeline.py`
- `core/event_discovery/topic_expansion.py`
- `core/event_discovery/encoder.py`
- `core/event_discovery/clustering.py`
- `core/event_discovery/event_builder.py`
- `core/event_discovery/title_features.py`

### 3. Timeline Reasoning Layer

The timeline reasoning layer reads candidate event clusters and builds compact `EventCard` objects. Each card contains representative title, evidence titles, event time fields, cluster size, source count, confidence, risk flags and quality metrics.

Events are routed by deterministic rules:

- `auto_accept`: low-risk event accepted without LLM call.
- `llm_review`: uncertain event sent to local LLM.
- `rule_reject`: structurally invalid event rejected by rules.

The LLM acts as a lightweight judge rather than a full generator. It decides topic relevance, final noise status, display title, time anchor, split / merge hint and decision confidence. Final ordering is deterministic and handled by code.

Main files:

- `core/timeline_reasoning/models.py`
- `core/timeline_reasoning/event_cards.py`
- `core/timeline_reasoning/filters.py`
- `core/timeline_reasoning/llm_judge.py`
- `core/timeline_reasoning/ordering.py`
- `core/timeline_reasoning/persistence.py`
- `core/timeline_reasoning/pipeline.py`
- `core/timeline_reasoning/topic_profile.py`

### 4. Web Display Layer

The display layer is a FastAPI-served static application. It intentionally avoids a Node build step.

Frontend capabilities:

- topic input;
- `fast` / `standard` / `full` mode selector;
- fixed dataset date range selector;
- force regenerate toggle;
- recent timeline records;
- job progress and estimated remaining time;
- cancel current generation;
- horizontal interactive timeline;
- hover article preview;
- node detail drawer with article links and model decision information.

Main files:

- `services/timeline_api.py`
- `code/script/run_timeline_web_job.py`
- `frontend/static/index.html`
- `frontend/static/styles.css`
- `frontend/static/app.js`

## Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
cd /Users/hjfl/newsline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start or verify MySQL, then update local database settings if needed:

```text
configs/db_config.py
```

Start Ollama and make sure the local reasoning model is available:

```bash
ollama pull qwen3.5:9b
ollama serve
```

The embedding model is loaded through the Python model stack and should be available locally or downloadable through the normal Hugging Face cache flow.

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

Run timeline reasoning on the latest discovery run:

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

Development mode:

```bash
uvicorn services.timeline_api:app --host 127.0.0.1 --port 8000 --reload
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

If a completed result already exists for the same topic, mode and date range, the API can reuse the MySQL result instead of recomputing SBERT and LLM stages. The force regenerate switch skips this cache.

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

For the Web app, MySQL is the formal result source. The frontend reads:

```text
timeline_reasoning_runs
timeline_nodes
timeline_node_articles
```

Large generated files such as embedding indexes, reports and timeline JSONs are local artifacts and are ignored by Git unless explicitly needed.

## Tests

Run the current unit test suite:

```bash
.venv/bin/python -m pytest
```

Current coverage focuses on:

- active capability imports;
- event discovery behavior;
- title risk features and graph-link clustering;
- timeline reasoning routing and decision materialization;
- API-facing data model compatibility.

## Project Documents

Key handoff and design notes:

- `sbert_layer_v3_handoff.md`
- `llm_layer_v3_handoff.md`
- `frontend_display_layer_v2_handoff.md`
- `cross_language_retrieval_schemes.md`
- `TECHNICAL_REPORT.md`

## Final Result

The current result is a runnable end-to-end NewsLine prototype:

1. It reads parsed news records from MySQL.
2. It discovers topic-related candidate events through multilingual recall and embedding graph linkage.
3. It uses lightweight local LLM reasoning only on uncertain event cards.
4. It persists timeline results with article-level traceability.
5. It exposes a browser-based interface for topic-driven timeline generation and inspection.
