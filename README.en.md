# NewsLine

[中文](README.md)

NewsLine is a multilingual news-title event discovery and timeline reconstruction system. It reads parsed news titles, sources, URLs, and time fields from MySQL, then builds topic-centered candidate sets, event clusters, timeline reasoning results, persistent records, and a browser-based timeline view.

## Features

- Multilingual topic alias expansion for better cross-language recall.
- Title embedding and similarity-graph event discovery.
- Noise filtering with time constraints, title-risk flags, and cluster-quality metrics.
- Structured reasoning for uncertain events through a local Ollama model.
- MySQL persistence for event discovery outputs and final timeline nodes.
- FastAPI service and static frontend for job creation, progress polling, cached result reuse, and timeline browsing.

## Pipeline

```text
MySQL parsed news
-> topic alias expansion
-> candidate recall
-> title embedding
-> graph-based event discovery
-> rule routing and LLM reasoning
-> timeline persistence
-> FastAPI and static web UI
```

## Repository Layout

```text
newsline/
├── configs/                  # database, model, path, and pipeline configuration
├── core/
│   ├── event_discovery/      # topic expansion, embedding, graph clustering, event building
│   ├── llm/                  # Ollama client
│   └── timeline_reasoning/   # rule routing, LLM judging, ordering, persistence
├── data_pipeline/            # scraping, cleaning, normalization, and time parsing modules
├── database/                 # MySQL helpers and schema
├── code/script/              # command-line entry points and runners
├── services/                 # FastAPI service
├── frontend/static/          # static HTML/CSS/JavaScript frontend
├── newsdata/                 # sample or historical news data files
├── datasets/                 # open dataset exports
├── outputs/                  # generated runtime artifacts
├── tests/                    # core unit tests
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.14
- MySQL 8.x or compatible
- Ollama with local `qwen3.5:9b`
- Enough local memory or GPU memory for the embedding model

Main model configuration is in [configs/model_config.py](configs/model_config.py):

- `Qwen/Qwen3-Embedding-4B`
- `qwen3.5:9b`
- `facebook/nllb-200-distilled-600M`

## Installation

```bash
cd newsline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare the Ollama model:

```bash
ollama pull qwen3.5:9b
ollama serve
```

## Database Configuration

Update the connection settings in [configs/db_config.py](configs/db_config.py):

```python
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "database": "news_db",
    "charset": "utf8mb4",
}
```

The main pipeline reads news from `parser_newsdata`. Expected fields include:

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

Event discovery and timeline reasoning write to:

- `event_discovery_events`
- `event_discovery_assignments`
- `event_discovery_graph`
- `timeline_reasoning_runs`
- `timeline_event_decisions`
- `timeline_nodes`
- `timeline_node_articles`

See [database/schema.sql](database/schema.sql) for schema details.

## CLI Usage

Run event discovery:

```bash
.venv/bin/python code/script/run_event_discovery.py --topic "Apple"
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
  --topic "Apple" \
  --mode standard \
  --start-date 2025-06-01 \
  --end-date 2026-04-01
```

Available `mode` values:

- `fast`: fewer LLM reviews for quick previews.
- `standard`: balanced default mode.
- `full`: more complete LLM review with longer runtime.

## Web Service

```bash
cd newsline
source .venv/bin/activate
uvicorn services.timeline_api:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Main API routes:

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

[outputs](outputs) stores generated runtime artifacts. Historical JSON files and reports are not part of the core source code.

Common outputs:

- `outputs/clustered/`: event discovery JSON files.
- `outputs/timeline/`: timeline JSON files.
- `outputs/logs/`: runtime logs.
- `outputs/parsed/`: reserved preprocessing output directory.

The web frontend prefers formal timeline results from MySQL. File outputs are mainly for debugging and offline inspection.

## Open Dataset

The MySQL news metadata has been exported to [datasets/newsline-news-metadata](datasets/newsline-news-metadata).

The dataset contains fields from the `parser_newsdata` table, including titles, sources, URLs, raw time expressions, normalized time fields, parser modes, and experimental labels. It does not include article body text. The directory provides CSV, JSONL, compressed copies, a MySQL schema, metadata summary, SHA256 checksums, and data reuse notes.

## Tests

Run the full test suite:

```bash
.venv/bin/python -m pytest
```

Run tests without creating local cache files:

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider
```

## Maintenance Notes


- Historical files under `outputs/` are generated runtime artifacts; the main pipeline recreates them when needed.
- The frontend is static and does not require a Node build step.
- Database settings, model names, and pipeline defaults live in `configs/db_config.py`, `configs/model_config.py`, and `configs/pipeline_config.py`.
