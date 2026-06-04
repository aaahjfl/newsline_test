# NewsLine

[English](README.en.md)

NewsLine 是一个面向多语种新闻标题的事件发现与时间线重构系统。系统从 MySQL 中读取已解析的新闻标题、来源、链接和时间字段，围绕用户输入的 topic 完成候选新闻召回、事件聚类、时间线推理、结果持久化和 Web 可视化。

## 核心能力

- 多语种 topic alias 扩展，提升不同语言新闻标题的召回率。
- 基于标题 embedding 的相似度图构建与事件簇发现。
- 结合时间约束、标题风险标记和聚类质量指标过滤噪声事件。
- 使用本地 Ollama 模型对不确定事件进行结构化推理。
- 将事件发现结果和最终时间线写入 MySQL，同时导出 JSON 运行产物。
- 提供 FastAPI 后端和静态前端，用于创建任务、查看进度、复用历史结果和浏览时间线。

## 工作流

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

## 目录结构

```text
newsline/
├── configs/                  # 数据库、模型、路径和流水线配置
├── core/
│   ├── event_discovery/      # topic 扩展、embedding、图聚类和事件构建
│   ├── llm/                  # Ollama 客户端
│   └── timeline_reasoning/   # 规则路由、LLM 裁判、排序和持久化
├── data_pipeline/            # 数据抓取、清洗、标准化和时间解析模块
├── database/                 # MySQL 连接工具和 schema
├── code/script/              # 命令行入口和运行脚本
├── services/                 # FastAPI 服务
├── frontend/static/          # 原生 HTML/CSS/JavaScript 前端
├── newsdata/                 # 示例/历史新闻数据文件
├── outputs/                  # 运行生成物目录，内容可重新生成
├── tests/                    # 核心单元测试
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.14
- MySQL 8.x 或兼容版本
- Ollama，本地提供 `qwen3.5:9b`
- 足够的本地内存/显存用于加载 embedding 模型

主要模型配置位于 [configs/model_config.py](configs/model_config.py)：

- `Qwen/Qwen3-Embedding-4B`
- `qwen3.5:9b`
- `facebook/nllb-200-distilled-600M`

## 安装

```bash
cd newsline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

准备 Ollama 模型：

```bash
ollama pull qwen3.5:9b
ollama serve
```

## 数据库配置

修改 [configs/db_config.py](configs/db_config.py) 中的连接信息：

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

主流程默认从 `parser_newsdata` 表读取新闻。核心字段包括：

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

事件发现和时间线结果会写入以下表：

- `event_discovery_events`
- `event_discovery_assignments`
- `event_discovery_graph`
- `timeline_reasoning_runs`
- `timeline_event_decisions`
- `timeline_nodes`
- `timeline_node_articles`

表结构参考 [database/schema.sql](database/schema.sql)。

## 命令行使用

运行事件发现：

```bash
.venv/bin/python code/script/run_event_discovery.py --topic "Apple"
```

按日期范围运行事件发现：

```bash
.venv/bin/python code/script/run_event_discovery.py \
  --topic "Apple" \
  --start-date 2025-06-01 \
  --end-date 2026-04-01
```

运行时间线推理：

```bash
.venv/bin/python code/script/run_timeline_reasoning.py \
  --topic "Apple" \
  --mode standard \
  --llm-batch-size 4 \
  --llm-timeout-seconds 300
```

运行 Web 任务脚本：

```bash
.venv/bin/python code/script/run_timeline_web_job.py \
  --topic "Apple" \
  --mode standard \
  --start-date 2025-06-01 \
  --end-date 2026-04-01
```

`mode` 可选值：

- `fast`：更少 LLM 复核，适合快速预览。
- `standard`：默认平衡模式。
- `full`：更充分的 LLM 复核，耗时更长。

## 启动 Web 服务

```bash
cd newsline
source .venv/bin/activate
uvicorn services.timeline_api:app --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000
```

主要 API：

```text
GET  /api/health
POST /api/timeline/jobs
GET  /api/timeline/jobs/{job_id}/status
POST /api/timeline/jobs/{job_id}/cancel
GET  /api/timeline/jobs/{job_id}/result
GET  /api/timeline/results/{reasoning_run_id}
GET  /api/timeline/recent?limit=6
```

## 输出目录

[outputs](outputs) 是运行生成物目录。源码本身不依赖其中的历史 JSON 或报告文件。

常见输出：

- `outputs/clustered/`：事件发现 JSON。
- `outputs/timeline/`：时间线 JSON。
- `outputs/logs/`：运行日志。
- `outputs/parsed/`：预处理输出预留目录。

Web 前端优先从 MySQL 读取正式时间线结果；文件输出主要用于调试和离线检查。

## 测试

运行全部测试：

```bash
.venv/bin/python -m pytest
```

如果希望测试后不生成本地缓存：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider
```

## 维护说明

- `outputs/` 中的历史结果属于运行生成物，主流程会按需重新生成。
- 前端为静态文件，不需要单独的 Node 构建步骤。
- 数据库连接、模型名称和流水线默认值分别在 `configs/db_config.py`、`configs/model_config.py` 和 `configs/pipeline_config.py` 中维护。
