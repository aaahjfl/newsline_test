# NewsLine 项目技术报告

## 1. 项目概述

项目名称：NewsLine 新闻时序重构系统

毕业设计题目：基于 SBERT 与轻量大模型的新闻时序重构技术研究

报告日期：2026-05-05

NewsLine 面向多语种新闻标题集合，完成 topic 约束下的候选事件发现、事件有效性判断、时间线重构和可视化展示。系统输入为 MySQL 中的新闻标题、来源、链接和标准化时间字段，输出为可追溯的结构化时间线结果。

项目关注的问题包括：新闻标题短文本语义稀疏、多语种报道形式不一致、同一事件重复报道、事件时间表达不统一、topic 相关性判断困难以及时间线结果可解释性不足。系统通过 embedding 语义表示、图链接事件发现、轻量 LLM 裁判和数据库持久化机制，构建一条可复查的新闻时序重构流水线。

## 2. 开发环境

| 类别 | 配置 |
| --- | --- |
| 操作系统 | macOS 本地开发环境 |
| Python | Python 3.14.x，当前测试环境为 Python 3.14.3 |
| 虚拟环境 | `.venv` |
| 数据库 | MySQL 兼容数据库 |
| 默认数据库 | `news_db` |
| LLM 服务 | Ollama HTTP API |
| 本地 LLM 地址 | `http://127.0.0.1:11434` |
| Web 服务 | FastAPI + Uvicorn |
| 版本管理 | Git + GitHub |

依赖安装：

```bash
cd /Users/hjfl/newsline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Web 服务启动：

```bash
uvicorn services.timeline_api:app --host 127.0.0.1 --port 8000
```

测试命令：

```bash
.venv/bin/python -m pytest
```

## 3. 开发语言

| 语言 | 用途 |
| --- | --- |
| Python | 后端服务、数据处理、事件发现、LLM 调用、数据库读写、测试脚本 |
| JavaScript | 前端交互、任务轮询、时间线渲染、节点详情展示 |
| HTML | Web 页面结构 |
| CSS | 页面布局、响应式样式、时间线视觉呈现 |
| SQL | 数据表定义、查询和持久化 |

Python 是项目的核心开发语言。JavaScript、HTML 和 CSS 用于完成轻量级 Web 展示层。SQL 用于组织 MySQL 中的输入数据、中间结果和最终时间线结果。

## 4. 技术栈

### 4.1 后端与服务框架

- FastAPI：提供任务 API、结果 API、历史记录 API 和静态页面服务。
- Uvicorn：运行 ASGI Web 服务。
- Pydantic：校验 API 请求参数，包括 topic、mode、日期范围和重新生成开关。
- PyMySQL：连接 MySQL 并执行查询、插入和结果读取。
- subprocess / threading：在 Web 服务中启动独立后端任务，并持续读取任务进度。

### 4.2 数据处理与 NLP

- spaCy：新闻文本处理和事件时间解析入口。
- langdetect：语言检测辅助。
- sentence-transformers / Transformers：加载 embedding 与翻译模型。
- NumPy：向量矩阵、相似度矩阵和聚类指标计算。
- scikit-learn：机器学习与向量处理依赖。

### 4.3 模型配置

主要模型配置位于 `configs/model_config.py`：

```text
embedding_model: Qwen/Qwen3-Embedding-4B
topic_alias_model: qwen3.5:9b
topic_translation_model: facebook/nllb-200-distilled-600M
reasoning_model: qwen3.5:9b
time_parser_primary: spaCy
```

`Qwen/Qwen3-Embedding-4B` 用于标题向量化。`qwen3.5:9b` 通过 Ollama 本地调用，用于 topic alias 生成和事件裁判。`facebook/nllb-200-distilled-600M` 用于 topic 翻译辅助。

### 4.4 前端技术

前端采用静态页面实现：

- `frontend/static/index.html`：页面结构。
- `frontend/static/styles.css`：界面布局、时间线样式、响应式适配。
- `frontend/static/app.js`：API 调用、任务轮询、进度更新、时间线渲染、节点抽屉交互。

该实现方式依赖少，便于本地运行和毕业设计演示。

## 5. 系统总体架构

系统整体流程如下：

```text
parser_newsdata
-> topic alias expansion
-> candidate news retrieval
-> title filtering
-> title normalization and deduplication
-> Qwen embedding encoding
-> similarity graph construction
-> graph-link event clustering
-> event quality evaluation
-> event_discovery_* persistence
-> EventCard construction
-> rule routing and LLM judging
-> deterministic timeline ordering
-> timeline_* persistence
-> FastAPI result API
-> browser display
```

系统分为五个核心层次：

1. 数据处理层：完成新闻清洗、语言处理和时间字段标准化。
2. 事件发现层：基于标题语义向量和图链接方法生成候选事件簇。
3. 时间线推理层：结合规则和轻量 LLM 生成事件级决策。
4. 持久化层：保存输入、候选事件、图边、决策和最终时间线。
5. 展示层：提供 Web API 和交互式时间线页面。

## 6. 数据库设计

项目使用 MySQL 存储新闻数据和系统结果。主要表结构如下：

| 表名 | 作用 |
| --- | --- |
| `raw_news_data` | 原始新闻数据 |
| `parser_newsdata` | 解析后新闻数据，包含标题、来源、链接和事件时间字段 |
| `event_discovery_events` | 候选事件簇 |
| `event_discovery_assignments` | 新闻标题与事件簇的归属关系 |
| `event_discovery_graph` | 标题相似度图边 |
| `timeline_reasoning_runs` | 时间线推理运行记录 |
| `timeline_event_decisions` | 候选事件的规则或 LLM 决策 |
| `timeline_nodes` | 最终时间线节点 |
| `timeline_node_articles` | 时间线节点关联的新闻标题、来源和链接 |

其中，`event_discovery_graph` 保存相似度、时间间隔和边类型，便于分析事件簇形成过程。`timeline_event_decisions` 保存事件保留、降噪、topic 相关性、时间锚点和置信度等裁判结果。`timeline_node_articles` 保证最终结果可以回溯到新闻来源。

## 7. 核心模块设计

### 7.1 数据处理与时间解析

数据处理模块位于：

```text
data_pipeline/
```

其中 `data_pipeline/processors/spacy_pipeline.py` 是当前文本处理和事件时间解析入口。该层负责新闻标题处理、base time 标准化、事件时间抽取和解析后数据写入，为后续 topic 召回和时间线排序提供时间字段。

### 7.2 多语种 topic 召回

topic 召回模块将用户输入扩展为多语种 alias，并在 MySQL 中按标题字段检索候选新闻。相关配置位于：

```text
configs/pipeline_config.py
```

主要参数包括 alias 语言集合、每种语言 alias 数量、总 alias 限制和 Ollama 请求配置。该模块提高系统对多语种报道和不同标题表达形式的覆盖能力。

### 7.3 事件发现层

事件发现层位于：

```text
core/event_discovery/
```

主要文件：

- `pipeline.py`
- `encoder.py`
- `clustering.py`
- `event_builder.py`
- `title_features.py`
- `topic_expansion.py`

核心流程：

1. 根据 topic 和 alias 从 MySQL 召回候选新闻。
2. 对候选标题进行过滤和归一化。
3. 使用 embedding 模型生成标题向量。
4. 计算标题之间的语义相似度。
5. 结合相似度阈值和事件时间窗口建立图边。
6. 将连通分量转化为候选事件簇。
7. 对低内聚组件进行细化。
8. 构建标准化事件节点和新闻归属关系。

图链接聚类保留了事件簇内部的相似度证据，并将边类型记录为 `semantic_only`、`semantic_and_time` 或 `semantic_override`。这使系统能够分析某一事件簇的形成依据、时间跨度和语义内聚度。

### 7.4 事件质量评估

事件节点构建阶段生成质量指标：

- `semantic_cohesion`：事件簇语义内聚度。
- `temporal_coherence`：事件时间一致性。
- `support_score`：标题数量和来源支撑度。
- `graph_density`：簇内图边密度。
- `duplicate_ratio`：重复标题比例。
- `unique_title_count`：唯一标题数量。
- `article_count`：簇内文章数量。
- `time_span_days`：事件簇时间跨度。

系统同时生成风险标记：

- `long_time_span`
- `high_duplicate_ratio`
- `low_graph_density`
- `low_temporal_coherence`
- `rolling_coverage`
- `translated_topic_alias_risk`
- `ambiguous_topic_low_support`

这些指标参与后续规则路由和 LLM 审查，也为论文分析和系统解释提供依据。

### 7.5 时间线推理层

时间线推理层位于：

```text
core/timeline_reasoning/
```

主要文件：

- `models.py`
- `event_cards.py`
- `filters.py`
- `llm_judge.py`
- `ordering.py`
- `persistence.py`
- `pipeline.py`
- `topic_profile.py`

该层将候选事件转化为 `EventCard`。每张卡片包含事件标题、代表性证据、时间字段、风险标记、质量摘要和 topic profile。规则模块根据结构完整性、置信度、时间跨度、图密度和 topic 风险决定事件处理路径。

推理模式：

| 模式 | 说明 |
| --- | --- |
| `fast` | 优先使用规则，仅审查高风险事件 |
| `standard` | 平衡规则与 LLM 审查，适合常规展示 |
| `full` | 对候选事件执行更充分的 LLM 审查 |

LLM 输出结构化 JSON 决策，包括：

- 是否保留事件；
- 是否与 topic 相关；
- 是否为最终噪声；
- 展示标题；
- 解析后时间锚点；
- split / merge 提示；
- 决策置信度；
- 裁判理由。

最终时间线排序由 `ordering.py` 根据解析时间、模型时间锚点和稳定排序规则完成。

### 7.6 Web API 与任务机制

Web API 位于：

```text
services/timeline_api.py
```

主要接口：

```text
GET  /api/health
POST /api/timeline/jobs
GET  /api/timeline/jobs/{job_id}/status
POST /api/timeline/jobs/{job_id}/cancel
GET  /api/timeline/jobs/{job_id}/result
GET  /api/timeline/results/{reasoning_run_id}
GET  /api/timeline/recent?limit=6
```

Web 任务运行脚本为：

```text
code/script/run_timeline_web_job.py
```

该脚本依次调用事件发现和时间线推理流水线，并通过标准输出发送 `NEWSLINE_JOB_EVENT` 进度事件。FastAPI 服务读取进度事件并维护 job 状态，前端通过轮询接口更新页面。

### 7.7 前端展示层

前端页面包含三个状态：

| 状态 | 功能 |
| --- | --- |
| idle | topic 输入、mode 选择、日期范围、重新生成、最近记录 |
| running | 阶段提示、进度条、已用时间、预计剩余、取消任务 |
| result | 横向时间线、月份导航、文章预览、节点详情抽屉 |

用户可以在浏览器中输入 topic，选择推理模式和日期范围，启动完整流水线或复用历史结果。结果页展示时间线节点、相关新闻标题、来源链接、风险标记和模型决策信息。

## 8. 运行流程

### 8.1 事件发现

```bash
.venv/bin/python code/script/run_event_discovery.py --topic "Trump"
```

带日期范围：

```bash
.venv/bin/python code/script/run_event_discovery.py \
  --topic "Apple" \
  --start-date 2025-06-01 \
  --end-date 2026-04-01
```

### 8.2 时间线推理

```bash
.venv/bin/python code/script/run_timeline_reasoning.py \
  --topic "Apple" \
  --mode standard \
  --llm-batch-size 4 \
  --llm-timeout-seconds 300
```

### 8.3 Web 系统

```bash
source .venv/bin/activate
uvicorn services.timeline_api:app --host 127.0.0.1 --port 8000
```

访问地址：

```text
http://127.0.0.1:8000
```

## 9. 测试与验证

项目使用 `pytest` 进行测试：

```bash
.venv/bin/python -m pytest
```

当前测试覆盖：

- 活动模块导入；
- 事件发现流程；
- 图链接聚类逻辑；
- 标题风险特征；
- 时间线推理路由；
- LLM 决策结构；
- timeline record 构建。

最近一次测试结果：

```text
37 passed
```

## 10. 最终成果

项目当前形成了完整的新闻时序重构原型系统，主要成果如下：

1. 新闻处理与时间解析模块：支持新闻标题处理、事件时间字段提取和 MySQL 数据写入。
2. 多语种 topic 召回模块：支持 topic alias 扩展和跨语言新闻候选召回。
3. embedding 事件发现模块：使用标题向量和图链接方法生成候选事件簇。
4. 事件质量评估模块：输出语义内聚度、时间一致性、图密度、重复率和风险标记。
5. 轻量 LLM 时间线推理模块：支持事件保留、降噪、topic 相关性、时间锚点和展示标题裁判。
6. MySQL 持久化模块：保存候选事件、新闻归属、图边、LLM 决策、时间线节点和节点文章。
7. FastAPI 服务模块：提供任务创建、状态轮询、取消任务、结果读取和最近记录接口。
8. Web 展示系统：支持 topic 输入、日期筛选、生成进度、历史复用、横向时间线和节点详情查看。
9. 测试与文档：包含单元测试、README、技术报告和分层 handoff 文档。

## 11. 技术特点

### 11.1 可追溯的分层流水线

系统将新闻召回、事件发现、事件裁判和展示持久化分层实现。每个阶段都有明确的数据结构和输出结果，便于调试、复现实验和撰写论文。

### 11.2 面向短文本的语义事件发现

系统使用标题 embedding 表达新闻语义，并通过图链接方式组织候选事件。图边记录语义相似度、时间间隔和边类型，为事件簇质量分析提供证据。

### 11.3 轻量 LLM 结构化裁判

LLM 处理 compact event card，并输出结构化决策。规则路由控制模型调用范围，降低推理成本，同时保留对复杂事件的语义判断能力。

### 11.4 数据库级结果存证

候选事件、图边、模型决策、最终节点和关联文章均写入 MySQL。最终展示结果可以追溯到原始新闻标题和模型裁判记录。

### 11.5 可交互 Web 原型

FastAPI 与静态前端构成完整演示系统。用户可以输入 topic 运行任务，也可以复用历史时间线结果并查看节点详情。

## 12. 当前限制与后续工作

当前系统已具备完整原型能力，但仍有进一步完善空间：

- 构建人工标注 benchmark，用于评估事件聚类质量、topic relevance 和时间线排序质量。
- 为 embedding 编码、图构建和 LLM 批处理增加更细粒度的进度回调。
- 将本地数据库和模型配置迁移到环境变量或部署配置文件。
- 增加时间线结果导出功能，支持 Markdown、PDF 或 DOCX 报告。
- 优化大规模时间线的前端渲染性能和导航体验。

## 13. 结论

NewsLine 已实现从新闻标题输入到交互式时间线展示的完整技术闭环。系统以 spaCy 解析和 MySQL 数据为基础，使用 `Qwen/Qwen3-Embedding-4B` 完成标题语义表示，使用图链接方法完成候选事件发现，使用本地 `qwen3.5:9b` 完成不确定事件裁判，并通过 FastAPI 与静态前端提供可操作的演示界面。

该系统体现了面向新闻短文本时序重构任务的工程可行性：语义表示用于提升候选事件召回和聚合质量，轻量大模型用于补充复杂语义判断，数据库持久化用于保证结果可追溯和可复核。
