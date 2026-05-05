# NewsLine 项目技术报告

## 1. 项目基本信息

项目名称：NewsLine 新闻时序重构系统

毕业设计题目：基于 SBERT 与轻量大模型的新闻时序重构技术研究

报告日期：2026-05-05

项目定位：面向多语种新闻标题集合的事件发现、时间线重构与可视化原型系统。

NewsLine 的目标是从数据库中已有的新闻标题、来源、链接和标准化时间字段出发，围绕用户输入的 topic 自动发现候选事件，判断事件是否有效和相关，并生成可追溯、可复查、可展示的结构化新闻时间线。

## 2. 项目背景与问题定义

互联网新闻数据具有明显的碎片化特征。同一事件可能被多个媒体以不同语言、不同标题、不同时间表达重复报道；同一 topic 下也可能混入滚动报道、背景介绍、误召回标题和弱相关新闻。若直接依赖关键词匹配，系统容易漏掉语义相近但字面不同的标题；若直接依赖大模型生成完整时间线，又会面临成本高、过程黑盒、难以溯源和事实幻觉等问题。

本项目采用“候选事件发现 + 轻量事件裁判 + 数据库存证 + Web 展示”的分层方案。SBERT / embedding 层负责高召回地组织候选事件，轻量 LLM 层只对不确定事件做判断，最终时间线由规则、模型判断和数据库记录共同构成。

## 3. 与开题方案相比的技术调整

开题报告中的总体研究方向仍然保留：使用语义向量模型解决新闻标题短文本语义稀疏问题，使用轻量大模型辅助事件时序判断。但具体工程技术栈已经根据实际实验效果进行了调整。

| 开题阶段设想 | 当前实现方案 | 调整原因 |
| --- | --- | --- |
| HeidelTime / spaCy 时间抽取并行探索 | spaCy 作为当前主线，HeidelTime 代码保留为历史兼容 | spaCy 更易集成到当前 Python 流水线，也更适合多语种处理入口统一 |
| DBSCAN 聚类 | embedding 相似度图链接聚类 | 图结构可以保留边、边原因和连通分量，更便于调试，也能保留 singleton 事件 |
| Qwen3-8B 轻量模型 | 本地 Ollama `qwen3.5:9b` | 当前本地推理环境以 Ollama 为主，便于统一 HTTP 调用、控制上下文和超时 |
| Streamlit 展示层 | FastAPI + 静态 HTML/CSS/JavaScript | FastAPI 更适合提供任务 API、缓存复用、异步 job 状态和前后端解耦 |
| 直接时间线图展示 | 横向交互时间线 + 节点详情抽屉 | 新闻事件数量可能很大，横向时间线更适合展示时间顺序和文章溯源 |

## 4. 开发环境

当前项目在本地开发环境中运行，核心环境如下：

| 类别 | 环境 / 工具 |
| --- | --- |
| 操作系统 | macOS 本地开发环境 |
| Python | Python 3.14.x，本项目测试环境为 Python 3.14.3 |
| 虚拟环境 | `.venv` |
| 数据库 | MySQL 兼容数据库，默认数据库名为 `news_db` |
| LLM 服务 | Ollama，本地 HTTP 服务地址 `http://127.0.0.1:11434` |
| 前端运行方式 | FastAPI 直接托管静态页面，无 Node 构建步骤 |
| 版本管理 | Git + GitHub |

依赖安装方式：

```bash
cd /Users/hjfl/newsline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

测试命令：

```bash
.venv/bin/python -m pytest
```

Web 服务启动命令：

```bash
uvicorn services.timeline_api:app --host 127.0.0.1 --port 8000
```

## 5. 开发语言

项目主要使用三类语言：

| 语言 | 用途 |
| --- | --- |
| Python | 数据处理、时间解析入口、事件发现、LLM 调用、MySQL 读写、FastAPI 服务 |
| JavaScript | 前端交互逻辑、任务轮询、时间线渲染、节点详情交互 |
| HTML / CSS | Web 页面结构与视觉样式 |
| SQL | MySQL 表结构、查询、持久化结果读取 |

## 6. 技术栈

### 6.1 后端与服务层

- FastAPI：提供 Web API 和静态文件服务。
- Pydantic：校验 topic、mode、日期范围、force regenerate 等请求字段。
- PyMySQL：连接 MySQL 并执行结果查询和持久化。
- subprocess + threading：Web 任务启动独立 Python 子进程，主服务通过标准输出读取机器可解析进度事件。

### 6.2 数据处理与 NLP

- spaCy：当前主线文本处理与事件时间解析入口。
- langdetect / 多语种模型：辅助语言识别和多语种处理。
- NLLB：用于 topic 翻译和跨语言召回辅助。
- sentence-transformers / Transformers：加载标题 embedding 模型。
- NumPy / scikit-learn：向量矩阵、相似度计算和实验分析。

### 6.3 模型

当前主要模型配置位于 `configs/model_config.py`：

```text
embedding_model: Qwen/Qwen3-Embedding-4B
topic_alias_model: qwen3.5:9b
topic_translation_model: facebook/nllb-200-distilled-600M
reasoning_model: qwen3.5:9b
time_parser_primary: spaCy
```

`qwen3.5:9b` 通过 Ollama 本地服务调用，主要承担 topic alias 扩展和 timeline reasoning 中的不确定事件裁判任务。

### 6.4 前端

前端不使用 React / Vue / Streamlit，而是采用原生 HTML、CSS 和 JavaScript：

- `frontend/static/index.html`：页面结构。
- `frontend/static/styles.css`：响应式布局、时间线、抽屉、模式选择、进度面板样式。
- `frontend/static/app.js`：任务创建、轮询、缓存结果读取、时间线渲染、节点交互、最近记录加载。

这样做的优点是部署简单、依赖少，适合毕业设计原型系统演示。

## 7. 系统总体架构

系统采用分层架构：

```text
MySQL parser_newsdata
-> topic alias expansion
-> candidate news retrieval
-> title filtering and deduplication
-> Qwen embedding encoding
-> graph-link clustering
-> event node construction
-> event_discovery_* persistence
-> EventCard construction
-> rule routing / LLM review
-> deterministic ordering
-> timeline_* persistence
-> FastAPI result API
-> browser timeline display
```

各层之间通过标准化数据结构衔接。事件发现层输出 `EventDiscoveryResult`、`EventNode`、assignments 和 graph edges；时间线推理层读取这些候选事件，输出 `TimelineReasoningResult`、`TimelineRecord` 和 `EventDecision`；展示层最终从 MySQL 的 timeline 表读取正式结果。

## 8. 数据库设计

项目使用 MySQL 存储输入数据和中间 / 最终结果。主要表包括：

| 表名 | 作用 |
| --- | --- |
| `raw_news_data` | 原始新闻数据占位表 |
| `parser_newsdata` | 经过解析和时间标准化后的新闻数据 |
| `event_discovery_events` | SBERT / embedding 层生成的候选事件簇 |
| `event_discovery_assignments` | 新闻标题到候选事件簇的归属关系 |
| `event_discovery_graph` | 标题相似度图的边和边原因 |
| `timeline_reasoning_runs` | LLM 时间线推理运行记录 |
| `timeline_event_decisions` | 每个候选事件的规则或 LLM 决策 |
| `timeline_nodes` | 最终展示用时间线节点 |
| `timeline_node_articles` | 时间线节点关联的原始新闻标题、来源和链接 |

这种设计保证最终展示结果可以回溯到原始新闻标题，并能解释每个节点来自哪个事件簇、经过了哪些风险判断和模型裁判。

## 9. 核心模块设计

### 9.1 spaCy 时间解析入口

当前正式入口为：

```text
data_pipeline/processors/spacy_pipeline.py
```

该模块作为新架构入口，复用已有 MVP 阶段的 spaCy 解析实现，提供：

- active spaCy parser 路径获取；
- 模型映射查询；
- base time 标准化；
- 标题事件时间抽取；
- 数据库处理流水线入口。

项目仍保留 HeidelTime 相关历史文件，但当前报告和 README 以 spaCy 为主线。

### 9.2 事件发现层

事件发现层核心文件：

```text
core/event_discovery/pipeline.py
core/event_discovery/clustering.py
core/event_discovery/event_builder.py
core/event_discovery/title_features.py
```

主要步骤：

1. 校验 topic。
2. 调用 topic expansion 生成多语种 alias。
3. 从 `parser_newsdata` 中按标题和日期范围召回候选新闻。
4. 过滤弱相关标题。
5. 归一化标题并折叠完全重复标题。
6. 使用 `Qwen/Qwen3-Embedding-4B` 编码标题。
7. 计算标题相似度矩阵。
8. 根据相似度阈值和时间窗口建立图边。
9. 对图连通分量做聚类、细化和保守合并。
10. 输出标准化事件节点、新闻归属和图边。

图链接聚类的优点：

- 可以保留单条新闻构成的 singleton event。
- 可以记录每条边是 `semantic_only`、`semantic_and_time` 还是 `semantic_override`。
- 可以通过图密度、平均相似度和时间一致性判断簇质量。
- 对大组件可以递归提高阈值进行细化。

### 9.3 事件质量与风险标记

事件节点构建时会生成：

- `semantic_cohesion`
- `temporal_coherence`
- `support_score`
- `graph_density`
- `duplicate_ratio`
- `unique_title_count`
- `article_count`
- `time_span_days`

同时生成风险标记，例如：

- `long_time_span`
- `high_duplicate_ratio`
- `low_graph_density`
- `low_temporal_coherence`
- `rolling_coverage`
- `translated_topic_alias_risk`
- `ambiguous_topic_low_support`

这些指标不仅用于后续 LLM 路由，也能在答辩或论文中解释系统为什么认为某个事件可靠或可疑。

### 9.4 时间线推理层

时间线推理层核心文件：

```text
core/timeline_reasoning/models.py
core/timeline_reasoning/event_cards.py
core/timeline_reasoning/filters.py
core/timeline_reasoning/llm_judge.py
core/timeline_reasoning/ordering.py
core/timeline_reasoning/persistence.py
core/timeline_reasoning/pipeline.py
```

系统不会把所有原始新闻全文交给 LLM，而是构造紧凑的 `EventCard`。每张卡片包含事件标题、时间字段、风险标记、质量摘要和少量代表性证据标题。

推理层支持三种模式：

| 模式 | 特点 |
| --- | --- |
| `fast` | 更依赖规则，仅将明显风险事件交给 LLM |
| `standard` | 规则与 LLM 均衡，额外抽取部分不确定事件复核 |
| `full` | 尽可能让 LLM 审查全部事件 |

最终时间线的顺序由程序根据解析时间和决策时间锚点确定，而不是由大模型自由生成，降低幻觉和顺序漂移风险。

### 9.5 Web API 与任务机制

FastAPI 服务位于：

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

Web job runner 位于：

```text
code/script/run_timeline_web_job.py
```

它负责串联：

```text
run_event_discovery()
-> run_timeline_reasoning_pipeline()
```

并通过标准输出发送形如 `NEWSLINE_JOB_EVENT {...}` 的进度事件。FastAPI 主进程读取这些事件，更新内存中的 job 状态，前端则轮询 status API。

### 9.6 前端展示层

前端展示层包括三种状态：

| 状态 | 功能 |
| --- | --- |
| idle | topic 输入、mode 选择、日期范围、重新生成、最近记录 |
| running | 阶段提示、进度条、已用时间、预计剩余、取消按钮 |
| result | 横向时间线、月份导航、hover 预览、节点详情抽屉 |

前端结果不直接读取 JSON 文件，而是通过 API 从 MySQL 中读取正式时间线结果。这保证 Web 展示、实验输出和数据库存证保持一致。

## 10. 运行流程

### 10.1 命令行运行事件发现

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

### 10.2 命令行运行时间线推理

```bash
.venv/bin/python code/script/run_timeline_reasoning.py \
  --topic "Apple" \
  --mode standard \
  --llm-batch-size 4 \
  --llm-timeout-seconds 300
```

### 10.3 启动 Web 演示系统

```bash
source .venv/bin/activate
uvicorn services.timeline_api:app --host 127.0.0.1 --port 8000
```

浏览器访问：

```text
http://127.0.0.1:8000
```

## 11. 测试与验证

当前项目使用 `pytest` 运行单元测试：

```bash
.venv/bin/python -m pytest
```

最近一次测试结果：

```text
37 passed
```

测试覆盖重点：

- 模块导入与活动能力检查；
- 事件发现的聚类和事件节点构建逻辑；
- 标题风险特征；
- LLM 路由规则；
- timeline reasoning 数据模型；
- 持久化和结果结构兼容性。

## 12. 最终成果

项目当前已经形成一套可运行、可演示、可继续扩展的毕业设计原型系统，最终成果包括：

1. 一套完整的后端流水线代码：包含 spaCy 时间解析入口、topic alias 扩展、embedding 编码、图链接事件发现、事件质量评估和 MySQL 持久化。
2. 一套轻量 LLM 时间线推理模块：支持 fast / standard / full 三种模式，能对不确定事件进行语义裁判，并输出可解释决策。
3. 一套数据库结果体系：保存候选事件、新闻归属、相似度图、LLM 决策、最终时间线节点和节点关联文章。
4. 一套 Web 原型系统：支持用户输入 topic 自动生成或复用时间线，并在浏览器中查看横向时间线和节点详情。
5. 一组测试用例和诊断脚本：用于保证关键逻辑可运行，并为后续实验评估提供基础。
6. 多份技术交接文档：包括 SBERT 层、LLM 层、前端展示层和跨语言召回方案说明。

## 13. 项目创新点

### 13.1 分层式时间线重构

系统没有把新闻标题直接交给大模型生成时间线，而是拆成“事件发现”和“事件裁判”两层。这样既降低模型调用成本，也保留了可解释的中间结果。

### 13.2 embedding 图链接替代传统聚类

早期 DBSCAN 对短文本新闻标题的参数敏感，且不易解释聚类边界。当前图链接方案保留了边、时间约束、相似度和组件质量指标，更适合毕业设计中的方法分析和可视化诊断。

### 13.3 轻量 LLM 作为裁判而非生成器

LLM 只审查不确定事件，输出结构化 JSON 决策。最终排序、持久化和展示由代码完成，减少幻觉风险。

### 13.4 面向溯源的数据库设计

每个时间线节点都能追溯到候选事件簇和原始新闻标题，适合舆情分析、新闻复盘、开源情报和论文实验分析。

### 13.5 原型系统可直接演示

FastAPI + 静态前端让系统从算法实验升级为可交互原型。用户可以输入 topic、选择模式、筛选日期、复用历史结果，并检查每个节点的新闻来源。

## 14. 当前限制与后续工作

当前系统仍有一些限制：

- 时间线质量评估仍以规则诊断和局部测试为主，后续需要构建人工标注 benchmark。
- Web 生成进度是阶段级进度，SBERT 和 LLM 内部还没有细粒度 callback。
- 当前 MySQL 配置偏本地开发环境，需要在部署时改成环境变量或配置文件注入。
- `outputs/` 下的部分 JSON 属于实验产物，不适合直接作为长期版本化数据。
- 时间线节点数量很大时目前仍是前端一次性渲染，未来可考虑虚拟列表或缩略导航。

后续优化方向：

- 构建人工标注数据集，计算 topic relevance、event clustering purity、Kendall's tau 等指标。
- 增强 LLM 决策 prompt 与错误恢复逻辑。
- 为 SBERT 编码、图构建、LLM 批处理增加真实进度回调。
- 将本地数据库和模型配置改为更安全的环境变量读取。
- 增加导出报告功能，把某次时间线结果导出为 Markdown / PDF / DOCX。

## 15. 结论

NewsLine 当前已经从开题阶段的技术构想演进为一个端到端可运行的新闻时序重构系统。它以 spaCy 解析和 MySQL 数据为基础，以 `Qwen/Qwen3-Embedding-4B` 完成短文本语义表示，以图链接聚类完成候选事件发现，以本地 `qwen3.5:9b` 完成不确定事件裁判，并通过 FastAPI 与原生前端提供交互式展示。

该系统的核心价值在于：在避免大模型直接生成带来黑盒和幻觉风险的同时，利用 embedding 和轻量 LLM 分别解决短文本语义聚合和复杂事件判断问题，形成一条可解释、可溯源、可演示的新闻时间线重构技术路线。
