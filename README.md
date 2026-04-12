# NewsLine

新闻时序重构项目当前进入“项目架构调整 + 基础骨架搭建”阶段。此次调整遵循根目录中的 [Codex_项目架构调整操作说明.md](/Users/hjfl/newsline/Codex_项目架构调整操作说明.md:1)；重点是把正式版工程结构搭起来，同时保留现有业务实现和历史实验结果，不直接改写算法逻辑。

## 当前阶段

- 当前可认为：架构调整已基本完成。
- 已完成：正式版目录骨架、集中配置、核心数据结构、服务入口、前端占位、输出目录、最小导入测试。
- 已完成：新架构到旧脚本的兼容适配层，正式目录中的部分入口现在可以懒加载调用原实现。
- 明确保留：`archive_mvp/` 作为历史实验区，不参与正式主流程。
- 明确保留：`code/` 下现有处理脚本与评估脚本继续作为历史/待迁移实现存在，不做强制搬迁。
- 尚未处理：spaCy / SBERT / LLM 的正式模块化迁移与主流程接线。
- 当前新入口属于“结构入口 / 兼容入口”，不是完整业务主流程入口。

## 项目结构

```text
newsline/
├── archive_mvp/              # 历史 MVP、实验脚本、对照数据
├── code/                     # 历史正式实现与评估脚本，当前保留为待迁移区
├── configs/                  # 新架构主配置层
├── database/                 # 数据库连接、CRUD、schema 草案
├── data_pipeline/            # 当前现役数据集构建与 spaCy 处理所在层
├── core/                     # 正式版核心算法层骨架与映射
├── services/                 # 结构级服务门面与兼容入口
├── frontend/                 # Streamlit 前端入口占位
├── outputs/                  # 新架构正式输出目录
├── tests/                    # 结构与导入验证
├── newsdata/                 # 历史输入数据 / 历史采集结果目录
├── requirements.txt
└── README.md
```

## 正式版目录职责

- `configs/`：新架构的主配置来源，统一收口数据库参数、模型名、路径与流程级开关。
- `database/`：提供数据库连接、公用 CRUD 和最小 schema 占位。
- `data_pipeline/datasets/`：当前现役的数据集构建层，已落位 RSS 数据集构建与 GDELT 增量构建能力。
- `data_pipeline/processors/`：当前现役的预处理层，已落位 DCT 标准化与 spaCy 时间处理入口。
- `data_pipeline/scrapers/`：采集兼容入口层，当前为服务层保留轻量路由。
- `core/event_discovery/`：承接事件发现层；当前仍是骨架与历史实现映射，不是正式 SBERT 实现。
- `core/timeline_reasoning/`：承接时序推理层；当前仍是骨架与历史实现映射，不是正式 LLM 实现。
- `core/timeline_builder.py`：用于统一最终时间线输出结构。
- `services/api_server.py`：为前端或未来 API 提供统一服务门面；当前以结构编排和兼容路由为主。
- `frontend/app.py`：正式版 Streamlit 入口占位。
- `outputs/`：新架构统一落盘 `parsed / clustered / timelines / logs` 的正式输出目录。
- `tests/`：提供结构与导入层面的最小验证。
- `archive_mvp/`：历史 MVP 与实验区，只保留、不参与正式主流程。
- `code/`：历史实现与评估区，当前仍承担“旧逻辑对照组”的角色。
- `newsdata/`：当前保留中的历史数据目录，不是新架构正式输出目录。

## 当前主流程入口

- 服务层入口：[services/api_server.py](/Users/hjfl/newsline/services/api_server.py:1)
- 前端入口：[frontend/app.py](/Users/hjfl/newsline/frontend/app.py:1)
- 核心数据结构：[core/schemas.py](/Users/hjfl/newsline/core/schemas.py:1)
- 兼容处理入口：[data_pipeline/processors/time_parser.py](/Users/hjfl/newsline/data_pipeline/processors/time_parser.py:1)
- 兼容采集入口：[data_pipeline/scrapers](/Users/hjfl/newsline/data_pipeline/scrapers)
- 核心层旧实现映射：
  [core/event_discovery/legacy_adapter.py](/Users/hjfl/newsline/core/event_discovery/legacy_adapter.py:1)、
  [core/timeline_reasoning/legacy_adapter.py](/Users/hjfl/newsline/core/timeline_reasoning/legacy_adapter.py:1)

说明：当前这些入口主要承担“稳定边界”“兼容路由”“后续迁移落点”的职责；复杂逻辑仍保留在旧实现中，尚未形成完整的新业务主流程。

## 当前现役能力

当前真正已经落位到新架构、并可继续演进的能力只有两类：

1. 数据集构建
   - RSS 数据集构建：
     [data_pipeline/datasets/rss_dataset.py](/Users/hjfl/newsline/data_pipeline/datasets/rss_dataset.py:1)
   - GDELT 增量数据集构建：
     [data_pipeline/datasets/gdelt_dataset.py](/Users/hjfl/newsline/data_pipeline/datasets/gdelt_dataset.py:1)

2. spaCy 处理
   - DCT 标准化：
     [data_pipeline/processors/time_standardizer.py](/Users/hjfl/newsline/data_pipeline/processors/time_standardizer.py:1)
   - 当前主 spaCy 处理入口：
     [data_pipeline/processors/spacy_pipeline.py](/Users/hjfl/newsline/data_pipeline/processors/spacy_pipeline.py:1)

这些模块是本轮“现役能力迁移”的目标。

明确不属于本轮现役迁移的内容：
- `archive_mvp/`：历史归档，不参与本轮迁移
- HeidelTime 路线：保留旧文件，不作为现役能力
- SBERT / 聚类 / event discovery / timeline reasoning：继续保持骨架或历史映射，不提前实现

## 配置体系说明

当前项目存在三类配置来源：

1. 新架构主配置源
   [configs/db_config.py](/Users/hjfl/newsline/configs/db_config.py:1)、
   [configs/model_config.py](/Users/hjfl/newsline/configs/model_config.py:1)、
   [configs/pipeline_config.py](/Users/hjfl/newsline/configs/pipeline_config.py:1)、
   [configs/path_config.py](/Users/hjfl/newsline/configs/path_config.py:1)

2. 历史兼容 / 外部依赖配置
   [config.props](/Users/hjfl/newsline/config.props:1)
   当前仅作为 HeidelTime 时代的外部兼容配置保留，不是新架构主配置源。

3. 历史脚本内嵌常量
   `code/` 与 `archive_mvp/` 下部分脚本仍保留内嵌数据库配置、模型名、端口和路径常量。这些属于尚未统一的历史残留项。

当前建议理解为：
- 新模块开发与新顶层目录应以 `configs/` 为准
- `config.props` 视为历史外部依赖配置，不作为新架构默认配置体系的一部分
- 历史脚本里的内嵌常量暂时保留，不在本阶段强行统一
- 当前现役数据集构建相关配置集中在 [configs/dataset_config.py](/Users/hjfl/newsline/configs/dataset_config.py:1)

## 输入输出目录说明

- 正式输出目录：`outputs/`
  当前新架构约定的正式运行结果应优先写入这里。目录说明见 [outputs/README.md](/Users/hjfl/newsline/outputs/README.md:1)

- 历史数据目录：`newsdata/`
  当前主要承载历史原始数据与历史采集结果。目录说明见 [newsdata/README.md](/Users/hjfl/newsline/newsdata/README.md:1)

- 历史测试数据目录：`archive_mvp/newsdata_for_test/`
  当前仅保留为历史实验、对照测试、样例数据目录。目录说明见 [README.md](/Users/hjfl/newsline/archive_mvp/newsdata_for_test/README.md:1)

当前边界定义：
- `outputs/`：正式输出目标
- `newsdata/`：历史输入数据 / 历史采集结果保留区
- `archive_mvp/newsdata_for_test/`：历史实验与对照测试数据保留区

当前仍未完全收口的问题：
- 个别历史脚本仍会把结果写回 `newsdata/` 或 `newsdata_for_test/`
- 这些历史输出路径目前保留，但不应视为新架构正式输出目标

## 已保留的旧实现

- `archive_mvp/`：完整保留。
- `code/data_pipeline/processors/trans_standard.py`：当前发布时间标准化脚本。
- `code/data_pipeline/processors/spacy_parser.py`：当前主要时间解析实现。
- `code/data_pipeline/processors/spacy_parser_v1.py`：旧版时间解析实现。
- `code/data_pipeline/processors/heideltime_parser.py`：HeidelTime 英语解析实现。
- `code/data_pipeline/lnaguage/language_count.py`：语言分布扫描脚本。
- `archive_mvp/time_handling_test/time_sberting.py`：历史 SBERT 聚类实验。
- `archive_mvp/time_handling_test/timeline_reconstruction.py`：历史时间线重构实验。
- `code/script/script_for_nyt.py`：保留中的可选采集脚本，当前不视为现役迁移目标。
- `code/script/script_forcsv.py`：保留中的旧 GDELT 回填脚本，当前不视为现役迁移目标。
- `code/script/*.py`：数据脚本、评估脚本、对比脚本。

## 待迁移项

- 收口历史脚本中的配置常量与绝对路径，但不改核心算法逻辑。
- 继续将现役 `code/data_pipeline/processors/*.py` 中剩余可稳定迁移的内容逐步迁入 `data_pipeline/processors/`。
- 按“现役优先”原则整理 `code/script/` 中仍需要继续使用的脚本。
- 将数据库建表、字段约束、索引设计从脚本约定迁移到 `database/schema.sql` 与 CRUD 层。
- 将事件发现与时间线推理逻辑分别接入 `core/event_discovery/` 和 `core/timeline_reasoning/`。
- 修正历史目录中的命名问题，例如 `code/data_pipeline/lnaguage/`。

## 新增/迁移说明

- 新增集中配置：
  - [configs/dataset_config.py](/Users/hjfl/newsline/configs/dataset_config.py:1)
  - [configs/db_config.py](/Users/hjfl/newsline/configs/db_config.py:1)
  - [configs/model_config.py](/Users/hjfl/newsline/configs/model_config.py:1)
  - [configs/path_config.py](/Users/hjfl/newsline/configs/path_config.py:1)
  - [configs/pipeline_config.py](/Users/hjfl/newsline/configs/pipeline_config.py:1)
- 新增现役数据集构建模块：
  - [data_pipeline/datasets/rss_dataset.py](/Users/hjfl/newsline/data_pipeline/datasets/rss_dataset.py:1)
  - [data_pipeline/datasets/gdelt_dataset.py](/Users/hjfl/newsline/data_pipeline/datasets/gdelt_dataset.py:1)
- 新增现役 spaCy 处理模块：
  - [data_pipeline/processors/time_standardizer.py](/Users/hjfl/newsline/data_pipeline/processors/time_standardizer.py:1)
  - [data_pipeline/processors/spacy_pipeline.py](/Users/hjfl/newsline/data_pipeline/processors/spacy_pipeline.py:1)
- 新增数据库骨架：
  - [database/db_config.py](/Users/hjfl/newsline/database/db_config.py:1)
  - [database/db_utils.py](/Users/hjfl/newsline/database/db_utils.py:1)
  - [database/crud.py](/Users/hjfl/newsline/database/crud.py:1)
  - [database/schema.sql](/Users/hjfl/newsline/database/schema.sql:1)
- 新增核心骨架：
  - [core/schemas.py](/Users/hjfl/newsline/core/schemas.py:1)
  - [core/event_discovery/pipeline.py](/Users/hjfl/newsline/core/event_discovery/pipeline.py:1)
  - [core/timeline_reasoning/pipeline.py](/Users/hjfl/newsline/core/timeline_reasoning/pipeline.py:1)
  - [core/event_discovery/legacy_adapter.py](/Users/hjfl/newsline/core/event_discovery/legacy_adapter.py:1)
  - [core/timeline_reasoning/legacy_adapter.py](/Users/hjfl/newsline/core/timeline_reasoning/legacy_adapter.py:1)
- 新增兼容适配层：
  - [data_pipeline/_legacy.py](/Users/hjfl/newsline/data_pipeline/_legacy.py:1)
  - [data_pipeline/processors/time_parser.py](/Users/hjfl/newsline/data_pipeline/processors/time_parser.py:1)
  - [data_pipeline/processors/language_stats.py](/Users/hjfl/newsline/data_pipeline/processors/language_stats.py:1)
  - [data_pipeline/scrapers/rss.py](/Users/hjfl/newsline/data_pipeline/scrapers/rss.py:1)
  - [data_pipeline/scrapers/gdelt.py](/Users/hjfl/newsline/data_pipeline/scrapers/gdelt.py:1)
  - [data_pipeline/scrapers/nyt.py](/Users/hjfl/newsline/data_pipeline/scrapers/nyt.py:1)
- 新增服务与前端入口：
  - [services/api_server.py](/Users/hjfl/newsline/services/api_server.py:1)
  - [frontend/app.py](/Users/hjfl/newsline/frontend/app.py:1)
- 新增测试：
  - [tests/test_imports.py](/Users/hjfl/newsline/tests/test_imports.py:1)

## 启动与验证

- 基础结构校验：

```bash
python -m unittest discover -s tests
```

- 最小导入验证：

```bash
python - <<'PY'
import importlib

for name in [
    "configs",
    "database.db_utils",
    "data_pipeline.processors.time_parser",
    "core.schemas",
    "services.api_server",
    "frontend.app",
]:
    importlib.import_module(name)

print("root imports ok")
PY
```

- 通过服务层调用旧处理任务的示例：

```python
from services.api_server import NewsTimelineService

service = NewsTimelineService()
service.run_legacy_time_standardization()
service.run_legacy_processing_job("spacy_v2")
print(service.list_legacy_core_modules())
```

- 现役能力最小验证：

```bash
/Users/hjfl/newsline/.venv/bin/python - <<'PY'
from data_pipeline.datasets.rss_dataset import get_rss_output_path
from data_pipeline.datasets.gdelt_dataset import normalize_gdelt_time
from data_pipeline.processors.spacy_pipeline import extract_event_time

print(get_rss_output_path())
print(normalize_gdelt_time("20260401123045"))
parsed, mode = extract_event_time(
    "The summit will take place on April 15, 2026",
    "2026-04-10 09:00:00",
)
print(mode, parsed.anchor.isoformat() if parsed else None)
PY
```

- 当前前端占位入口：

```bash
python frontend/app.py
```

- 未来迁移原则：
  - 先迁配置与入口，再迁单个业务模块。
  - 每次迁移尽量保持一个旧文件对应一个新落点。
  - 在正式模块接线前，不删除旧脚本。
