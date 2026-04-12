# outputs/

`outputs/` 是新架构约定的正式输出目录。

当前阶段说明：
- 新顶层模块在需要统一落盘时，应优先写入 `outputs/`
- 推荐将正式运行产物按 `parsed/`、`clustered/`、`timelines/`、`logs/` 分层保存
- 历史脚本尚未全部切换到这里，因此当前仓库中仍存在其他历史输出位置

当前不应混淆的目录：
- `newsdata/`：保留中的历史/原始数据目录，不视为正式输出目录
- `archive_mvp/newsdata_for_test/`：历史实验与对照测试数据目录，不视为正式输出目录
