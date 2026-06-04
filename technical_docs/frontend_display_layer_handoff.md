# NewsLine 前端展示层 Handoff

本文档用于交接当前 NewsLine 前端展示层，供下一轮界面优化或功能扩展使用。

当前日期：2026-04-29

## 1. 当前定位

当前展示层已从 `frontend/app.py` 的 Streamlit 占位迁移为：

```text
FastAPI + 静态 HTML/CSS/JS 前端
```

展示层目标是：

- 用户只输入一次 topic 和 LLM mode
- 后端正式执行 SBERT 事件发现层
- 后端继续执行 LLM 时间线决断层
- 结果写入 MySQL
- 前端从 MySQL 查询并展示时间线

测试用 JSON 仍可保留用于调试，但正式前端展示不读取 `outputs/timeline/*.json`。

## 2. 启动方式

项目根目录：

```bash
cd /Users/hjfl/newsline
source .venv/bin/activate
uvicorn services.timeline_api:app --host 127.0.0.1 --port 8000
```

浏览器访问：

```text
http://127.0.0.1:8000
```

如果需要开发时自动重载：

```bash
uvicorn services.timeline_api:app --host 127.0.0.1 --port 8000 --reload
```

## 3. 主要文件

后端 API：

```text
services/timeline_api.py
```

Web job runner：

```text
code/script/run_timeline_web_job.py
```

前端静态页面：

```text
frontend/static/index.html
frontend/static/styles.css
frontend/static/app.js
```

旧 Streamlit 入口：

```text
frontend/app.py
```

目前 `frontend/app.py` 只保留为 legacy 说明入口，正式展示层由 FastAPI 托管。

## 4. 页面状态

前端当前有三个状态：

```text
idle     首页输入态
running 生成中态
result  时间线结果态
```

状态通过 `.app-shell[data-view="..."]` 控制。

## 5. 首页设计

视觉方向：

```text
Apple 式冷静、高留白、低饱和科技感
```

不走赛博、霓虹、复杂 HUD 风格。

首页布局：

```text
左侧：
- NewsLine
- 新闻事件 / 时间线重构系统
- 项目简介
- topic 输入框
- mode 下拉框：fast / standard / full
- 生成时间线按钮

右侧：
- 低饱和 canvas 地球
- 克制的信息流弧线
```

地球不是 Three.js，而是在 `frontend/static/app.js` 中用 Canvas 绘制，避免额外前端构建依赖。

## 6. 生成流程

用户提交：

```json
{
  "topic": "Apple",
  "mode": "fast"
}
```

后端流程：

```text
POST /api/timeline/jobs
-> 检查 MySQL 中是否已有同 topic + mode 的 completed 时间线
-> 若命中缓存，直接返回历史结果
-> 若未命中，启动子进程 run_timeline_web_job.py
-> run_event_discovery(topic)
-> 写入 event_discovery_* MySQL 表
-> run_timeline_reasoning_pipeline(topic, run_id, mode, dry_run=False)
-> 写入 timeline_* MySQL 表
-> 前端轮询状态
-> 完成后读取 MySQL 时间线结果
```

## 7. API 接口

主要接口：

```text
GET  /api/health
POST /api/timeline/jobs
GET  /api/timeline/jobs/{job_id}/status
POST /api/timeline/jobs/{job_id}/cancel
GET  /api/timeline/jobs/{job_id}/result
GET  /api/timeline/results/{reasoning_run_id}
```

创建任务：

```http
POST /api/timeline/jobs
```

请求体：

```json
{
  "topic": "Apple",
  "mode": "fast"
}
```

## 8. 缓存复用逻辑

为了固定数据集场景下节省时间，创建 job 时会先查：

```text
timeline_reasoning_runs
```

命中条件：

```text
topic 精确一致
mode 精确一致
status = completed
```

命中后：

```text
不重复运行 SBERT / LLM
直接读取 timeline_nodes 和 timeline_node_articles
前端提示“已复用历史结果”
```

注意：

如果某 topic 只跑过 SBERT 层，LLM 层只是 dry-run 生成 JSON，没有正式写入 MySQL，则不会命中缓存，会按全新任务执行完整流程。

## 9. 生成中页面

生成中 UI 包含：

```text
- 当前阶段标题
- 百分比
- 进度条
- 已用时间
- 预计总时间
- 预计剩余
- 当前详细进度说明
- 错误或缓存提示卡
- 停止生成按钮
```

当前进度条不是严格的逐条真实进度。

原因：

```text
SBERT 层和 LLM 层当前主要是大函数调用，内部没有细粒度 callback。
```

当前实现是：

```text
后端上报真实阶段边界
前端在阶段之间做个位数平滑推进
```

默认预计总时间：

```text
fast     约 4 分钟
standard 约 7 分钟
full     约 12 分钟
```

运行过程中会根据已用时间和当前进度动态修正。

相关逻辑：

```text
services/timeline_api.py
frontend/static/app.js
```

## 10. Web Job Runner

位置：

```text
code/script/run_timeline_web_job.py
```

作用：

```text
作为子进程运行正式后端流程
通过 stdout 输出 NEWSLINE_JOB_EVENT JSON 行
API 读取这些事件并更新 job 状态
```

输出格式示例：

```text
NEWSLINE_JOB_EVENT {"event": "stage", "progress": 72, "stage": "正在进行 LLM 时间线决断", "...": "..."}
```

该脚本会自动把项目根目录加入 `sys.path`，避免从 `code/script/` 启动时出现：

```text
ModuleNotFoundError: No module named 'core'
```

## 11. 错误提示

前端不使用浏览器 alert。

错误提示显示在进度面板中的克制风格提示卡里。

后端会尝试识别：

```text
MySQL 不可用
Ollama 不可用
本地模型 / embedding 环境未就绪
无候选事件
Python import path 问题
其他后端异常
```

错误提示逻辑：

```text
services/timeline_api.py
_friendly_error_hint()
```

## 12. 结果页数据来源

结果页只从 MySQL 读取：

```text
timeline_reasoning_runs
timeline_nodes
timeline_node_articles
```

主要使用字段：

```text
timeline_nodes.display_date
timeline_nodes.display_title
timeline_nodes.canonical_title
timeline_nodes.order_index
timeline_node_articles.title
timeline_node_articles.source
timeline_node_articles.url
```

## 13. 时间线结果页

结果页隐藏：

```text
地球
输入栏
进度条
```

只显示时间线主体。

顶部：

```text
Timeline
{topic} 时间线
节点数 · mode · reasoning_run_id
返回按钮
```

时间线交互：

```text
一屏约 6 个节点
节点自动横向排布
鼠标靠左侧热区，时间线向左滚动
鼠标靠右侧热区，时间线向右滚动
鼠标悬浮节点标题，展示该簇新闻 popover
```

当前常量：

```js
const VISIBLE_TIMELINE_NODES = 6;
```

位置：

```text
frontend/static/app.js
```

## 14. 节点标题展示

最近优化：

```text
一屏节点数从 10 降为 6
节点宽度加大
节点标题最多显示 5 行
标题字号略微增加
极长标题仍用省略号兜底
```

相关 CSS：

```text
.timeline-node
.timeline-title
```

位置：

```text
frontend/static/styles.css
```

## 15. 新闻 Popover

鼠标悬浮节点标题时展示新闻列表。

当前设计：

```text
popover 宽度约 360px
顶部展示节点标题
显示“共 N 条相关新闻”
新闻列表内部滚动
默认高度约显示两条多一点
超过内容通过内部滚动查看
```

这样可以避免新闻列表过高，遮挡其他时间线节点。

相关 CSS：

```text
.article-popover
.article-count
.article-list
.article-source
```

相关 JS：

```text
showPopover()
```

位置：

```text
frontend/static/app.js
```

## 16. 当前已知限制

### 16.1 进度条不是严格真实进度

当前 SBERT 和 LLM 内部没有细粒度 callback，因此只能做到阶段真实 + 前端平滑估算。

未来可改：

```text
SBERT 层：
- alias 扩展完成
- SQL 召回完成
- Python 过滤完成
- embedding 编码进度
- 图聚类完成
- MySQL 写入完成

LLM 层：
- review_event_count 总数
- 当前已裁判事件数
- 当前批次耗时
- 已写入节点数
```

### 16.2 停止生成是终止子进程

当前点击停止会终止子进程。

未来更优方式：

```text
pipeline 内部支持 cancellation token
每个阶段主动检查是否取消
```

### 16.3 缓存逻辑默认数据集固定

当前复用逻辑适合固定数据集。

如果后续新闻数据会更新，需要增加：

```text
强制重新生成
数据版本号
新闻表最新更新时间
缓存失效机制
```

### 16.4 极长标题仍可能省略

当前节点标题最多 5 行。

未来可增加：

```text
点击节点打开详情侧栏
详情侧栏展示完整 display_title
展示全部 articles
展示 decision_reason / confidence / risk_flags
```

## 17. 下一轮优化建议

优先级较高：

```text
1. 增加节点详情侧栏
   - 点击节点后固定展示完整事件标题
   - 展示全部新闻
   - 展示 source / url
   - 展示 LLM decision_reason
   - 展示 confidence / risk_flags

2. 增加“重新生成”开关
   - 默认复用历史结果
   - 用户可选择强制重新跑 SBERT + LLM

3. 增加最近生成记录
   - 首页展示最近 topic
   - 点击直接打开历史时间线

4. 增加真实后端进度 callback
   - 尤其是 embedding 编码和 LLM 裁判阶段
```

视觉方向：

```text
保持 Apple 式克制风格
避免霓虹、赛博、过度粒子效果
优先保证信息阅读舒适度
结果页更像事件分析工具，不像营销页
```

## 18. 验证方式

基础检查：

```bash
cd /Users/hjfl/newsline
source .venv/bin/activate
.venv/bin/python -m py_compile services/timeline_api.py code/script/run_timeline_web_job.py frontend/app.py
node --check frontend/static/app.js
```

启动服务：

```bash
uvicorn services.timeline_api:app --host 127.0.0.1 --port 8000
```

检查接口：

```bash
curl http://127.0.0.1:8000/api/health
```

预期：

```json
{"status":"ok","service":"newsline-timeline"}
```
