# NewsLine 前端展示层 V2 Handoff

本文档用于交接当前 NewsLine 前端展示层 V2，供下一轮界面优化、功能扩展或答辩前整理使用。

当前日期：2026-05-04

## 1. 当前定位

当前前端是：

```text
FastAPI + 静态 HTML/CSS/JS
```
正式入口仍然是：

```text
services.timeline_api:app
```

视觉方向已经确定为：

```text
Apple 式冷静、克制、高留白、低饱和科技感
```

不要走赛博、霓虹、HUD、强粒子动画路线。当前页面更像一个事件分析工具，而不是营销 landing page。

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

开发时自动重载：

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

前端静态资源：

```text
frontend/static/index.html
frontend/static/styles.css
frontend/static/app.js
```

旧 Streamlit 入口：

```text
frontend/app.py
```

`frontend/app.py` 只作为 legacy 说明入口保留，当前正式展示层不使用 Streamlit。

## 4. 页面状态

前端由 `.app-shell[data-view="..."]` 控制三种状态：

```text
idle     首页输入态
running 生成中态
result  时间线结果态
```

主要状态切换逻辑在：

```text
frontend/static/app.js
setView()
```

## 5. 首页 V2

首页布局：

```text
左侧：
- NewsLine
- 新闻事件 / 时间线重构系统
- 项目简介
- topic 输入框
- mode 下拉框：fast / standard / full
- 生成时间线按钮
- 时间范围选择
- 重新生成开关
- 最近生成记录

右侧：
- 低饱和 canvas 地球
- 克制的信息流弧线
```

地球是 Canvas 绘制，不依赖 Three.js：

```text
frontend/static/app.js
drawGlobe()
```

### 5.1 时间范围选择

当前数据集范围固定为：

```text
2025-06-01 至 2026-04-01
```

前端不使用浏览器原生 `type=date`，而是自定义年月日 select，避免出现无关年份，并保持整体视觉风格。

相关常量：

```js
const DATASET_START = "2025-06-01";
const DATASET_END = "2026-04-01";
```

相关函数：

```text
initDateControls()
fillDateControls()
getDateValue()
```

后端接收：

```json
{
  "start_date": "2025-06-01",
  "end_date": "2026-04-01"
}
```

### 5.2 重新生成开关

首页已增加：

```text
重新生成
```

默认关闭：

```text
同 topic + mode + date range 命中历史结果时直接复用缓存
```

开启后：

```text
force_regenerate = true
跳过历史缓存
重新运行 SBERT + LLM
```

请求体字段：

```json
{
  "force_regenerate": true
}
```

后端逻辑：

```text
services/timeline_api.py
CreateTimelineJobRequest.force_regenerate
create_timeline_job()
```

### 5.3 最近生成记录

首页已增加最近生成记录面板。

数据来源：

```text
MySQL timeline_reasoning_runs + timeline_nodes
```

接口：

```text
GET /api/timeline/recent?limit=6
```

返回字段：

```json
{
  "items": [
    {
      "topic": "Trump",
      "mode": "standard",
      "reasoning_run_id": "Trump_timeline_20260504_010300_7c62d132",
      "generated_at": "2026-05-04 01:54:03",
      "node_count": 1956,
      "start_date": "2025-06-01",
      "end_date": "2026-04-01"
    }
  ]
}
```

前端展示格式：

```text
Trump
1956 节点 · standard 模式 · 2025-06-01 至 2026-04-01 · 生成于 2026年05月04日 01:54
```

点击最近记录：

```text
GET /api/timeline/results/{reasoning_run_id}
直接打开历史时间线
不重新运行后端流程
```

最近记录面板已针对非全屏窗口做过细节优化：

```text
- 面板内部可纵向滚动
- 滚动条保持克制风格
- 低高度桌面窗口下压缩记录高度
- 避免 Apple / Trump 记录被页面底部裁掉
```

相关函数：

```text
loadRecentTimelines()
formatRecentMeta()
formatGeneratedTime()
loadRecentTimeline()
```

## 6. 生成流程

用户提交后：

```text
POST /api/timeline/jobs
```

请求体：

```json
{
  "topic": "Trump",
  "mode": "standard",
  "start_date": "2025-06-01",
  "end_date": "2026-04-01",
  "force_regenerate": false
}
```

后端流程：

```text
1. 检查 topic / mode / date range 是否有 completed 时间线
2. 若命中缓存且 force_regenerate=false，直接返回 cached job
3. 若未命中或 force_regenerate=true，启动 run_timeline_web_job.py 子进程
4. 子进程执行 SBERT 事件发现层
5. 事件发现结果写入 event_discovery_* MySQL 表
6. 子进程执行 LLM 时间线决断层
7. 时间线结果写入 timeline_reasoning_runs / timeline_nodes / timeline_node_articles
8. 前端轮询 job status
9. 完成后前端读取 MySQL 结果并进入 result 页面
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
GET  /api/timeline/recent?limit=6
```

### 7.1 缓存匹配逻辑

缓存查询表：

```text
timeline_reasoning_runs
```

命中条件：

```text
topic 精确一致
mode 精确一致
status = completed
date range 一致
```

注意：

```text
无 date range 的旧 run 和完整数据集范围 2025-06-01 至 2026-04-01 互相兼容。
```

原因：

```text
旧版本没有 date range 字段，默认语义等价于全数据集。
```

相关函数：

```text
find_cached_timeline_run()
_config_matches_date_range()
_build_cached_job()
```

## 8. 生成中页面

生成中页面包含：

```text
- 当前阶段
- 百分比
- 进度条
- 已用时间
- 预计总时间
- 预计剩余
- 当前详细提示
- 错误提示卡
- 停止生成按钮
```

当前进度条是：

```text
后端阶段真实进度 + 前端个位数平滑推进
```

不是严格逐条真实进度。原因是 SBERT / LLM 层内部还没有细粒度 callback。

默认预计时间：

```text
fast     约 4 分钟
standard 约 7 分钟
full     约 12 分钟
```

相关函数：

```text
setProgress()
ensureProgressLoop()
updateTimeMeta()
getModeEstimateSeconds()
```

## 9. 错误提示

前端不使用浏览器 alert。

错误提示显示在进度面板中的克制风格 notice card。

后端会尝试识别：

```text
MySQL 不可用
Ollama 不可用
本地模型 / embedding 环境未就绪
无候选事件
Python import path 问题
其他后端异常
```

相关函数：

```text
services/timeline_api.py
_friendly_error_hint()
```

## 10. 时间线结果页 V2

结果页隐藏首页元素：

```text
地球
输入栏
进度条
最近生成记录
```

顶部信息：

```text
Timeline
{topic} 时间线
节点数 · mode · date range · 总用时 / 复用历史结果 · reasoning_run_id
返回按钮
```

示例：

```text
1956 个时间线节点 · standard 模式 · 2025-06-01 至 2026-04-01 · 复用历史结果 · Trump_timeline_20260504_010300_7c62d132
```

如果是新生成任务，完成后会显示：

```text
总用时 X 分 X 秒
```

相关函数：

```text
renderTimeline()
formatDateRange()
formatDuration()
```

## 11. 时间线展示

当前一屏节点数：

```js
const VISIBLE_TIMELINE_NODES = 6;
```

原因：

```text
Trump 这类 topic 节点可能接近 2000 个。
一屏 6 个节点能提高标题可读性，并减少标题省略。
```

滚动方式：

```text
鼠标靠左侧热区，时间线向左滚动
鼠标靠右侧热区，时间线向右滚动
滚动速度随距离边缘远近变化
```

边缘渐隐：

```text
中间滚动时左右边缘有渐隐
开头时左侧不渐隐
结尾时右侧不渐隐
```

相关函数：

```text
updateTimelineEdgeFades()
timelineFrame mousemove listener
ensureScrollLoop()
```

### 11.1 节点标题块

节点标题块已优化为：

```text
- 圆角矩形
- 最多显示 5 行
- 宽度按一屏 6 个节点动态计算
- 选中状态使用克制浅灰底
- 时间轴主线固定在节点底层，不压住标题
```

相关 CSS：

```text
.timeline-node
.timeline-title
.timeline-node[data-selected="true"] .timeline-title
.timeline-rail::before
```

注意：

```text
时间轴主线使用 z-index 放在节点下方。
timeline-rail 使用 isolation: isolate，避免层级穿透。
```

## 12. Hover 新闻预览 Popover

鼠标悬浮节点标题时展示快速新闻预览。

当前设计：

```text
- 宽度约 360px
- 顶部显示节点标题
- 显示“共 N 条相关新闻”
- 新闻列表内部滚动
- 只作为预览，不承担主要点击跳转职责
```

重要交互修复：

```text
article-popover 设置 pointer-events: none
```

原因：

```text
长标题 / 多新闻节点的 popover 可能视觉上覆盖原节点。
如果 popover 接管鼠标事件，会导致用户难以点击节点打开右侧详情栏。
```

当前策略：

```text
popover 是视觉浮层，鼠标点击会穿透到下面的时间节点。
新闻链接点击统一放到右侧详情栏里处理。
```

相关 CSS：

```text
.article-popover {
  pointer-events: none;
}
```

相关函数：

```text
showPopover()
scheduleHidePopover()
hidePopover()
```

## 13. 节点详情侧栏

点击任意时间线节点标题，会打开右侧节点详情侧栏。

侧栏内容：

```text
- 节点完整 display title
- canonical title
- 时间锚点
- 相关新闻数量
- 聚类规模
- 置信度
- risk flags
- decision_reason
- split_reason / merge_reason
- 全部相关新闻标题、source、url
```

交互：

```text
点击节点标题：打开 / 切换侧栏内容
点击关闭按钮：关闭侧栏
点击侧栏外空白：关闭侧栏
点击另一个时间节点：切换详情，不误关
点击侧栏内部：不关闭
```

相关函数：

```text
openNodeDrawer()
hideNodeDrawer()
formatConfidence()
document pointerdown listener
```

相关 CSS：

```text
.node-drawer
.node-drawer[data-open="true"]
.node-detail-grid
.risk-row
.node-reason
.drawer-article-list
```

移动端：

```text
侧栏在窄屏下变成底部抽屉。
```

## 14. 大规模节点性能

已用 Trump standard 测过接近：

```text
1956 个时间线节点
```

当前直接渲染全部节点，未做虚拟化。

实测浏览器不卡顿，因此暂时不做窗口虚拟化。

当前判断：

```text
2000 级别节点可以接受。
除非未来节点数上到 5000+ 或移动端明显卡顿，否则没必要引入复杂虚拟化。
```

如果未来要做虚拟化，需要特别注意：

```text
横向滚动中心定位
hover popover 定位
侧栏选中节点状态
边缘渐隐
滚动热区
```

## 15. 日期筛选与后端 SQL

前端 date range 会传入后端。

后端通过 Web job runner 继续传入 SBERT 层：

```text
code/script/run_timeline_web_job.py
--start-date
--end-date
```

SBERT 层 SQL 过滤使用事件时间字段：

```text
COALESCE(event_timestamp, event_time_start, event_time_end, standard_timestamp)
```

相关文件：

```text
core/event_discovery/pipeline.py
code/script/run_event_discovery.py
code/script/run_timeline_web_job.py
```

注意：

```text
加日期筛选不应改变原有纯后端 CLI 的默认逻辑。
不传 start_date / end_date 时仍走全数据集。
```

## 16. 当前已知限制

### 16.1 进度仍不是严格真实进度

当前依旧是阶段真实 + 前端平滑估算。

未来可在 SBERT / LLM 层增加 callback，例如：

```text
SBERT:
- SQL 召回完成
- embedding 编码进度
- 图构建完成
- 聚类完成
- MySQL 写入完成

LLM:
- review_event_count 总数
- 当前已裁判事件数
- 当前批次耗时
- 已写入节点数
```

### 16.2 停止生成仍是终止子进程

当前停止生成会 terminate 子进程。

更优方案：

```text
pipeline 内部支持 cancellation token
每个阶段主动检查是否取消
安全写入 job 状态
```

### 16.3 缓存逻辑假设数据集固定

当前缓存复用适合固定数据集。

如果后续新闻持续更新，需要增加：

```text
数据版本号
raw/parser 表最新更新时间
缓存失效机制
用户可选择是否复用旧 run
```

### 16.4 Hover 预览不再支持直接点链接

这是有意设计。

原因：

```text
为了解决 hover 浮层遮挡节点点击的问题，popover 已设置 pointer-events: none。
```

新闻跳转入口在：

```text
右侧节点详情侧栏
```

## 17. 下一轮优化建议

优先级较高：

```text
1. 进一步增强节点详情侧栏
   - 增加 event_id / order_index 调试信息开关
   - 增加复制节点信息按钮
   - 增加“在时间线上定位此节点”

2. 增加时间线局部导航
   - 顶部 mini-map 或日期跳转
   - 适合 Trump 这种 1000+ 节点时间线

3. 增加生成记录筛选
   - 最近生成记录按 topic / mode 过滤
   - 展示更多历史 run

4. 增加真实后端进度 callback
   - 这是答辩时最容易解释清楚的工程优化点
```

视觉原则：

```text
保持克制、清爽、低饱和
少加装饰，多提升信息可读性
卡片圆角控制在 8px-18px 范围
避免过亮蓝色、霓虹、复杂阴影
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

接口检查：

```bash
curl http://127.0.0.1:8000/api/health
curl 'http://127.0.0.1:8000/api/timeline/recent?limit=3'
```

预期 health：

```json
{"status":"ok","service":"newsline-timeline"}
```

## 19. 当前最重要的交互回归点

下一次改前端时，建议优先手动检查：

```text
1. 首页非全屏窗口下：
   - 最近生成记录是否完整
   - 左侧上下留白是否均衡
   - 右侧地球是否视觉居中

2. 时间线页：
   - 开头左侧是否不渐隐
   - 结尾右侧是否不渐隐
   - 中间滚动时左右渐隐是否正常
   - hover popover 是否不挡点击节点
   - 点击节点是否打开侧栏
   - 点击空白是否关闭侧栏
   - 时间轴线是否不压住节点标题

3. 缓存逻辑：
   - 同 topic / mode / date range 默认复用
   - 开启重新生成后会跳过缓存
   - 最近生成记录点击后直接打开历史时间线
```
