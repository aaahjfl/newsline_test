# Codex 指令：实现 `core/event_discovery/` 正式版事件发现层

请只完成 `core/event_discovery/` 的正式实现，当前 **只做 SBERT 事件发现层**

## 目标
实现一个基于 **`qwen3-4b-embedding` + 图链接聚类** 的事件发现模块：

- 数据来源是现有 MySQL 中的 `parser_newsdata`
- 每次运行由用户提供一个 `topic`
- 系统先按 `topic` 在新闻标题中筛选候选新闻
- 再对候选新闻做 embedding、构图、聚类
- 最终输出标准化事件对象 JSON 到 `outputs/clustered/`

## 固定要求
- 模型固定：`qwen3-4b-embedding`
- 聚类主方案固定：**图链接 / 无向图连通分量**
- 不要做多模型切换
- 不要重写 spaCy 时间解析逻辑
- 不要对整个数据库做全局聚类
- 不要自由发挥成别的架构

## 输入表
兼容 `parser_newsdata`，至少使用这些字段：

- `id`
- `title`
- `source`
- `url`
- `event_timestamp`
- `event_time_start`
- `event_time_end`
- `time_granularity`
- `is_noise`

其中：
- `event_timestamp` 作为事件 anchor 时间
- `event_time_start` / `event_time_end` 作为事件时间范围
- 时间缺失时允许容错，但不能因此崩溃

## 目录要求
请正式实现以下文件：

### 1. `core/event_discovery/encoder.py`
职责：
- 加载 `qwen3-4b-embedding`
- 批量编码标题
- 输出归一化 embedding
- 自动选择设备（MPS / CUDA / CPU）

接口至少提供：
```python
encode_titles(titles: list[str]) -> np.ndarray
```

要求：
- 只以标题作为 embedding 输入
- 不要拼接时间字段
- 不要加复杂 prompt

### 2. `core/event_discovery/clustering.py`
职责：
- 基于 embedding 相似度构图
- 基于无向图连通分量做事件聚类

固定逻辑：
1. 对候选新闻标题编码
2. 计算两两 cosine similarity
3. 满足条件时连边：
   - 语义相似度超过阈值
   - 且时间差没有大到离谱
4. 求连通分量作为事件簇
5. 单条新闻也保留为单新闻事件

时间约束固定为：
- 若两条新闻都存在 `event_timestamp`，计算 anchor 时间差
- 若时间差超过上限（例如 30 天），默认不连边
- 若语义相似度极高，可保留边
- 若时间缺失，不直接删边，只按语义判断

不要引入复杂候选召回系统。
当前版本直接在候选集合内部做 pairwise 相似度计算。

### 3. `core/event_discovery/event_builder.py`
职责：
- 将图簇转为标准事件对象

每个事件对象至少包含：
- `event_id`
- `topic`
- `member_news_ids`
- `cluster_size`
- `canonical_title`
- `representative_news_id`
- `event_time_start`
- `event_time_end`
- `event_time_anchor`
- `source_count`
- `confidence`

固定规则：
- `canonical_title`：选簇内最接近中心的标题
- `representative_news_id`：对应 canonical title 的新闻 id
- `event_time_anchor`：取簇内 anchor 时间中位数或最集中值
- `event_time_start`：取簇内最早 start
- `event_time_end`：取簇内最晚 end
- `source_count`：去重后的 source 数量
- `confidence`：由簇内平均相似度、簇大小、时间一致性简单加权得到

不要使用 LLM 改写标题。

### 4. `core/event_discovery/pipeline.py`
职责：
- 提供统一主入口

请实现：
```python
run_event_discovery(topic: str, limit: int | None = None) -> EventDiscoveryResult
```

固定流程：
1. 从 MySQL `parser_newsdata` 读取候选新闻
2. 过滤空标题和明显噪声
3. 编码标题
4. 构图并聚类
5. 生成事件对象
6. 返回结构化结果
7. 导出 JSON 到 `outputs/clustered/`

### 5. `core/event_discovery/__init__.py`
保证外部可以直接：
```python
from core.event_discovery import run_event_discovery
```

### 6. `core/event_discovery/legacy_adapter.py`
保留最小空壳即可，主流程不要依赖它。

## `core/schemas.py` 同步补齐
请补齐并统一以下结构：
- `NewsItem`
- `EventCluster`
- `EventNode`
- `EventDiscoveryResult`

时间字段统一命名：
- `event_time_start`
- `event_time_end`
- `event_time_anchor`

## 数据库读取要求
当前阶段直接用 MySQL

查询逻辑固定为：
- 按 `topic` 在 `title` 字段做模糊匹配
- 默认按时间升序读取
- 支持 `limit`

例如：
```sql
WHERE title LIKE %topic%
```

## 输出要求
把结果导出到：
- `outputs/clustered/{topic}_events.json`
- `outputs/clustered/{topic}_assignments.json`
- `outputs/clustered/{topic}_graph.json`

说明：
- `events.json`：事件对象列表
- `assignments.json`：新闻到事件簇的映射
- `graph.json`：边列表，便于调试构图效果

## 测试要求
补一个基础测试文件：
- `tests/test_event_discovery.py`

至少验证：
- 输入一个 topic 能跑通
- 空结果不崩溃
- 单条新闻也能形成事件对象
- 返回字段完整

## 完成后请汇报
请输出：
1. 修改了哪些文件
2. 每个文件负责什么
3. 主入口如何调用
4. 一个最小运行示例

## 严禁事项
- 不做 LLM
- 不做前端
- 不做数据库写回
- 不做多模型切换
- 不改 spaCy 层
- 不做全库自动主题发现
