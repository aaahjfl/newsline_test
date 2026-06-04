# 独立复核流程记录

## 1. 废止同源满分结果

最初生成的 `pair_annotation_time_anchor_draft.csv` 与 `pair_annotation_reviewed.csv` 主要依据系统输出的 `resolved_time_anchor` 预填标签。由于系统排序本身也由该时间锚点决定，若继续使用该标签计算 Kendall's tau 和排序 Accuracy，会形成同源评价，导致结果被推高到 1.0000。因此该结果仅保留为 same-anchor sanity check，并归档到 `metrics_summary_same_anchor_superseded.csv`，不作为论文最终实验结果。

## 2. 独立复核单位

正式复核不直接逐对凭系统顺序判定，而是先对 137 个抽样时间线节点建立独立参考日期，再由参考日期推导 430 个事件对标签。这样可以减少重复判断，并保证同一节点在不同事件对中的参考依据一致。

## 3. 独立参考日期判定规则

复核脚本为 `code/script/review_timeline_order_independent.py`。它不使用 `resolved_time_anchor` 作为参考真值，而是按以下保守规则判断：

1. 单文章节点：优先使用原始新闻数据中的 `raw_time`；若缺失，则使用来源 URL 中的日期；再缺失则使用标题中的显式日期。
2. 多文章节点：只有当某篇成员文章标题与节点代表标题或 canonical title 匹配时，才使用该成员文章的独立日期。
3. 多文章节点若没有代表标题匹配，或成员文章语义混杂、时间跨度较大，则标记为 `uncertain`。
4. 含歧义的斜杠日期格式不作为可靠日期使用，避免把英文 `09/01/2025` 与非英文日期格式混淆。

节点级复核结果写入 `node_reference_independent.csv`。

## 4. 事件对标签生成

事件对标签写入 `pair_annotation_independent_review.csv`：

- 若左右节点都有独立参考日期，且左侧日期早于右侧日期，标记为 `left_before`。
- 若右侧日期早于左侧日期，标记为 `right_before`。
- 若左右节点独立参考日期相同，标记为 `same_time`。
- 若任一节点缺乏独立参考日期，标记为 `uncertain`。

`same_time` 与 `uncertain` 不纳入 Kendall's tau 和排序 Accuracy 计算。

## 5. 人工复核检查

在生成独立标签后，重点复查了所有 discordant 事件对。确认这些不一致主要来自以下情况：

- Apple 中 2025-08-26 的 DW 新闻被排在 2025-08-25 的 Al Jazeera 新闻之前。
- Fed/美联储中 2026-03-18 的美联储利率新闻被排在 2026-03-13 的 Powell 传票新闻之前。
- China 中部分 Xinhua URL 日期显示的真实发布时间与系统位置不一致。
- Trump 中 2025-11-03 的 Tomahawk 相关 DW 新闻被排在 2025-10-24 的中美贸易调查新闻之前。

这些样本均由来源 URL 日期或标题显式日期支持，因此计为 discordant。

## 6. 最终统计

最终统计文件为 `metrics_summary.csv`。430 个事件对中：

- 有效事件对：159
- concordant：150
- discordant：9
- same_time：4
- uncertain：267

宏平均 Kendall's tau 为 0.8743，宏平均排序 Accuracy 为 0.9372。

## 7. 实验局限

本实验是抽样复核，不是全量人工参考时间线。对于高频主题，多文章簇可能包含滚动报道、语义相近但时间不同的新闻，以及代表标题与成员文章不完全一致的情况。为避免引入错误人工真值，本实验对这些样本采用保守剔除策略，因此有效事件对数量小于标注事件对数量。
