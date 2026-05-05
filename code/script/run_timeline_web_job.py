"""Run the web-facing timeline job and stream machine-readable progress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.event_discovery.pipeline import run_event_discovery
from core.timeline_reasoning.pipeline import run_timeline_reasoning_pipeline


def _emit(event: str, **payload) -> None:
    print(
        "NEWSLINE_JOB_EVENT "
        + json.dumps({"event": event, **payload}, ensure_ascii=False, default=str),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NewsLine SBERT and LLM layers for the web UI.")
    parser.add_argument("--topic", required=True, help="Topic entered by the user.")
    parser.add_argument(
        "--mode",
        default="fast",
        choices=("fast", "standard", "full"),
        help="Timeline reasoning mode.",
    )
    parser.add_argument("--llm-batch-size", type=int, default=4)
    parser.add_argument("--llm-timeout-seconds", type=int, default=300)
    parser.add_argument("--start-date", default=None, help="Optional candidate start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Optional candidate end date, YYYY-MM-DD.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    topic = args.topic.strip()
    if not topic:
        _emit("error", progress=0, stage="输入 topic 为空", error="topic must not be empty.")
        return 2

    date_range_text = ""
    if args.start_date or args.end_date:
        date_range_text = f"；时间范围：{args.start_date or '最早'} 至 {args.end_date or '最晚'}"

    try:
        _emit(
            "stage",
            progress=5,
            stage="准备生成任务",
            message=f"已接收 topic「{topic}」，当前使用 {args.mode} 模式{date_range_text}。正在准备调用后端流水线。",
        )
        _emit(
            "stage",
            progress=8,
            stage="正在启动 SBERT 事件发现层",
            message="即将进行多语种 topic alias 扩展、MySQL 标题召回、标题过滤、embedding 编码与图链接聚类。",
        )
        _emit(
            "stage",
            progress=16,
            stage="正在发现候选事件簇",
            message="SBERT 层正在处理候选新闻。该阶段会从 MySQL 读取新闻标题，并用 embedding 相似度组织候选事件簇。",
        )
        discovery_result = run_event_discovery(
            topic,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        _emit(
            "stage",
            progress=64,
            stage="候选事件簇已写入 MySQL",
            message=(
                f"SBERT 层完成：SQL 召回 {discovery_result.candidate_count} 条，"
                f"过滤后 {discovery_result.filtered_count} 条，形成 {len(discovery_result.events)} 个候选事件簇。"
            ),
            discovery_run_id=discovery_result.run_id,
            candidate_count=discovery_result.candidate_count,
            filtered_count=discovery_result.filtered_count,
        )
        _emit(
            "stage",
            progress=72,
            stage="正在进行 LLM 时间线决断",
            message=(
                f"LLM 层正在读取候选事件簇，使用 {args.mode} 模式进行保留/降噪/标题修正/时间锚点决断，"
                "并将正式时间线写入 MySQL。"
            ),
            discovery_run_id=discovery_result.run_id,
        )
        reasoning_result = run_timeline_reasoning_pipeline(
            topic,
            run_id=discovery_result.run_id,
            mode=args.mode,
            dry_run=False,
            llm_batch_size=args.llm_batch_size,
            llm_timeout_seconds=args.llm_timeout_seconds,
            extra_config={
                "start_date": args.start_date,
                "end_date": args.end_date,
            },
        )
        _emit(
            "done",
            progress=100,
            stage="时间线生成完成",
            message=(
                f"LLM 决断完成：输入 {reasoning_result.input_event_count} 个事件，"
                f"保留 {reasoning_result.accepted_event_count} 个，生成 {len(reasoning_result.timeline)} 个时间线节点。"
            ),
            discovery_run_id=discovery_result.run_id,
            reasoning_run_id=reasoning_result.reasoning_run_id,
            timeline_count=len(reasoning_result.timeline),
        )
        return 0
    except Exception as exc:  # pragma: no cover - exercised through the web process.
        _emit(
            "error",
            progress=0,
            stage="生成失败",
            message=str(exc),
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
