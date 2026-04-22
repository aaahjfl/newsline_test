"""CLI for the formal LLM timeline reasoning layer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.timeline_reasoning.pipeline import run_timeline_reasoning_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行正式版 LLM 时间线决断层")
    parser.add_argument("--topic", help="要生成时间线的主题关键词")
    parser.add_argument("--run-id", default=None, help="指定 SBERT 事件发现层 run_id")
    parser.add_argument("--mode", default="standard", choices=["fast", "standard", "full"], help="决断模式")
    parser.add_argument("--limit-events", type=int, default=None, help="调试时限制处理的事件数")
    parser.add_argument("--dry-run", action="store_true", help="只输出 JSON，不写入 MySQL")
    parser.add_argument("--llm-batch-size", type=int, default=1, help="每次发送给 LLM 的事件卡片数量")
    parser.add_argument("--llm-timeout-seconds", type=int, default=300, help="单次 LLM 请求超时时间")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    topic = args.topic.strip() if args.topic else ""
    if not topic:
        topic = input("请输入 topic: ").strip()

    if not topic:
        print("topic 不能为空，已退出。")
        return 1

    result = run_timeline_reasoning_pipeline(
        topic,
        run_id=args.run_id,
        mode=args.mode,
        limit_events=args.limit_events,
        dry_run=args.dry_run,
        llm_batch_size=args.llm_batch_size,
        llm_timeout_seconds=args.llm_timeout_seconds,
    )

    print(f"topic: {result.topic}")
    print(f"discovery_run_id: {result.discovery_run_id}")
    print(f"reasoning_run_id: {result.reasoning_run_id}")
    print(f"model_name: {result.model_name}")
    print(f"mode: {result.mode}")
    print(f"input_event_count: {result.input_event_count}")
    print(f"review_event_count: {result.review_event_count}")
    print(f"accepted_event_count: {result.accepted_event_count}")
    print(f"rejected_event_count: {result.rejected_event_count}")
    print("outputs:")
    for name, path in result.output_paths.items():
        print(f"  {name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
