"""Simple CLI for the formal SBERT event discovery layer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.event_discovery import run_event_discovery


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行正式版 SBERT 事件发现层")
    parser.add_argument("--topic", help="要检索和聚类的主题关键词")
    parser.add_argument("--limit", type=int, default=None, help="候选新闻读取上限")
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

    result = run_event_discovery(topic, limit=args.limit)

    print(f"topic: {result.topic}")
    print(f"run_id: {result.run_id}")
    print(f"topic_aliases: {result.topic_aliases}")
    print(f"candidate_count: {result.candidate_count}")
    print(f"filtered_count: {result.filtered_count}")
    print(f"event_count: {len(result.events)}")
    print("outputs:")
    for name, path in result.output_paths.items():
        print(f"  {name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
