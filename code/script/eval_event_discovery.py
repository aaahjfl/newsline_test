"""Evaluate one formal event discovery run in a compact, readable format."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.path_config import OUTPUTS_DIR
from core.timeline_reasoning.pipeline import (
    get_latest_event_discovery_run_id,
    load_event_assignments_for_timeline,
    load_event_nodes_for_timeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估正式版事件发现结果")
    parser.add_argument("--topic", help="topic；未传 run_id 时用于读取最新批次")
    parser.add_argument("--run-id", help="指定事件发现批次 run_id")
    parser.add_argument("--top-k", type=int, default=10, help="展示前几个代表事件")
    return parser


def _resolve_run(topic: str | None, run_id: str | None) -> tuple[str, str]:
    if run_id:
        if not topic:
            topic = run_id.split("_", 1)[0]
        return topic, run_id

    if not topic:
        raise ValueError("请至少提供 --topic 或 --run-id 之一。")

    resolved_run_id = get_latest_event_discovery_run_id(topic)
    if not resolved_run_id:
        raise ValueError(f"没有找到 topic={topic!r} 的事件发现结果。")
    return topic, resolved_run_id


def _output_report_path(topic: str, run_id: str) -> Path:
    reports_dir = OUTPUTS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    safe_topic = topic.replace("/", "_").replace(" ", "_")
    return reports_dir / f"{safe_topic}_event_discovery_eval_{run_id}.json"


def main() -> int:
    args = build_parser().parse_args()
    topic, run_id = _resolve_run(args.topic, args.run_id)
    event_payload_path = OUTPUTS_DIR / "clustered" / f"{topic.replace('/', '_').replace(' ', '_')}_events.json"
    topic_aliases: list[str] = []
    if event_payload_path.exists():
        try:
            topic_aliases = json.loads(event_payload_path.read_text(encoding="utf-8")).get("topic_aliases", [])
        except Exception:
            topic_aliases = []

    resolved_run_id, events = load_event_nodes_for_timeline(topic, run_id=run_id)
    assignments = load_event_assignments_for_timeline(resolved_run_id)

    if not events:
        print(f"未找到 topic={topic} run_id={resolved_run_id} 的事件结果。")
        return 1

    size_counter = Counter(event.cluster_size for event in events)
    singletons = sum(1 for event in events if event.cluster_size == 1)
    non_noise_events = [event for event in events if not event.system_is_noise]
    avg_confidence = statistics.mean(event.confidence for event in events)
    avg_non_noise_confidence = (
        statistics.mean(event.confidence for event in non_noise_events) if non_noise_events else 0.0
    )

    summary = {
        "topic": topic,
        "run_id": resolved_run_id,
        "topic_aliases": topic_aliases,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "event_count": len(events),
        "assignment_count": len(assignments),
        "singleton_count": singletons,
        "singleton_ratio": round(singletons / len(events), 4),
        "non_noise_event_count": len(non_noise_events),
        "avg_confidence": round(avg_confidence, 4),
        "avg_non_noise_confidence": round(avg_non_noise_confidence, 4),
        "cluster_size_distribution": dict(sorted(size_counter.items())),
        "top_events": [],
    }

    assignments_by_event: dict[str, list[dict[str, object]]] = {}
    for assignment in assignments:
        assignments_by_event.setdefault(str(assignment["event_id"]), []).append(assignment)

    sorted_events = sorted(events, key=lambda event: (-event.cluster_size, -event.confidence, event.event_id))
    for event in sorted_events[: args.top_k]:
        summary["top_events"].append(
            {
                "event_id": event.event_id,
                "cluster_size": event.cluster_size,
                "confidence": event.confidence,
                "system_is_noise": event.system_is_noise,
                "noise_reason": event.noise_reason,
                "canonical_title": event.canonical_title,
                "event_time_anchor": event.event_time_anchor,
                "member_titles": [item.get("title") for item in assignments_by_event.get(event.event_id, [])],
            }
        )

    report_path = _output_report_path(topic, resolved_run_id)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"topic: {topic}")
    print(f"run_id: {resolved_run_id}")
    print(f"topic_aliases: {topic_aliases}")
    print(f"event_count: {summary['event_count']}")
    print(f"assignment_count: {summary['assignment_count']}")
    print(f"singleton_count: {summary['singleton_count']} ({summary['singleton_ratio']:.2%})")
    print(f"non_noise_event_count: {summary['non_noise_event_count']}")
    print(f"avg_confidence: {summary['avg_confidence']}")
    print(f"avg_non_noise_confidence: {summary['avg_non_noise_confidence']}")
    print(f"cluster_size_distribution: {summary['cluster_size_distribution']}")
    print(f"report: {report_path}")
    print("top_events:")
    for event in summary["top_events"]:
        print(
            f"  size={event['cluster_size']} conf={event['confidence']} "
            f"noise={event['system_is_noise']} title={event['canonical_title']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
