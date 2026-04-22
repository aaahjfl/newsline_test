"""Evaluate one formal event discovery run in a compact, readable format."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
from typing import Any


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
    parser.add_argument("--large-cluster-min", type=int, default=10, help="大簇诊断阈值")
    parser.add_argument("--long-span-days", type=float, default=45.0, help="长时间跨度诊断阈值")
    parser.add_argument("--sample-titles", type=int, default=5, help="每个诊断事件展示的标题样本数")
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


def _clustered_payload_path(topic: str, suffix: str) -> Path:
    safe_topic = topic.replace("/", "_").replace(" ", "_")
    return OUTPUTS_DIR / "clustered" / f"{safe_topic}_{suffix}.json"


def _load_json_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _span_days(start: str | None, end: str | None) -> float | None:
    start_dt = _parse_datetime(start)
    end_dt = _parse_datetime(end)
    if start_dt is None or end_dt is None:
        return None
    return round(abs((end_dt - start_dt).total_seconds()) / 86400.0, 3)


def _graph_summary(graph_payload: dict[str, Any]) -> dict[str, Any]:
    edges = graph_payload.get("graph_edges")
    if not isinstance(edges, list):
        edges = []

    edge_reasons = Counter(str(edge.get("edge_reason") or "unknown") for edge in edges if isinstance(edge, dict))
    similarities = [
        float(edge["similarity"])
        for edge in edges
        if isinstance(edge, dict) and edge.get("similarity") is not None
    ]
    return {
        "edge_count": len(edges),
        "edge_reasons": dict(sorted(edge_reasons.items())),
        "min_similarity": round(min(similarities), 6) if similarities else None,
        "median_similarity": round(statistics.median(similarities), 6) if similarities else None,
        "max_similarity": round(max(similarities), 6) if similarities else None,
    }


def _looks_like_rolling_title(title: str | None) -> bool:
    text = str(title or "").casefold()
    markers = ("live", "timeline", "latest", "updates", "breaking", "news |", "直播", "快讯")
    return any(marker in text for marker in markers)


def _sample_assignments(
    event_id: str,
    assignments_by_event: dict[str, list[dict[str, object]]],
    limit: int,
) -> list[str | None]:
    return [
        item.get("title")
        for item in assignments_by_event.get(event_id, [])[: max(limit, 0)]
    ]


def main() -> int:
    args = build_parser().parse_args()
    topic, run_id = _resolve_run(args.topic, args.run_id)
    event_payload = _load_json_payload(_clustered_payload_path(topic, "events"))
    graph_payload = _load_json_payload(_clustered_payload_path(topic, "graph"))
    topic_aliases = event_payload.get("topic_aliases", [])
    topic_alias_details = event_payload.get("topic_alias_details", [])
    candidate_count = event_payload.get("candidate_count")
    filtered_count = event_payload.get("filtered_count")

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

    large_cluster_min = max(2, int(args.large_cluster_min))
    long_span_days = float(args.long_span_days)
    diagnostic_events: list[dict[str, Any]] = []
    for event in events:
        span = _span_days(event.event_time_start, event.event_time_end)
        reasons: list[str] = []
        if event.cluster_size >= large_cluster_min:
            reasons.append("large_cluster")
        if span is not None and span > long_span_days:
            reasons.append("long_time_span")
        if _looks_like_rolling_title(event.canonical_title):
            reasons.append("rolling_or_live_title")
        if event.cluster_size >= large_cluster_min and event.confidence < 0.8:
            reasons.append("large_cluster_low_confidence")
        if not reasons:
            continue
        diagnostic_events.append(
            {
                "event_id": event.event_id,
                "cluster_size": event.cluster_size,
                "confidence": event.confidence,
                "diagnostic_reasons": reasons,
                "time_span_days": span,
                "canonical_title": event.canonical_title,
                "event_time_start": event.event_time_start,
                "event_time_end": event.event_time_end,
                "sample_member_titles": _sample_assignments(event.event_id, assignments_by_event, args.sample_titles),
            }
        )

    diagnostic_events.sort(
        key=lambda item: (
            -len(item["diagnostic_reasons"]),
            -int(item["cluster_size"]),
            -float(item["confidence"]),
            str(item["event_id"]),
        )
    )

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
                "time_span_days": _span_days(event.event_time_start, event.event_time_end),
                "member_titles": [item.get("title") for item in assignments_by_event.get(event.event_id, [])],
            }
        )

    summary["topic_alias_details"] = topic_alias_details
    summary["topic_alias_count"] = len(topic_aliases) if isinstance(topic_aliases, list) else 0
    summary["topic_alias_detail_count"] = len(topic_alias_details) if isinstance(topic_alias_details, list) else 0
    summary["candidate_count"] = candidate_count
    summary["filtered_count"] = filtered_count
    summary["graph_summary"] = _graph_summary(graph_payload)
    summary["diagnostic_thresholds"] = {
        "large_cluster_min": large_cluster_min,
        "long_span_days": long_span_days,
    }
    summary["diagnostic_event_count"] = len(diagnostic_events)
    summary["diagnostic_events"] = diagnostic_events[: args.top_k]

    report_path = _output_report_path(topic, resolved_run_id)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"topic: {topic}")
    print(f"run_id: {resolved_run_id}")
    print(f"topic_aliases: {topic_aliases}")
    print(f"topic_alias_count: {summary['topic_alias_count']}")
    print(f"topic_alias_detail_count: {summary['topic_alias_detail_count']}")
    if candidate_count is not None:
        print(f"candidate_count: {candidate_count}")
    if filtered_count is not None:
        print(f"filtered_count: {filtered_count}")
    print(f"event_count: {summary['event_count']}")
    print(f"assignment_count: {summary['assignment_count']}")
    print(f"singleton_count: {summary['singleton_count']} ({summary['singleton_ratio']:.2%})")
    print(f"non_noise_event_count: {summary['non_noise_event_count']}")
    print(f"avg_confidence: {summary['avg_confidence']}")
    print(f"avg_non_noise_confidence: {summary['avg_non_noise_confidence']}")
    print(f"cluster_size_distribution: {summary['cluster_size_distribution']}")
    print(f"graph_summary: {summary['graph_summary']}")
    print(f"diagnostic_event_count: {summary['diagnostic_event_count']}")
    print(f"report: {report_path}")
    print("top_events:")
    for event in summary["top_events"]:
        print(
            f"  size={event['cluster_size']} conf={event['confidence']} "
            f"noise={event['system_is_noise']} title={event['canonical_title']}"
        )
    print("diagnostic_events:")
    for event in summary["diagnostic_events"]:
        print(
            f"  size={event['cluster_size']} conf={event['confidence']} "
            f"reasons={event['diagnostic_reasons']} title={event['canonical_title']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
