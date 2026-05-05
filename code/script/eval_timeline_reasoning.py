"""Evaluate and summarize timeline reasoning JSON outputs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _counter_by(items: list[dict[str, Any]], key: str) -> Counter:
    counter: Counter = Counter()
    for item in items:
        value = item.get(key) if key in item else None
        counter["null" if value is None else str(value)] += 1
    return counter


def _risk_flag_counter(contexts: dict[str, dict[str, Any]]) -> Counter:
    counter: Counter = Counter()
    for context in contexts.values():
        counter.update(str(flag) for flag in context.get("risk_flags") or [])
    return counter


def _decision_lookup(decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(decision.get("event_id")): decision for decision in decisions if decision.get("event_id")}


def _suspicious_kept_events(data: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    contexts = data.get("decision_contexts") or {}
    decisions = _decision_lookup(data.get("decisions") or [])
    rows: list[dict[str, Any]] = []
    for event_id, decision in decisions.items():
        if not decision.get("keep_event"):
            continue
        context = contexts.get(event_id, {})
        risk_flags = set(context.get("risk_flags") or [])
        reasons = []
        if decision.get("decision_confidence", 1.0) < 0.5:
            reasons.append("low_decision_confidence")
        if decision.get("time_confidence", 1.0) < 0.5:
            reasons.append("low_time_confidence")
        if context.get("cluster_size", 0) > 1 and not decision.get("needs_split"):
            if risk_flags.intersection({"long_time_span", "low_temporal_coherence", "low_semantic_cohesion"}):
                reasons.append("multi_article_structural_risk_without_split")
        if risk_flags.intersection({"rolling_coverage", "rolling_coverage_title"}):
            reasons.append("rolling_kept")
        if reasons:
            rows.append(
                {
                    "event_id": event_id,
                    "title": decision.get("display_title") or context.get("canonical_title"),
                    "reasons": reasons,
                    "risk_flags": sorted(risk_flags),
                    "decision_reason": decision.get("decision_reason"),
                }
            )
    return rows[:limit]


def build_report(data: dict[str, Any], *, suspicious_limit: int = 20) -> str:
    decisions = data.get("decisions") or []
    timeline = data.get("timeline") or []
    contexts = data.get("decision_contexts") or {}
    risk_counter = _risk_flag_counter(contexts)

    lines = [
        "# Timeline Reasoning Evaluation",
        "",
        f"- topic: {data.get('topic')}",
        f"- discovery_run_id: {data.get('discovery_run_id')}",
        f"- reasoning_run_id: {data.get('reasoning_run_id')}",
        f"- mode: {data.get('mode')}",
        f"- prompt_version: {data.get('prompt_version')}",
        f"- input_event_count: {data.get('summary', {}).get('input_event_count')}",
        f"- timeline_count: {len(timeline)}",
        "",
        "## Decision Sources",
    ]
    for name, count in _counter_by(decisions, "decision_source").most_common():
        lines.append(f"- {name}: {count}")

    boolean_fields = ("keep_event", "final_is_noise", "needs_split", "needs_merge")
    for field in boolean_fields:
        lines.extend(["", f"## {field}"])
        for name, count in _counter_by(decisions, field).most_common():
            lines.append(f"- {name}: {count}")

    lines.extend(["", "## Risk Flags"])
    if risk_counter:
        for name, count in risk_counter.most_common():
            lines.append(f"- {name}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Suspicious Kept Events"])
    suspicious = _suspicious_kept_events(data, limit=suspicious_limit)
    if suspicious:
        for row in suspicious:
            lines.append(f"- {row['event_id']}: {row['title']}")
            lines.append(f"  reasons: {', '.join(row['reasons'])}")
            lines.append(f"  risk_flags: {', '.join(row['risk_flags']) or 'none'}")
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a timeline reasoning JSON output.")
    parser.add_argument("timeline_json", type=Path, help="Path to outputs/timeline/*.json")
    parser.add_argument("--markdown-out", type=Path, default=None, help="Optional markdown report path")
    parser.add_argument("--suspicious-limit", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = _load_json(args.timeline_json)
    report = build_report(data, suspicious_limit=max(0, args.suspicious_limit))
    print(report)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
