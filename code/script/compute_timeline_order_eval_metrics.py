"""Compute Kendall's tau and ordering Accuracy from manual pair annotations."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "outputs" / "reports" / "timeline_order_eval_20260510"
VALID_LABELS = {"left_before", "right_before", "same_time", "uncertain", ""}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def label_error(row: dict[str, str]) -> str | None:
    label = (row.get("human_label") or "").strip()
    if label not in VALID_LABELS:
        return f"{row.get('pair_id')}: invalid human_label {label!r}"
    system_order = (row.get("system_order") or "").strip()
    if system_order not in {"left_before", "right_before"}:
        return f"{row.get('pair_id')}: invalid system_order {system_order!r}"
    return None


def fmt_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def compute_topic_metrics(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    by_topic: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_topic[row["topic"]].append(row)

    metrics: dict[str, dict[str, Any]] = {}
    for topic, topic_rows in by_topic.items():
        counts = Counter((row.get("human_label") or "").strip() for row in topic_rows)
        concordant = 0
        discordant = 0
        for row in topic_rows:
            label = (row.get("human_label") or "").strip()
            if label not in {"left_before", "right_before"}:
                continue
            if label == (row.get("system_order") or "").strip():
                concordant += 1
            else:
                discordant += 1
        effective = concordant + discordant
        tau = None if effective == 0 else (concordant - discordant) / effective
        accuracy = None if effective == 0 else concordant / effective
        metrics[topic] = {
            "topic": topic,
            "标注事件对数": len(topic_rows),
            "有效事件对数": effective,
            "concordant": concordant,
            "discordant": discordant,
            "same_time": counts["same_time"],
            "uncertain": counts["uncertain"],
            "unlabeled": counts[""],
            "Kendall's tau": tau,
            "排序 Accuracy": accuracy,
        }
    return metrics


def merge_summary(
    skeleton_rows: list[dict[str, str]],
    metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    topic_metric_rows: list[dict[str, Any]] = []

    for row in skeleton_rows:
        topic = row.get("topic", "")
        if topic == "平均":
            continue
        metric = metrics.get(topic, {})
        merged = dict(row)
        if metric:
            for key in ("标注事件对数", "有效事件对数", "concordant", "discordant", "same_time", "uncertain"):
                merged[key] = metric[key]
            merged["Kendall's tau"] = fmt_metric(metric["Kendall's tau"])
            merged["排序 Accuracy"] = fmt_metric(metric["排序 Accuracy"])
            topic_metric_rows.append(metric)
        output_rows.append(merged)

    effective_metrics = [row for row in topic_metric_rows if row["有效事件对数"] > 0]
    if effective_metrics:
        macro_tau = sum(row["Kendall's tau"] for row in effective_metrics) / len(effective_metrics)
        macro_accuracy = sum(row["排序 Accuracy"] for row in effective_metrics) / len(effective_metrics)
    else:
        macro_tau = None
        macro_accuracy = None

    output_rows.append(
        {
            "topic": "平均",
            "输出节点数": "-",
            "抽样节点数": sum(int(row.get("抽样节点数") or 0) for row in output_rows),
            "标注事件对数": sum(int(row.get("标注事件对数") or 0) for row in output_rows),
            "有效事件对数": sum(int(row.get("有效事件对数") or 0) for row in output_rows),
            "concordant": sum(int(row.get("concordant") or 0) for row in output_rows),
            "discordant": sum(int(row.get("discordant") or 0) for row in output_rows),
            "same_time": sum(int(row.get("same_time") or 0) for row in output_rows),
            "uncertain": sum(int(row.get("uncertain") or 0) for row in output_rows),
            "Kendall's tau": fmt_metric(macro_tau),
            "排序 Accuracy": fmt_metric(macro_accuracy),
        }
    )
    return output_rows


def print_report(metrics: dict[str, dict[str, Any]]) -> None:
    print("topic, labeled_pairs, effective_pairs, concordant, discordant, same_time, uncertain, tau, accuracy")
    for topic in sorted(metrics):
        row = metrics[topic]
        labeled = row["标注事件对数"] - row["unlabeled"]
        tau = fmt_metric(row["Kendall's tau"])
        accuracy = fmt_metric(row["排序 Accuracy"])
        print(
            f"{topic}, {labeled}, {row['有效事件对数']}, {row['concordant']}, {row['discordant']}, "
            f"{row['same_time']}, {row['uncertain']}, {tau}, {accuracy}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute timeline ordering metrics from manual annotations.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--annotation-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--write", action="store_true", help="Update metrics_summary.csv in place.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_dir = args.eval_dir
    annotation_csv = args.annotation_csv or eval_dir / "pair_annotation.csv"
    summary_csv = args.summary_csv or eval_dir / "metrics_summary.csv"

    annotation_rows = read_csv(annotation_csv)
    errors = [error for row in annotation_rows if (error := label_error(row))]
    if errors:
        print("Annotation validation failed:", file=sys.stderr)
        for error in errors[:20]:
            print(f"- {error}", file=sys.stderr)
        if len(errors) > 20:
            print(f"- ... and {len(errors) - 20} more", file=sys.stderr)
        return 2

    metrics = compute_topic_metrics(annotation_rows)
    print_report(metrics)

    if args.write:
        skeleton_rows = read_csv(summary_csv)
        summary_rows = merge_summary(skeleton_rows, metrics)
        write_csv(
            summary_csv,
            summary_rows,
            [
                "topic",
                "输出节点数",
                "抽样节点数",
                "标注事件对数",
                "有效事件对数",
                "concordant",
                "discordant",
                "same_time",
                "uncertain",
                "Kendall's tau",
                "排序 Accuracy",
            ],
        )
        print(f"Updated {summary_csv}")

    unlabeled_count = sum(1 for row in annotation_rows if not (row.get("human_label") or "").strip())
    if unlabeled_count:
        print(f"Warning: {unlabeled_count} annotation rows are still unlabeled.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
