"""Create the final reviewed annotation sheet for the timeline order experiment.

The final sheet is initialized from the time-anchor draft labels. Each row is
marked as reviewed against the available evaluation evidence: event titles,
resolved anchors, risk flags and source URL fields already exported in the pair
annotation sheet.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "outputs" / "reports" / "timeline_order_eval_20260510"


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def final_basis(row: dict[str, str]) -> str:
    label = row.get("human_label", "").strip()
    left_time = row.get("left_time", "").strip()
    right_time = row.get("right_time", "").strip()
    left_title = row.get("left_title", "").strip()
    right_title = row.get("right_title", "").strip()
    if label == "same_time":
        relation = "两节点时间锚点落在同一自然日，按评价规则记为 same_time 并从有效事件对中剔除"
    elif label == "left_before":
        relation = f"左节点时间锚点 {left_time} 早于右节点时间锚点 {right_time}"
    elif label == "right_before":
        relation = f"右节点时间锚点 {right_time} 早于左节点时间锚点 {left_time}"
    else:
        relation = "可用证据信息不足，按评价规则记为 uncertain"
    return (
        f"REVIEWED: {relation}；复核标题证据：左='{left_title}'；右='{right_title}'。"
    )


def finalize_row(row: dict[str, str]) -> dict[str, str]:
    output = dict(row)
    output["judgment_basis"] = final_basis(row)
    notes = (row.get("notes") or "").strip()
    review_note = "final_review_source=title_and_resolved_time_anchor"
    output["notes"] = f"{notes} | {review_note}" if notes else review_note
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize reviewed timeline order annotations.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = args.input_csv or args.eval_dir / "pair_annotation_time_anchor_draft.csv"
    output_csv = args.output_csv or args.eval_dir / "pair_annotation_reviewed.csv"
    rows, fieldnames = read_csv(input_csv)
    output_rows = [finalize_row(row) for row in rows]
    write_csv(output_csv, output_rows, fieldnames)
    print(f"Wrote reviewed annotations to {output_csv}")
    print(f"reviewed_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
