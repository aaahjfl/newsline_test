"""Create a review-required draft annotation file from timeline time anchors.

This helper does not replace manual annotation. It only pre-fills labels by
comparing the two resolved event time anchors so the reviewer can correct them.
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


def parse_time(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    for candidate, fmt in (
        (text, "%Y-%m-%d %H:%M:%S"),
        (text[:19], "%Y-%m-%d %H:%M:%S"),
        (text[:10], "%Y-%m-%d"),
    ):
        try:
            return datetime.strptime(candidate, fmt)
        except ValueError:
            continue
    return None


def prelabel_row(row: dict[str, str]) -> dict[str, str]:
    output = dict(row)
    left_time = parse_time(row.get("left_time", ""))
    right_time = parse_time(row.get("right_time", ""))
    if left_time is None or right_time is None:
        output["human_label"] = "uncertain"
        reason = "DRAFT_PRELABEL: missing or unparsable resolved_time_anchor; manual review required"
    elif left_time.date() == right_time.date():
        output["human_label"] = "same_time"
        reason = "DRAFT_PRELABEL: same calendar date by resolved_time_anchor; manual review required"
    elif left_time < right_time:
        output["human_label"] = "left_before"
        reason = "DRAFT_PRELABEL: left resolved_time_anchor is earlier; manual review required"
    else:
        output["human_label"] = "right_before"
        reason = "DRAFT_PRELABEL: right resolved_time_anchor is earlier; manual review required"

    existing_basis = (row.get("judgment_basis") or "").strip()
    output["judgment_basis"] = f"{existing_basis} | {reason}" if existing_basis else reason
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-fill timeline order annotations for human review.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = args.input_csv or args.eval_dir / "pair_annotation.csv"
    output_csv = args.output_csv or args.eval_dir / "pair_annotation_time_anchor_draft.csv"
    rows, fieldnames = read_csv(input_csv)
    output_rows = [prelabel_row(row) for row in rows]
    write_csv(output_csv, output_rows, fieldnames)
    print(f"Wrote draft annotations to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
