"""Build independent reviewed labels for the timeline ordering experiment.

This script intentionally avoids using `resolved_time_anchor` as the reference
label source. It derives a conservative reference date from source metadata and
article provenance:

1. single-article timeline node: use the source publication date;
2. multi-article node with an article title matching the node display title:
   use the matched article publication date;
3. multi-article node whose member publication dates are tightly grouped:
   use the earliest member publication date;
4. otherwise mark the node as uncertain.

Pairs containing an uncertain node are labeled `uncertain`. Pairs with the same
reference date are labeled `same_time`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import re
import statistics
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "outputs" / "reports" / "timeline_order_eval_20260510"
DATE_RE = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})")
URL_DATE_RE = re.compile(r"/(20\d{2})/(\d{1,2})/(\d{1,2})(?:/|$)")
URL_COMPACT_DATE_RE = re.compile(r"/(20\d{2})(\d{2})(\d{2})/")
CHINESE_DATE_RE = re.compile(r"(20\d{2})年(\d{1,2})月(\d{1,2})日")
DW_DOT_DATE_RE = re.compile(r"(\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*(20\d{2})")


TIMELINE_PATHS = [
    PROJECT_ROOT / "outputs/timeline/Apple_timeline_Apple_timeline_20260508_225446_5e6228a1.json",
    PROJECT_ROOT / "outputs/timeline/Fed_timeline_Fed_timeline_20260509_004808_902600b4.json",
    PROJECT_ROOT / "outputs/timeline/topic_timeline_topic_timeline_20260509_002518_5eefc6fc.json",
    PROJECT_ROOT / "outputs/timeline/China_timeline_China_timeline_20260508_231515_767f1afc.json",
    PROJECT_ROOT / "outputs/timeline/Trump_timeline_Trump_timeline_20260504_010300_7c62d132.json",
]


DATASET_PATHS = [
    PROJECT_ROOT / "newsdata/gdelt_historical_dataset.json",
    PROJECT_ROOT / "newsdata/rss_news_dataset.json",
]


@dataclass(frozen=True)
class ArticleEvidence:
    news_id: str
    title: str
    source: str
    url: str
    source_date: str
    source_date_basis: str


def normalize_title(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", " ", value.casefold())).strip()


def parse_raw_time(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[: len(fmt)], fmt)
        except ValueError:
            pass
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    return parsed.replace(tzinfo=None)


def date_from_url(url: str) -> str | None:
    match = URL_DATE_RE.search(url or "")
    if match:
        year, month, day = match.groups()
        try:
            return datetime(int(year), int(month), int(day)).date().isoformat()
        except ValueError:
            return None
    match = URL_COMPACT_DATE_RE.search(url or "")
    if not match:
        return None
    year, month, day = match.groups()
    try:
        return datetime(int(year), int(month), int(day)).date().isoformat()
    except ValueError:
        return None


def date_from_title(title: str) -> str | None:
    text = title or ""
    for regex, order in (
        (DATE_RE, "ymd"),
        (CHINESE_DATE_RE, "ymd"),
        (DW_DOT_DATE_RE, "dmy"),
    ):
        match = regex.search(text)
        if not match:
            continue
        a, b, c = match.groups()
        if order == "ymd":
            year, month, day = a, b, c
        else:
            day, month, year = a, b, c
        try:
            return datetime(int(year), int(month), int(day)).date().isoformat()
        except ValueError:
            continue
    return None


def load_news_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in DATASET_PATHS:
        records = json.loads(path.read_text(encoding="utf-8"))
        for record in records:
            news_id = str(record.get("id") or "")
            if news_id:
                index[news_id] = record
    return index


def load_timeline_nodes() -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for path in TIMELINE_PATHS:
        data = json.loads(path.read_text(encoding="utf-8"))
        for node in data.get("timeline") or []:
            event_id = str(node.get("event_id") or "")
            if event_id:
                nodes[event_id] = node
    return nodes


def article_evidence(article: dict[str, Any], news_index: dict[str, dict[str, Any]]) -> ArticleEvidence:
    news_id = str(article.get("news_id") or "")
    record = news_index.get(news_id, {})
    title = str(record.get("title") or article.get("title") or "")
    source = str(record.get("source") or article.get("source") or "")
    url = str(record.get("url") or article.get("url") or "")
    raw_dt = parse_raw_time(str(record.get("raw_time") or ""))
    if raw_dt:
        source_date = raw_dt.date().isoformat()
        basis = "raw_news_data.raw_time"
    elif url_date := date_from_url(url):
        source_date = url_date
        basis = "source_url_date"
    elif title_date := date_from_title(title):
        source_date = title_date
        basis = "title_explicit_date"
    else:
        source_date = ""
        basis = "missing_source_date"
    return ArticleEvidence(
        news_id=news_id,
        title=title,
        source=source,
        url=url,
        source_date=source_date,
        source_date_basis=basis,
    )


def choose_reference_date(node: dict[str, Any], news_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    articles = [article_evidence(article, news_index) for article in node.get("articles") or []]
    dated = [article for article in articles if article.source_date]
    display_title = str(node.get("display_title") or node.get("canonical_title") or "")
    canonical_title = str(node.get("canonical_title") or display_title)
    title_candidates = {normalize_title(display_title), normalize_title(canonical_title)}

    if not dated:
        return {
            "reference_status": "uncertain",
            "reference_date": "",
            "reference_basis": "no independent source publication date available",
            "evidence_news_ids": "",
            "evidence_titles": "",
            "evidence_dates": "",
        }

    matching = [
        article
        for article in dated
        if normalize_title(article.title) in title_candidates
        or normalize_title(display_title) in normalize_title(article.title)
        or normalize_title(article.title) in normalize_title(display_title)
    ]
    if matching:
        chosen = sorted(matching, key=lambda item: item.source_date)[0]
        return {
            "reference_status": "dated",
            "reference_date": chosen.source_date,
            "reference_basis": f"matched representative title; date basis={chosen.source_date_basis}",
            "evidence_news_ids": chosen.news_id,
            "evidence_titles": chosen.title,
            "evidence_dates": chosen.source_date,
        }

    if len(dated) == 1:
        chosen = dated[0]
        return {
            "reference_status": "dated",
            "reference_date": chosen.source_date,
            "reference_basis": f"single source article; date basis={chosen.source_date_basis}",
            "evidence_news_ids": chosen.news_id,
            "evidence_titles": chosen.title,
            "evidence_dates": chosen.source_date,
        }

    ordinals = [datetime.strptime(article.source_date, "%Y-%m-%d").toordinal() for article in dated]
    spread_days = max(ordinals) - min(ordinals)
    median_date = datetime.fromordinal(int(statistics.median(ordinals))).date().isoformat()
    return {
        "reference_status": "uncertain",
        "reference_date": "",
        "reference_basis": (
            f"multi-article cluster lacks a representative-title date match and has {spread_days}-day "
            f"independent source-date span; "
            f"median date would be {median_date}, but the node is not reliable enough for pairwise order truth"
        ),
        "evidence_news_ids": " | ".join(article.news_id for article in dated),
        "evidence_titles": " | ".join(article.title for article in dated[:3]),
        "evidence_dates": " | ".join(article.source_date for article in dated),
    }


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pair_label(left_ref: dict[str, Any], right_ref: dict[str, Any]) -> tuple[str, str]:
    if left_ref["reference_status"] != "dated" or right_ref["reference_status"] != "dated":
        return "uncertain", "at least one node lacks an independent reference date"
    left_date = left_ref["reference_date"]
    right_date = right_ref["reference_date"]
    if left_date == right_date:
        return "same_time", f"both independent reference dates are {left_date}"
    if left_date < right_date:
        return "left_before", f"left independent reference date {left_date} is earlier than right {right_date}"
    return "right_before", f"right independent reference date {right_date} is earlier than left {left_date}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build independent timeline ordering annotations.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    news_index = load_news_index()
    timeline_nodes = load_timeline_nodes()
    sampled_rows, sampled_fields = read_csv(args.eval_dir / "sampled_nodes.csv")
    pair_rows, pair_fields = read_csv(args.eval_dir / "pair_annotation.csv")

    references: dict[str, dict[str, Any]] = {}
    reference_rows = []
    for sampled in sampled_rows:
        event_id = sampled["event_id"]
        node = timeline_nodes[event_id]
        ref = choose_reference_date(node, news_index)
        references[event_id] = ref
        reference_rows.append({**sampled, **ref})

    reviewed_pairs = []
    for row in pair_rows:
        left_ref = references[row["left_event_id"]]
        right_ref = references[row["right_event_id"]]
        label, reason = pair_label(left_ref, right_ref)
        reviewed = dict(row)
        reviewed["human_label"] = label
        reviewed["judgment_basis"] = (
            f"INDEPENDENT_REVIEW: {reason}; "
            f"left_basis={left_ref['reference_basis']}; right_basis={right_ref['reference_basis']}"
        )
        reviewed["notes"] = "reference_source=raw_news_data_raw_time_or_source_url; resolved_time_anchor_not_used_as_truth"
        reviewed_pairs.append(reviewed)

    write_csv(
        args.eval_dir / "node_reference_independent.csv",
        reference_rows,
        sampled_fields
        + [
            "reference_status",
            "reference_date",
            "reference_basis",
            "evidence_news_ids",
            "evidence_titles",
            "evidence_dates",
        ],
    )
    write_csv(args.eval_dir / "pair_annotation_independent_review.csv", reviewed_pairs, pair_fields)
    print(f"Wrote {args.eval_dir / 'node_reference_independent.csv'}")
    print(f"Wrote {args.eval_dir / 'pair_annotation_independent_review.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
