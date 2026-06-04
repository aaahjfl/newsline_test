"""Generate traceable manual-evaluation materials for timeline ordering metrics.

The generated files support the Kendall's tau and ordering Accuracy experiment
described in thesis sections 6.2.3 and 6.3.1.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports" / "timeline_order_eval_20260510"
SEED = 20260510


@dataclass(frozen=True)
class TopicConfig:
    topic: str
    timeline_json: Path
    node_limit: int | None
    pair_target: int


TOPIC_CONFIGS = [
    TopicConfig(
        topic="Apple",
        timeline_json=PROJECT_ROOT
        / "outputs/timeline/Apple_timeline_Apple_timeline_20260508_225446_5e6228a1.json",
        node_limit=None,
        pair_target=70,
    ),
    TopicConfig(
        topic="Fed",
        timeline_json=PROJECT_ROOT / "outputs/timeline/Fed_timeline_Fed_timeline_20260509_004808_902600b4.json",
        node_limit=24,
        pair_target=80,
    ),
    TopicConfig(
        topic="美联储",
        timeline_json=PROJECT_ROOT / "outputs/timeline/topic_timeline_topic_timeline_20260509_002518_5eefc6fc.json",
        node_limit=24,
        pair_target=80,
    ),
    TopicConfig(
        topic="China",
        timeline_json=PROJECT_ROOT / "outputs/timeline/China_timeline_China_timeline_20260508_231515_767f1afc.json",
        node_limit=36,
        pair_target=100,
    ),
    TopicConfig(
        topic="Trump",
        timeline_json=PROJECT_ROOT / "outputs/timeline/Trump_timeline_Trump_timeline_20260504_010300_7c62d132.json",
        node_limit=36,
        pair_target=100,
    ),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_node_key(node: dict[str, Any]) -> tuple[int, str]:
    order = node.get("order_index")
    try:
        order_value = int(order)
    except (TypeError, ValueError):
        order_value = 10**9
    return order_value, str(node.get("event_id") or "")


def confidence_value(node: dict[str, Any]) -> float:
    value = node.get("confidence")
    if isinstance(value, (int, float)):
        return float(value)
    return 1.0


def add_selected(
    selected: dict[str, dict[str, Any]],
    node: dict[str, Any],
    *,
    stratum: str,
    reason: str,
) -> None:
    event_id = str(node.get("event_id") or "")
    if not event_id:
        return
    entry = selected.setdefault(event_id, {"node": node, "strata": [], "reasons": []})
    if stratum not in entry["strata"]:
        entry["strata"].append(stratum)
    if reason not in entry["reasons"]:
        entry["reasons"].append(reason)


def evenly_spaced(nodes: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not nodes:
        return []
    if count >= len(nodes):
        return list(nodes)
    positions = []
    for i in range(count):
        pos = round(i * (len(nodes) - 1) / (count - 1))
        positions.append(pos)
    return [nodes[pos] for pos in dict.fromkeys(positions)]


def sample_nodes(
    timeline: list[dict[str, Any]],
    *,
    limit: int | None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    ordered = sorted(timeline, key=stable_node_key)
    if limit is None or len(ordered) <= limit:
        return [
            {
                "node": node,
                "strata": ["full_topic"],
                "reasons": ["topic node count is small enough for full-node evaluation"],
            }
            for node in ordered
        ]

    selected: dict[str, dict[str, Any]] = {}
    time_count = max(8, limit // 3)
    low_conf_count = max(4, limit // 4)
    risk_count = max(4, limit // 4)

    for node in evenly_spaced(ordered, time_count):
        add_selected(
            selected,
            node,
            stratum="time_position",
            reason="evenly spaced by system order to cover beginning, middle and end",
        )

    low_conf_nodes = sorted(ordered, key=lambda n: (confidence_value(n), stable_node_key(n)))
    for node in low_conf_nodes[:low_conf_count]:
        add_selected(
            selected,
            node,
            stratum="low_confidence",
            reason="among lowest confidence timeline nodes",
        )

    risk_nodes = sorted(
        [node for node in ordered if node.get("risk_flags")],
        key=lambda n: (-len(n.get("risk_flags") or []), confidence_value(n), stable_node_key(n)),
    )
    for node in risk_nodes[:risk_count]:
        add_selected(
            selected,
            node,
            stratum="risk_flag",
            reason="contains risk flags and therefore needs manual coverage",
        )

    remaining = [node for node in ordered if str(node.get("event_id") or "") not in selected]
    rng.shuffle(remaining)
    for node in remaining:
        if len(selected) >= limit:
            break
        add_selected(
            selected,
            node,
            stratum="random_fill",
            reason=f"deterministic random fill with seed {SEED}",
        )

    rows = list(selected.values())
    rows.sort(key=lambda row: stable_node_key(row["node"]))
    return rows


def compact_articles(node: dict[str, Any], *, limit: int = 3) -> tuple[str, str, str]:
    articles = node.get("articles") or []
    titles = []
    sources = []
    urls = []
    for article in articles[:limit]:
        titles.append(str(article.get("title") or ""))
        sources.append(str(article.get("source") or ""))
        urls.append(str(article.get("url") or ""))
    return " || ".join(titles), " || ".join(sources), " || ".join(urls)


def node_to_csv_row(topic: str, run_id: str, sampled: dict[str, Any]) -> dict[str, Any]:
    node = sampled["node"]
    article_titles, article_sources, article_urls = compact_articles(node)
    return {
        "topic": topic,
        "reasoning_run_id": run_id,
        "event_id": node.get("event_id"),
        "order_index": node.get("order_index"),
        "resolved_time_anchor": node.get("resolved_time_anchor"),
        "display_date": node.get("display_date"),
        "display_title": node.get("display_title") or node.get("canonical_title"),
        "confidence": node.get("confidence"),
        "risk_flags": "|".join(str(flag) for flag in node.get("risk_flags") or []),
        "cluster_size": node.get("cluster_size"),
        "source_count": node.get("source_count"),
        "article_titles": article_titles,
        "article_sources": article_sources,
        "article_urls": article_urls,
        "sample_stratum": "|".join(sampled["strata"]),
        "sample_reason": "|".join(sampled["reasons"]),
    }


def pair_key(a: dict[str, Any], b: dict[str, Any]) -> tuple[str, str]:
    left = str(a.get("event_id") or "")
    right = str(b.get("event_id") or "")
    return tuple(sorted((left, right)))


def add_pair(
    pairs: dict[tuple[str, str], dict[str, Any]],
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    category: str,
    rng: random.Random,
) -> None:
    if a.get("event_id") == b.get("event_id"):
        return
    key = pair_key(a, b)
    if key in pairs:
        if category not in pairs[key]["pair_category"].split("|"):
            pairs[key]["pair_category"] += f"|{category}"
        return
    left, right = (a, b) if rng.random() < 0.5 else (b, a)
    pairs[key] = {"left": left, "right": right, "pair_category": category}


def build_pairs(
    sampled_nodes: list[dict[str, Any]],
    *,
    target: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    nodes = [row["node"] for row in sorted(sampled_nodes, key=lambda row: stable_node_key(row["node"]))]
    pairs: dict[tuple[str, str], dict[str, Any]] = {}

    for left, right in zip(nodes, nodes[1:]):
        add_pair(pairs, left, right, category="adjacent_sampled_nodes", rng=rng)

    risk_or_low_conf = [
        node
        for node in nodes
        if node.get("risk_flags") or confidence_value(node) <= 0.643
    ]
    for anchor in risk_or_low_conf:
        candidates = [node for node in nodes if node.get("event_id") != anchor.get("event_id")]
        candidates.sort(key=lambda n: abs(int(n.get("order_index") or 0) - int(anchor.get("order_index") or 0)))
        for candidate in candidates[:2]:
            add_pair(pairs, anchor, candidate, category="risk_or_low_confidence_neighbor", rng=rng)
            if len(pairs) >= target:
                break
        if len(pairs) >= target:
            break

    all_pairs = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            gap = abs(int(a.get("order_index") or 0) - int(b.get("order_index") or 0))
            all_pairs.append((gap, a, b))
    long_range = [item for item in all_pairs if item[0] >= max(3, len(nodes) // 4)]
    rng.shuffle(long_range)
    for _, a, b in long_range:
        if len(pairs) >= target:
            break
        add_pair(pairs, a, b, category="long_range_order_check", rng=rng)

    rng.shuffle(all_pairs)
    for _, a, b in all_pairs:
        if len(pairs) >= target:
            break
        add_pair(pairs, a, b, category="random_fill", rng=rng)

    result = list(pairs.values())
    result.sort(
        key=lambda p: (
            min(int(p["left"].get("order_index") or 0), int(p["right"].get("order_index") or 0)),
            max(int(p["left"].get("order_index") or 0), int(p["right"].get("order_index") or 0)),
        )
    )
    return result[:target]


def pair_to_csv_row(topic: str, run_id: str, index: int, pair: dict[str, Any]) -> dict[str, Any]:
    left = pair["left"]
    right = pair["right"]
    left_order = int(left.get("order_index") or 0)
    right_order = int(right.get("order_index") or 0)
    system_order = "left_before" if left_order < right_order else "right_before"
    return {
        "pair_id": f"{topic}_{index:03d}",
        "topic": topic,
        "reasoning_run_id": run_id,
        "pair_category": pair["pair_category"],
        "left_event_id": left.get("event_id"),
        "right_event_id": right.get("event_id"),
        "left_order_index": left.get("order_index"),
        "right_order_index": right.get("order_index"),
        "left_time": left.get("resolved_time_anchor"),
        "right_time": right.get("resolved_time_anchor"),
        "left_title": left.get("display_title") or left.get("canonical_title"),
        "right_title": right.get("display_title") or right.get("canonical_title"),
        "left_risk_flags": "|".join(str(flag) for flag in left.get("risk_flags") or []),
        "right_risk_flags": "|".join(str(flag) for flag in right.get("risk_flags") or []),
        "system_order": system_order,
        "human_label": "",
        "judgment_basis": "",
        "notes": "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    path: Path,
    *,
    generated_at: str,
    topic_summaries: list[dict[str, Any]],
) -> None:
    lines = [
        "# Timeline Ordering Manual Evaluation Manifest",
        "",
        f"- generated_at: {generated_at}",
        f"- random_seed: {SEED}",
        "- thesis_sections: 6.2.2, 6.2.3, 6.3.1",
        "- metrics: Kendall's tau, ordering Accuracy",
        "",
        "## Input Timeline Outputs",
        "",
        "| topic | mode | generated_at | reasoning_run_id | output_nodes | timeline_json | sha256 |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in topic_summaries:
        lines.append(
            f"| {row['topic']} | {row['mode']} | {row['generated_at']} | {row['reasoning_run_id']} | "
            f"{row['output_nodes']} | {row['timeline_json']} | `{row['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Sampling Rules",
            "",
            "- Apple uses all final timeline nodes because the latest standard-mode output has only 17 nodes.",
            "- Fed and 美联储 use 24 sampled nodes each; China and Trump use 36 sampled nodes each.",
            "- Non-full topics are sampled by system-order position, low confidence, risk flags, and deterministic random fill.",
            "- Event pairs include adjacent sampled nodes, risk/low-confidence neighbor checks, long-range order checks, and deterministic random fill.",
            "- The random fill process uses the fixed seed above.",
            "",
            "## Annotation Labels",
            "",
            "- left_before: the left event happened before the right event.",
            "- right_before: the right event happened before the left event.",
            "- same_time: the two events are same-day or practically parallel for ordering evaluation.",
            "- uncertain: the evidence is insufficient to judge a reference order.",
            "",
            "## Metric Formulas",
            "",
            "- Effective pairs exclude same_time and uncertain labels.",
            "- C is the number of effective pairs where human_label equals system_order.",
            "- D is the number of effective pairs where human_label conflicts with system_order.",
            "- Kendall's tau = (C - D) / (C + D).",
            "- Ordering Accuracy = C / (C + D).",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(path: Path) -> None:
    text = """# Timeline Ordering Evaluation Materials

This directory contains the manual annotation materials for the section 6.3.1
end-to-end timeline ordering experiment.

## Files

- `manifest.md`: fixed input files, hashes, random seed, sampling rules and formulas.
- `sampled_nodes.csv`: sampled timeline nodes with titles, times, confidence, risk flags and evidence URLs.
- `pair_annotation.csv`: event-pair annotation sheet. Fill `human_label`, `judgment_basis` and optional `notes`.
- `metrics_summary.csv`: table skeleton for thesis table 6-2. Metric fields stay blank until annotation is complete.

## Valid Human Labels

- `left_before`
- `right_before`
- `same_time`
- `uncertain`

Only `left_before` and `right_before` pairs are effective for Kendall's tau and
ordering Accuracy.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate timeline ordering evaluation annotation files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sampled_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    topic_summaries: list[dict[str, Any]] = []

    for config in TOPIC_CONFIGS:
        data = load_json(config.timeline_json)
        timeline = data.get("timeline") or []
        if data.get("topic") != config.topic:
            raise ValueError(f"Topic mismatch for {config.timeline_json}: expected {config.topic}, got {data.get('topic')}")

        sampled = sample_nodes(timeline, limit=config.node_limit, rng=rng)
        pairs = build_pairs(sampled, target=config.pair_target, rng=rng)
        run_id = str(data.get("reasoning_run_id") or "")

        sampled_rows.extend(node_to_csv_row(config.topic, run_id, row) for row in sampled)
        pair_rows.extend(pair_to_csv_row(config.topic, run_id, idx, pair) for idx, pair in enumerate(pairs, start=1))
        metric_rows.append(
            {
                "topic": config.topic,
                "输出节点数": len(timeline),
                "抽样节点数": len(sampled),
                "标注事件对数": len(pairs),
                "有效事件对数": "",
                "concordant": "",
                "discordant": "",
                "same_time": "",
                "uncertain": "",
                "Kendall's tau": "",
                "排序 Accuracy": "",
            }
        )
        topic_summaries.append(
            {
                "topic": config.topic,
                "mode": data.get("mode"),
                "generated_at": data.get("generated_at"),
                "reasoning_run_id": run_id,
                "output_nodes": len(timeline),
                "timeline_json": str(config.timeline_json.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(config.timeline_json),
            }
        )

    metric_rows.append(
        {
            "topic": "平均",
            "输出节点数": "-",
            "抽样节点数": sum(int(row["抽样节点数"]) for row in metric_rows),
            "标注事件对数": sum(int(row["标注事件对数"]) for row in metric_rows),
            "有效事件对数": "",
            "concordant": "",
            "discordant": "",
            "same_time": "",
            "uncertain": "",
            "Kendall's tau": "",
            "排序 Accuracy": "",
        }
    )

    write_csv(
        output_dir / "sampled_nodes.csv",
        sampled_rows,
        [
            "topic",
            "reasoning_run_id",
            "event_id",
            "order_index",
            "resolved_time_anchor",
            "display_date",
            "display_title",
            "confidence",
            "risk_flags",
            "cluster_size",
            "source_count",
            "article_titles",
            "article_sources",
            "article_urls",
            "sample_stratum",
            "sample_reason",
        ],
    )
    write_csv(
        output_dir / "pair_annotation.csv",
        pair_rows,
        [
            "pair_id",
            "topic",
            "reasoning_run_id",
            "pair_category",
            "left_event_id",
            "right_event_id",
            "left_order_index",
            "right_order_index",
            "left_time",
            "right_time",
            "left_title",
            "right_title",
            "left_risk_flags",
            "right_risk_flags",
            "system_order",
            "human_label",
            "judgment_basis",
            "notes",
        ],
    )
    write_csv(
        output_dir / "metrics_summary.csv",
        metric_rows,
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
    write_manifest(output_dir / "manifest.md", generated_at=generated_at, topic_summaries=topic_summaries)
    write_readme(output_dir / "README.md")

    print(f"Wrote evaluation materials to {output_dir}")
    print(f"sampled_nodes: {len(sampled_rows)}")
    print(f"pair_annotation: {len(pair_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
