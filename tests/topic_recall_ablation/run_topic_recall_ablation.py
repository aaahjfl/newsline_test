"""Sidecar multilingual topic recall ablation for thesis section 4.8.4.

This script is deliberately kept outside the production pipeline. It reads the
news database, runs the candidate recall and event-discovery steps in memory,
and writes only experiment reports under outputs/reports/.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import MODEL_CONFIG
from core.event_discovery.clustering import cluster_embeddings
from core.event_discovery.encoder import encode_titles, get_embedding_model_name
from core.event_discovery.event_builder import build_event_nodes
from core.event_discovery.pipeline import (
    _filter_candidates,
    _prepare_news_for_clustering,
    fetch_candidate_news,
)
from core.timeline_reasoning.event_cards import build_event_cards
from core.timeline_reasoning.filters import build_rule_decision, route_event_card
from core.timeline_reasoning.ordering import build_timeline_records
from core.timeline_reasoning.pipeline import _route_and_decide


STRATEGY_RAW = "raw_topic"
STRATEGY_ALIAS = "topic_alias"
STRATEGY_MULTILINGUAL = "topic_alias_multilingual"

STRATEGY_LABELS = {
    STRATEGY_RAW: "原始 topic 直接召回",
    STRATEGY_ALIAS: "topic + alias 扩展",
    STRATEGY_MULTILINGUAL: "topic + alias + 多语种扩展",
}

DEFAULT_ALIAS_MANIFEST: dict[str, dict[str, list[str]]] = {
    "Fed": {
        STRATEGY_RAW: ["Fed"],
        STRATEGY_ALIAS: ["Fed", "Federal Reserve", "Federal Reserve System"],
        STRATEGY_MULTILINGUAL: [
            "Fed",
            "Federal Reserve",
            "Federal Reserve System",
            "美联储",
            "联邦储备系统",
            "美联准",
            "联邦储备",
            "Reserva Federal",
            "Sistema de Reserva Federal",
            "연방준비제도",
            "연준",
            "Réserve fédérale",
            "Système de réserve fédérale",
            "Федеральная резервная система",
            "ФРС",
            "Фед",
            "Федеральна резервна система",
            "Kiwanda cha Fed",
        ],
    },
    "美联储": {
        STRATEGY_RAW: ["美联储"],
        STRATEGY_ALIAS: ["美联储", "联邦储备", "联邦储备系统", "联邦储备局", "美联准"],
        STRATEGY_MULTILINGUAL: [
            "美联储",
            "Federal Reserve",
            "Fed",
            "Federal Reserve System",
            "联邦储备系统",
            "联邦储备局",
            "美联准",
            "Reserva Federal",
            "Sistema de Reserva Federal",
            "연방준비제도",
            "연준",
            "미국 연방준비제도",
            "Réserve fédérale",
            "Système de réserve fédérale",
            "Федеральная резервная система",
            "ФРС",
            "Федеральная резервная система США",
            "Фед",
            "Федеральна резервна система",
            "Федеральна резервна система США",
            "Sistema wa Fed",
        ],
    },
}


@dataclass(slots=True)
class StrategyResult:
    topic: str
    strategy: str
    strategy_label: str
    run_id: str
    aliases: list[str]
    sql_candidate_count: int
    filtered_count: int
    filter_retention_rate: float | None
    event_cluster_count: int
    final_timeline_node_count: int | None
    timeline_review_event_count: int | None
    timeline_rejected_event_count: int | None
    output_detail_path: str | None = None


def _now_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _dedupe_aliases(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _load_alias_manifest(path: Path | None) -> dict[str, dict[str, list[str]]]:
    if path is None:
        return {
            topic: {strategy: list(aliases) for strategy, aliases in strategies.items()}
            for topic, strategies in DEFAULT_ALIAS_MANIFEST.items()
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("alias manifest must be a JSON object.")
    manifest: dict[str, dict[str, list[str]]] = {}
    for topic, strategies in payload.items():
        if not isinstance(strategies, dict):
            raise ValueError(f"alias manifest topic must map to strategies: {topic}")
        manifest[str(topic)] = {
            strategy: _dedupe_aliases([str(item) for item in aliases])
            for strategy, aliases in strategies.items()
            if isinstance(aliases, list)
        }
    return manifest


def _graph_summary_from_edges(
    graph_edges: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    news_to_event = {str(item["news_id"]): str(item["event_id"]) for item in assignments}
    summary: dict[str, dict[str, int]] = {}
    for edge in graph_edges:
        for key in ("left_news_id", "right_news_id"):
            event_id = news_to_event.get(str(edge.get(key)))
            if not event_id:
                continue
            info = summary.setdefault(event_id, {"graph_edge_count": 0, "semantic_override_edge_count": 0})
            info["graph_edge_count"] += 1
            if edge.get("edge_reason") == "semantic_override":
                info["semantic_override_edge_count"] += 1
    return summary


def _run_rule_timeline_proxy(
    *,
    run_id: str,
    events,
    assignments: list[dict[str, Any]],
    graph_edges: list[dict[str, Any]],
    mode: str,
) -> tuple[int, int, int]:
    graph_summary = _graph_summary_from_edges(graph_edges, assignments)
    cards = build_event_cards(
        discovery_run_id=run_id,
        events=events,
        assignments=assignments,
        graph_summary=graph_summary,
    )
    decisions = []
    review_count = 0
    for card in cards:
        route = route_event_card(card, mode=mode)
        if route == "llm_review":
            review_count += 1
            route = "auto_accept"
        decisions.append(build_rule_decision(card, route=route))
    timeline = build_timeline_records(reasoning_run_id=f"{run_id}:rule_proxy", cards=cards, decisions=decisions)
    rejected_count = sum(1 for decision in decisions if not decision.keep_event)
    return len(timeline), review_count, rejected_count


def _run_standard_timeline(
    *,
    run_id: str,
    events,
    assignments: list[dict[str, Any]],
    graph_edges: list[dict[str, Any]],
    mode: str,
    llm_batch_size: int,
    llm_timeout_seconds: int,
) -> tuple[int, int, int]:
    graph_summary = _graph_summary_from_edges(graph_edges, assignments)
    cards = build_event_cards(
        discovery_run_id=run_id,
        events=events,
        assignments=assignments,
        graph_summary=graph_summary,
    )
    decisions, review_count = _route_and_decide(
        cards,
        mode=mode,
        model_name=str(MODEL_CONFIG.get("reasoning_model", "qwen3.5:9b")),
        llm_batch_size=llm_batch_size,
        llm_timeout_seconds=llm_timeout_seconds,
    )
    timeline = build_timeline_records(reasoning_run_id=f"{run_id}:timeline", cards=cards, decisions=decisions)
    rejected_count = sum(1 for decision in decisions if not decision.keep_event)
    return len(timeline), review_count, rejected_count


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_news_item_dict(item) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in (item.metadata or {}).items():
        if key == "duplicate_members" and isinstance(value, list):
            metadata[key] = [str(getattr(member, "news_id", "")) for member in value]
        else:
            metadata[key] = value
    return {
        "news_id": item.news_id,
        "title": item.title,
        "source": item.source,
        "url": item.url,
        "publish_time": item.publish_time,
        "event_time_anchor": item.event_time_anchor,
        "event_time_start": item.event_time_start,
        "event_time_end": item.event_time_end,
        "time_granularity": item.time_granularity,
        "is_noise": item.is_noise,
        "metadata": metadata,
    }


def run_strategy(
    *,
    topic: str,
    strategy: str,
    aliases: list[str],
    report_dir: Path,
    limit: int | None,
    start_date: str | None,
    end_date: str | None,
    timeline_mode: str,
    llm_batch_size: int,
    llm_timeout_seconds: int,
) -> StrategyResult:
    aliases = _dedupe_aliases(aliases)
    run_id = f"{topic}_{strategy}_{_now_token()}"

    candidate_news = fetch_candidate_news(
        topic,
        limit=limit,
        aliases=aliases,
        start_date=start_date,
        end_date=end_date,
    )
    filtered_news = _filter_candidates(candidate_news, aliases)
    retention_rate = (len(filtered_news) / len(candidate_news)) if candidate_news else None

    events = []
    assignments: list[dict[str, Any]] = []
    graph_edges: list[dict[str, Any]] = []
    final_timeline_count: int | None = None
    review_count: int | None = None
    rejected_count: int | None = None

    if filtered_news:
        clustering_news = _prepare_news_for_clustering(filtered_news)
        embeddings = encode_titles([item.title for item in clustering_news])
        clusters, edges, similarity_matrix = cluster_embeddings(clustering_news, embeddings, topic=topic)
        events, assignments = build_event_nodes(topic, clusters, clustering_news, similarity_matrix)
        graph_edges = [edge.to_dict() for edge in edges]

        if timeline_mode == "rule":
            final_timeline_count, review_count, rejected_count = _run_rule_timeline_proxy(
                run_id=run_id,
                events=events,
                assignments=assignments,
                graph_edges=graph_edges,
                mode="standard",
            )
        elif timeline_mode in {"fast", "standard", "full"}:
            final_timeline_count, review_count, rejected_count = _run_standard_timeline(
                run_id=run_id,
                events=events,
                assignments=assignments,
                graph_edges=graph_edges,
                mode=timeline_mode,
                llm_batch_size=llm_batch_size,
                llm_timeout_seconds=llm_timeout_seconds,
            )

    detail_path = report_dir / f"{topic}_{strategy}_details.json"
    _write_json(
        detail_path,
        {
            "topic": topic,
            "strategy": strategy,
            "strategy_label": STRATEGY_LABELS[strategy],
            "run_id": run_id,
            "aliases": aliases,
            "sql_candidate_count": len(candidate_news),
            "filtered_count": len(filtered_news),
            "filter_retention_rate": retention_rate,
            "event_cluster_count": len(events),
            "final_timeline_node_count": final_timeline_count,
            "timeline_review_event_count": review_count,
            "timeline_rejected_event_count": rejected_count,
            "filtered_news": [_safe_news_item_dict(item) for item in filtered_news],
            "events": [event.to_dict() for event in events],
            "assignments": assignments,
            "graph_edges": graph_edges,
        },
    )

    return StrategyResult(
        topic=topic,
        strategy=strategy,
        strategy_label=STRATEGY_LABELS[strategy],
        run_id=run_id,
        aliases=aliases,
        sql_candidate_count=len(candidate_news),
        filtered_count=len(filtered_news),
        filter_retention_rate=retention_rate,
        event_cluster_count=len(events),
        final_timeline_node_count=final_timeline_count,
        timeline_review_event_count=review_count,
        timeline_rejected_event_count=rejected_count,
        output_detail_path=str(detail_path),
    )


def _write_metrics_csv(path: Path, rows: list[StrategyResult]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["aliases"] = " | ".join(row.aliases)
            writer.writerow(payload)


def _format_rate(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _format_count(value: int | None) -> str:
    return "" if value is None else str(value)


def _write_markdown_table(path: Path, rows: list[StrategyResult]) -> None:
    lines = [
        "| 主题 | 召回方案 | SQL 召回数量 | 过滤后数量 | 过滤保留率 | 事件簇数量 | 最终时间线节点数 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.topic,
                    row.strategy_label,
                    str(row.sql_candidate_count),
                    str(row.filtered_count),
                    _format_rate(row.filter_retention_rate),
                    str(row.event_cluster_count),
                    _format_count(row.final_timeline_node_count),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 4.8.4 多语种 topic 召回消融实验")
    parser.add_argument("--topics", nargs="+", default=["Fed", "美联储"], help="实验主题，默认 Fed 美联储")
    parser.add_argument("--limit", type=int, default=None, help="候选读取上限，仅用于快速调试")
    parser.add_argument("--start-date", default=None, help="候选新闻起始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="候选新闻结束日期，格式 YYYY-MM-DD")
    parser.add_argument("--alias-manifest", default=None, help="自定义 alias manifest JSON")
    parser.add_argument("--output-dir", default=None, help="自定义报告目录")
    parser.add_argument(
        "--timeline-mode",
        default="none",
        choices=["none", "rule", "fast", "standard", "full"],
        help=(
            "none 只跑召回和事件发现；rule 使用规则代理计数；"
            "fast/standard/full 使用内存版 LLM 决断，不写数据库"
        ),
    )
    parser.add_argument("--llm-batch-size", type=int, default=4)
    parser.add_argument("--llm-timeout-seconds", type=int, default=300)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = _load_alias_manifest(Path(args.alias_manifest) if args.alias_manifest else None)

    report_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "outputs" / "reports" / f"topic_recall_ablation_{_now_token()}"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    selected_topics = [str(topic).strip() for topic in args.topics if str(topic).strip()]
    rows: list[StrategyResult] = []
    for topic in selected_topics:
        if topic not in manifest:
            raise ValueError(f"No alias strategies configured for topic: {topic}")
        for strategy in (STRATEGY_RAW, STRATEGY_ALIAS, STRATEGY_MULTILINGUAL):
            aliases = manifest[topic].get(strategy)
            if not aliases:
                raise ValueError(f"No aliases configured for {topic}/{strategy}")
            print(f"running {topic} / {STRATEGY_LABELS[strategy]} ...", flush=True)
            rows.append(
                run_strategy(
                    topic=topic,
                    strategy=strategy,
                    aliases=aliases,
                    report_dir=report_dir,
                    limit=args.limit,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    timeline_mode=args.timeline_mode,
                    llm_batch_size=args.llm_batch_size,
                    llm_timeout_seconds=args.llm_timeout_seconds,
                )
            )

    _write_json(
        report_dir / "alias_manifest.json",
        {
            "description": "Frozen alias sets used for thesis section 4.8.4 recall ablation.",
            "topics": {topic: manifest[topic] for topic in selected_topics},
        },
    )
    _write_json(
        report_dir / "run_config.json",
        {
            "generated_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "topics": selected_topics,
            "limit": args.limit,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "timeline_mode": args.timeline_mode,
            "embedding_model": get_embedding_model_name(),
            "database_effect": "read-only SELECT from parser_newsdata; no MySQL writes",
            "production_entrypoints_called": [],
        },
    )
    _write_json(report_dir / "metrics.json", [asdict(row) for row in rows])
    _write_metrics_csv(report_dir / "metrics.csv", rows)
    _write_markdown_table(report_dir / "table_4_10_results.md", rows)

    print(f"report_dir: {report_dir}")
    print(f"metrics_csv: {report_dir / 'metrics.csv'}")
    print(f"table_markdown: {report_dir / 'table_4_10_results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
