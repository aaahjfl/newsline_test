"""Formal timeline reasoning pipeline entry point."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import uuid
from typing import Any

from configs.model_config import MODEL_CONFIG
from configs.pipeline_config import PIPELINE_CONFIG
from database.db_utils import get_db_connection

from core.schemas import EventNode, TimelineNode

from .event_cards import build_event_cards
from .filters import build_rule_decision, route_event_card
from .llm_judge import judge_event_cards_with_llm
from .models import EventCard, EventDecision, TimelineReasoningResult
from .ordering import build_timeline_records
from .persistence import persist_timeline_reasoning_result
from .prompts import PROMPT_VERSION


EVENT_TABLE = "event_discovery_events"
ASSIGNMENT_TABLE = "event_discovery_assignments"
GRAPH_TABLE = "event_discovery_graph"


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    text = value.strip()
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


def _serialize_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    text = str(value).strip()
    return text or None


def get_latest_event_discovery_run_id(topic: str) -> str | None:
    """Return the latest event discovery run id for a topic."""
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT run_id
                FROM {EVENT_TABLE}
                WHERE topic = %s
                ORDER BY generated_at DESC, id DESC
                LIMIT 1
                """,
                (topic,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return row["run_id"]
    finally:
        connection.close()


def load_event_nodes_for_timeline(topic: str, run_id: str | None = None) -> tuple[str, list[EventNode]]:
    """Load one event discovery batch from MySQL for timeline reasoning."""
    resolved_run_id = run_id or get_latest_event_discovery_run_id(topic)
    if not resolved_run_id:
        return "", []

    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SHOW COLUMNS FROM {EVENT_TABLE}")
            event_columns = {row["Field"] for row in cursor.fetchall()}
            extra_columns = ""
            if {"risk_flags", "quality_metrics"}.issubset(event_columns):
                extra_columns = ",\n                risk_flags,\n                quality_metrics"

            cursor.execute(
                f"""
                SELECT
                    event_id,
                    topic,
                    cluster_size,
                    canonical_title,
                    representative_news_id,
                member_news_ids,
                event_time_start,
                event_time_end,
                event_time_anchor,
                source_count,
                confidence,
                system_is_noise,
                noise_reason
                {extra_columns}
                FROM {EVENT_TABLE}
                WHERE topic = %s AND run_id = %s
                ORDER BY COALESCE(event_time_anchor, event_time_start, event_time_end) ASC, id ASC
                """,
                (topic, resolved_run_id),
            )
            rows = list(cursor.fetchall())
    finally:
        connection.close()

    events: list[EventNode] = []
    for row in rows:
        member_news_ids_raw = row.get("member_news_ids")
        if isinstance(member_news_ids_raw, str):
            try:
                member_news_ids = json.loads(member_news_ids_raw)
            except json.JSONDecodeError:
                member_news_ids = [member_news_ids_raw]
        else:
            member_news_ids = member_news_ids_raw or []

        risk_flags_raw = row.get("risk_flags")
        if isinstance(risk_flags_raw, str):
            try:
                risk_flags = json.loads(risk_flags_raw)
            except json.JSONDecodeError:
                risk_flags = []
        else:
            risk_flags = risk_flags_raw or []
        if not isinstance(risk_flags, list):
            risk_flags = []

        quality_metrics_raw = row.get("quality_metrics")
        if isinstance(quality_metrics_raw, str):
            try:
                quality_metrics = json.loads(quality_metrics_raw)
            except json.JSONDecodeError:
                quality_metrics = {}
        else:
            quality_metrics = quality_metrics_raw or {}
        if not isinstance(quality_metrics, dict):
            quality_metrics = {}

        events.append(
            EventNode(
                event_id=row["event_id"],
                topic=row["topic"],
                member_news_ids=member_news_ids,
                cluster_size=int(row.get("cluster_size") or 0),
                canonical_title=row.get("canonical_title"),
                representative_news_id=row.get("representative_news_id"),
                event_time_start=_serialize_datetime(row.get("event_time_start")),
                event_time_end=_serialize_datetime(row.get("event_time_end")),
                event_time_anchor=_serialize_datetime(row.get("event_time_anchor")),
                source_count=int(row.get("source_count") or 0),
                confidence=float(row.get("confidence") or 0.0),
                system_is_noise=bool(row.get("system_is_noise")),
                noise_reason=row.get("noise_reason"),
                risk_flags=[str(flag) for flag in risk_flags],
                quality_metrics=quality_metrics,
            )
        )

    return resolved_run_id, events


def load_event_assignments_for_timeline(run_id: str) -> list[dict[str, Any]]:
    """Load news-level assignments for one event discovery run."""
    if not run_id:
        return []

    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    run_id,
                    topic,
                    event_id,
                    news_id,
                    title,
                    source,
                    url,
                    event_time_anchor,
                    cluster_size,
                    canonical_title,
                    system_is_noise,
                    noise_reason
                FROM {ASSIGNMENT_TABLE}
                WHERE run_id = %s
                ORDER BY event_id ASC, news_id ASC
                """,
                (run_id,),
            )
            rows = list(cursor.fetchall())
    finally:
        connection.close()

    return [
        {
            "run_id": row["run_id"],
            "topic": row["topic"],
            "event_id": row["event_id"],
            "news_id": row["news_id"],
            "title": row.get("title"),
            "source": row.get("source"),
            "url": row.get("url"),
            "event_time_anchor": _serialize_datetime(row.get("event_time_anchor")),
            "cluster_size": int(row.get("cluster_size") or 0),
            "canonical_title": row.get("canonical_title"),
            "system_is_noise": bool(row.get("system_is_noise")),
            "noise_reason": row.get("noise_reason"),
        }
        for row in rows
    ]


def load_event_graph_summary_for_timeline(run_id: str) -> dict[str, dict[str, int]]:
    """Load lightweight graph diagnostics by event id for one discovery run."""
    if not run_id:
        return {}

    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    left_event_id,
                    right_event_id,
                    edge_reason
                FROM {GRAPH_TABLE}
                WHERE run_id = %s
                """,
                (run_id,),
            )
            rows = list(cursor.fetchall())
    finally:
        connection.close()

    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        for key in ("left_event_id", "right_event_id"):
            event_id = row.get(key)
            if not event_id:
                continue
            info = summary.setdefault(
                str(event_id),
                {"graph_edge_count": 0, "semantic_override_edge_count": 0},
            )
            info["graph_edge_count"] += 1
            if row.get("edge_reason") == "semantic_override":
                info["semantic_override_edge_count"] += 1
    return summary


def build_initial_timeline(event_nodes: list[EventNode]) -> list[TimelineNode]:
    """Create a deterministic pre-LLM event order from event time fields."""
    ordered_events = sorted(
        event_nodes,
        key=lambda event: (
            _parse_iso_datetime(event.event_time_anchor)
            or _parse_iso_datetime(event.event_time_start)
            or _parse_iso_datetime(event.event_time_end)
            or datetime.max,
            event.event_id,
        ),
    )

    return [
        TimelineNode(
            event_id=event.event_id,
            order_index=index,
            reasoning_note="Initial rule-based order from event discovery timestamps.",
        )
        for index, event in enumerate(ordered_events, start=1)
    ]


def run_timeline_reasoning(
    topic_or_event_nodes: str | list[EventNode],
    run_id: str | None = None,
) -> list[TimelineNode]:
    """
    Build the initial timeline input for the future LLM reasoning stage.

    Accepted forms:
    - `run_timeline_reasoning(topic: str, run_id: str | None = None)`
    - `run_timeline_reasoning(event_nodes: list[EventNode])`
    """
    if isinstance(topic_or_event_nodes, str):
        _, event_nodes = load_event_nodes_for_timeline(topic_or_event_nodes, run_id=run_id)
        return build_initial_timeline(event_nodes)

    if isinstance(topic_or_event_nodes, list):
        return build_initial_timeline(topic_or_event_nodes)

    raise TypeError("run_timeline_reasoning expects a topic string or a list of EventNode objects.")


def _safe_topic(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("_") or "topic"


def _generate_reasoning_run_id(topic: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{_safe_topic(topic)}_timeline_{timestamp}_{suffix}"


def _export_timeline_result(result: TimelineReasoningResult) -> dict[str, str]:
    output_root = Path(str(PIPELINE_CONFIG.get("output_root", "outputs")))
    output_dir = output_root / "timeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_safe_topic(result.topic)}_timeline_{result.reasoning_run_id}.json"
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"timeline_json": str(path)}


def _slice_events_and_assignments(
    events: list[EventNode],
    assignments: list[dict[str, Any]],
    *,
    limit_events: int | None,
) -> tuple[list[EventNode], list[dict[str, Any]]]:
    if limit_events is None or limit_events <= 0:
        return events, assignments
    limited_events = events[:limit_events]
    event_ids = {event.event_id for event in limited_events}
    return limited_events, [assignment for assignment in assignments if assignment.get("event_id") in event_ids]


def _standard_review_extra_limit(card_count: int, existing_review_count: int) -> int:
    target = max(4, min(60, (card_count + 2) // 3))
    return max(0, target - existing_review_count)


def _standard_uncertainty_score(card: EventCard) -> tuple[int, float, int, str]:
    flags = set(card.risk_flags)
    score = 0
    if "translated_topic_alias_risk" in flags:
        score += 100
    if "ambiguous_topic_low_support" in flags:
        score += 40
    if "low_confidence" in flags:
        score += 20
    if "medium_confidence" in flags:
        score += 10
    if "low_source_support" in flags:
        score += 8
    if not (card.event_time_anchor or card.event_time_start or card.event_time_end):
        score += 6
    return (-score, card.confidence or 0.0, -card.cluster_size, card.event_id)


def _select_standard_uncertainty_reviews(
    cards: list[EventCard],
    *,
    existing_review_ids: set[str],
) -> list[EventCard]:
    limit = _standard_review_extra_limit(len(cards), len(existing_review_ids))
    if limit <= 0:
        return []
    candidates = [
        card
        for card in cards
        if card.event_id not in existing_review_ids
        and {
            "translated_topic_alias_risk",
            "ambiguous_topic_low_support",
            "low_confidence",
            "medium_confidence",
            "low_source_support",
        }.intersection(card.risk_flags)
    ]
    candidates.sort(key=_standard_uncertainty_score)
    return candidates[:limit]


def _route_and_decide(
    cards: list[EventCard],
    *,
    mode: str,
    model_name: str,
    llm_batch_size: int,
    llm_timeout_seconds: int,
) -> tuple[list[EventDecision], int]:
    decisions: list[EventDecision] = []
    review_cards: list[EventCard] = []
    rule_cards: list[tuple[EventCard, str]] = []

    for card in cards:
        route = route_event_card(card, mode=mode)
        if route == "llm_review":
            review_cards.append(card)
        else:
            rule_cards.append((card, route))

    normalized_mode = mode.strip().casefold()
    if normalized_mode == "standard":
        existing_review_ids = {card.event_id for card in review_cards}
        extra_reviews = _select_standard_uncertainty_reviews(cards, existing_review_ids=existing_review_ids)
        extra_review_ids = {card.event_id for card in extra_reviews}
        review_cards.extend(extra_reviews)
        for card, route in rule_cards:
            if card.event_id not in extra_review_ids:
                decisions.append(build_rule_decision(card, route=route))
    else:
        decisions.extend(build_rule_decision(card, route=route) for card, route in rule_cards)

    llm_decisions = judge_event_cards_with_llm(
        review_cards,
        model_name=model_name,
        batch_size=llm_batch_size,
        timeout_seconds=llm_timeout_seconds,
    )
    decisions.extend(llm_decisions)

    order = {card.event_id: index for index, card in enumerate(cards)}
    decisions.sort(key=lambda decision: order.get(decision.event_id, len(order)))
    return decisions, len(review_cards)


def run_timeline_reasoning_pipeline(
    topic: str,
    *,
    run_id: str | None = None,
    mode: str = "standard",
    limit_events: int | None = None,
    dry_run: bool = False,
    llm_batch_size: int = 4,
    model_name: str | None = None,
    llm_timeout_seconds: int = 300,
    extra_config: dict[str, Any] | None = None,
) -> TimelineReasoningResult:
    """Run the full LLM decision layer and materialize a display-ready timeline."""
    normalized_topic = topic.strip()
    if not normalized_topic:
        raise ValueError("topic must not be empty.")

    discovery_run_id, events = load_event_nodes_for_timeline(normalized_topic, run_id=run_id)
    if not discovery_run_id:
        raise ValueError(f"No event discovery run found for topic: {normalized_topic}")

    assignments = load_event_assignments_for_timeline(discovery_run_id)
    events, assignments = _slice_events_and_assignments(events, assignments, limit_events=limit_events)
    graph_summary = load_event_graph_summary_for_timeline(discovery_run_id)
    cards = build_event_cards(
        discovery_run_id=discovery_run_id,
        events=events,
        assignments=assignments,
        graph_summary=graph_summary,
    )

    resolved_model_name = model_name or str(MODEL_CONFIG.get("reasoning_model", "qwen3.5:9b"))
    reasoning_run_id = _generate_reasoning_run_id(normalized_topic)
    decisions, review_event_count = _route_and_decide(
        cards,
        mode=mode,
        model_name=resolved_model_name,
        llm_batch_size=llm_batch_size,
        llm_timeout_seconds=llm_timeout_seconds,
    )
    timeline = build_timeline_records(
        reasoning_run_id=reasoning_run_id,
        cards=cards,
        decisions=decisions,
    )

    result = TimelineReasoningResult(
        topic=normalized_topic,
        discovery_run_id=discovery_run_id,
        reasoning_run_id=reasoning_run_id,
        model_name=resolved_model_name,
        mode=mode,
        prompt_version=PROMPT_VERSION,
        input_event_count=len(cards),
        review_event_count=review_event_count,
        accepted_event_count=sum(1 for decision in decisions if decision.keep_event),
        rejected_event_count=sum(1 for decision in decisions if not decision.keep_event),
        timeline=timeline,
        decisions=decisions,
        decision_contexts={card.event_id: card.to_dict() for card in cards},
    )

    result.output_paths = _export_timeline_result(result)
    if not dry_run:
        config = {
            "mode": mode,
            "limit_events": limit_events,
            "llm_batch_size": llm_batch_size,
            "dry_run": dry_run,
        }
        if extra_config:
            config.update(extra_config)
        persist_timeline_reasoning_result(
            result,
            config=config,
        )
    return result
