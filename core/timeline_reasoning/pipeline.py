"""Formal timeline reasoning pipeline entry point."""

from __future__ import annotations

from datetime import datetime
import json
from typing import Any

from database.db_utils import get_db_connection

from core.schemas import EventNode, TimelineNode


EVENT_TABLE = "event_discovery_events"
ASSIGNMENT_TABLE = "event_discovery_assignments"


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
