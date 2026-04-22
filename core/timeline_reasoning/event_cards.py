"""Build compact event cards from SBERT discovery outputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from core.schemas import EventNode

from .models import EventCard


def _unique_nonempty(values: Iterable[Any], *, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def _sample_title_limit(cluster_size: int) -> int:
    if cluster_size <= 5:
        return 5
    if cluster_size <= 20:
        return 5
    return 8


def build_event_cards(
    *,
    discovery_run_id: str,
    events: list[EventNode],
    assignments: list[dict[str, Any]],
    graph_summary: dict[str, dict[str, int]] | None = None,
) -> list[EventCard]:
    """Convert event nodes and news assignments into compact, traceable cards."""
    assignments_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for assignment in assignments:
        event_id = str(assignment.get("event_id") or "")
        if event_id:
            assignments_by_event[event_id].append(assignment)

    graph_summary = graph_summary or {}
    cards: list[EventCard] = []
    for event in events:
        event_id = str(event.event_id)
        event_assignments = assignments_by_event.get(event_id, [])
        title_limit = _sample_title_limit(event.cluster_size)
        title_sample = _unique_nonempty(
            [event.canonical_title, *(assignment.get("title") for assignment in event_assignments)],
            limit=title_limit,
        )
        graph_info = graph_summary.get(event_id, {})

        articles = [
            {
                "news_id": str(assignment.get("news_id") or ""),
                "title": assignment.get("title"),
                "source": assignment.get("source"),
                "url": assignment.get("url"),
                "event_time_anchor": assignment.get("event_time_anchor"),
                "cluster_size": int(assignment.get("cluster_size") or event.cluster_size or 0),
                "canonical_title": assignment.get("canonical_title") or event.canonical_title,
                "system_is_noise": bool(assignment.get("system_is_noise")),
                "noise_reason": assignment.get("noise_reason"),
            }
            for assignment in event_assignments
        ]

        cards.append(
            EventCard(
                discovery_run_id=discovery_run_id,
                topic=event.topic,
                event_id=event_id,
                canonical_title=event.canonical_title,
                cluster_size=int(event.cluster_size or 0),
                source_count=int(event.source_count or 0),
                confidence=float(event.confidence or 0.0),
                system_is_noise=bool(event.system_is_noise),
                noise_reason=event.noise_reason,
                event_time_start=event.event_time_start,
                event_time_end=event.event_time_end,
                event_time_anchor=event.event_time_anchor,
                member_news_ids=list(event.member_news_ids),
                member_titles_sample=title_sample,
                articles=articles,
                semantic_override_edge_count=int(graph_info.get("semantic_override_edge_count") or 0),
                graph_edge_count=int(graph_info.get("graph_edge_count") or 0),
            )
        )

    return cards
