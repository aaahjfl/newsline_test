"""Build compact event cards from SBERT discovery outputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import re
from typing import Any

from core.schemas import EventNode

from .models import EventCard
from .topic_profile import build_topic_profile


TITLE_CONTAINER_RE = re.compile(
    r"\b(live|updates?|latest|timeline|breaking|rolling|as it happened)\b|直播|快讯",
    flags=re.IGNORECASE,
)

QUALITY_SUMMARY_KEYS = (
    "semantic_cohesion",
    "temporal_coherence",
    "support_score",
    "graph_density",
    "duplicate_ratio",
    "unique_title_count",
    "article_count",
    "time_span_days",
)


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


def _quality_summary(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in QUALITY_SUMMARY_KEYS if key in value}


def _evidence_item(
    *,
    title: Any,
    source: Any = None,
    event_time_anchor: Any = None,
    news_id: Any = None,
    evidence_role: str,
) -> dict[str, Any] | None:
    text = str(title or "").strip()
    if not text:
        return None
    return {
        "title": text,
        "source": str(source).strip() if source else None,
        "event_time_anchor": str(event_time_anchor).strip() if event_time_anchor else None,
        "news_id": str(news_id).strip() if news_id else None,
        "evidence_role": evidence_role,
    }


def _add_unique_evidence(
    result: list[dict[str, Any]],
    seen_titles: set[str],
    item: dict[str, Any] | None,
    *,
    limit: int,
) -> None:
    if item is None:
        return
    key = item["title"].casefold()
    if key in seen_titles:
        if item.get("evidence_role") == "container_title":
            for existing in result:
                if existing["title"].casefold() == key:
                    existing["evidence_role"] = "container_title"
                    break
        return
    if len(result) >= limit:
        return
    seen_titles.add(key)
    result.append(item)


def _build_member_title_evidence(
    *,
    canonical_title: str | None,
    event_time_anchor: str | None,
    assignments: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Pick a small, diverse title sample for cluster-coherence judgment."""
    result: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    _add_unique_evidence(
        result,
        seen_titles,
        _evidence_item(
            title=canonical_title,
            event_time_anchor=event_time_anchor,
            evidence_role="canonical",
        ),
        limit=limit,
    )

    candidates = [
        item
        for item in (
            _evidence_item(
                title=assignment.get("title"),
                source=assignment.get("source"),
                event_time_anchor=assignment.get("event_time_anchor"),
                news_id=assignment.get("news_id"),
                evidence_role="member",
            )
            for assignment in assignments
        )
        if item is not None
    ]
    if not candidates:
        return result

    time_sorted = sorted(
        candidates,
        key=lambda item: (
            item.get("event_time_anchor") or "9999-12-31 23:59:59",
            item.get("source") or "",
            item["title"],
        ),
    )
    strategic_indices = [0, len(time_sorted) // 2, len(time_sorted) - 1]
    for index in strategic_indices:
        if 0 <= index < len(time_sorted):
            item = dict(time_sorted[index])
            item["evidence_role"] = "time_spread"
            _add_unique_evidence(result, seen_titles, item, limit=limit)

    for item in candidates:
        if TITLE_CONTAINER_RE.search(item["title"]):
            item = dict(item)
            item["evidence_role"] = "container_title"
            _add_unique_evidence(result, seen_titles, item, limit=limit)

    seen_sources: set[str] = set()
    for item in candidates:
        source = item.get("source") or ""
        if not source or source in seen_sources:
            continue
        seen_sources.add(source)
        item = dict(item)
        item["evidence_role"] = "source_diverse"
        _add_unique_evidence(result, seen_titles, item, limit=limit)

    for item in candidates:
        _add_unique_evidence(result, seen_titles, item, limit=limit)
    return result


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
    topic_profiles: dict[str, dict[str, Any]] = {}
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
        title_evidence = _build_member_title_evidence(
            canonical_title=event.canonical_title,
            event_time_anchor=event.event_time_anchor,
            assignments=event_assignments,
            limit=title_limit,
        )

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
                member_titles_sample=_unique_nonempty((item.get("title") for item in title_evidence), limit=title_limit)
                or title_sample,
                member_title_evidence=title_evidence,
                articles=articles,
                semantic_override_edge_count=int(graph_info.get("semantic_override_edge_count") or 0),
                graph_edge_count=int(graph_info.get("graph_edge_count") or 0),
                risk_flags=list(event.risk_flags),
                quality_summary=_quality_summary(event.quality_metrics),
                topic_profile=topic_profiles.setdefault(event.topic, build_topic_profile(event.topic)),
            )
        )

    return cards
