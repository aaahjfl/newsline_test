"""Convert graph clusters into standardized event objects."""

from __future__ import annotations

from datetime import datetime
from math import log2
import re

import numpy as np

from core.schemas import EventCluster, EventNode, NewsItem


LONG_TIME_SPAN_DAYS = 45.0


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


def _to_iso_text(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat(sep=" ", timespec="seconds")


def _sanitize_topic_token(topic: str) -> str:
    normalized = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", topic.strip(), flags=re.UNICODE)
    collapsed = re.sub(r"_+", "_", normalized).strip("_")
    return collapsed or "topic"


def _cluster_times(cluster_items: list[NewsItem], field_name: str) -> list[datetime]:
    values = [_parse_iso_datetime(getattr(item, field_name)) for item in cluster_items]
    return sorted(value for value in values if value is not None)


def _expanded_cluster_items(cluster_items: list[NewsItem]) -> list[NewsItem]:
    expanded: list[NewsItem] = []
    for item in cluster_items:
        duplicate_members = item.metadata.get("duplicate_members") if item.metadata else None
        if isinstance(duplicate_members, list) and duplicate_members:
            expanded.extend(member for member in duplicate_members if isinstance(member, NewsItem))
        else:
            expanded.append(item)
    return expanded


def _title_risk_flags(cluster_items: list[NewsItem]) -> list[str]:
    flags: list[str] = []
    seen: set[str] = set()
    for item in cluster_items:
        values = item.metadata.get("title_risk_flags") if item.metadata else None
        if not isinstance(values, list):
            continue
        for value in values:
            text = str(value)
            if text and text not in seen:
                seen.add(text)
                flags.append(text)
    return flags


def _time_span_days(cluster_items: list[NewsItem]) -> float | None:
    starts = _cluster_times(cluster_items, "event_time_start")
    ends = _cluster_times(cluster_items, "event_time_end")
    if not starts or not ends:
        return None
    return abs((ends[-1] - starts[0]).total_seconds()) / 86400.0


def _unique_normalized_title_count(cluster_items: list[NewsItem]) -> int:
    keys: set[str] = set()
    for item in cluster_items:
        key = ""
        if item.metadata:
            key = str(item.metadata.get("normalized_title") or "").strip()
        keys.add(key or item.title.strip().casefold())
    return len(keys)


def _choose_representative_index(member_indices: list[int], similarity_matrix: np.ndarray) -> int:
    if len(member_indices) == 1:
        return member_indices[0]

    submatrix = similarity_matrix[np.ix_(member_indices, member_indices)]
    mean_similarity = np.mean(submatrix, axis=1)
    best_local_index = int(np.argmax(mean_similarity))
    return member_indices[best_local_index]


def _median_anchor(cluster_items: list[NewsItem]) -> str | None:
    anchors = _cluster_times(cluster_items, "event_time_anchor")
    if not anchors:
        return None
    return _to_iso_text(anchors[len(anchors) // 2])


def _time_range_start(cluster_items: list[NewsItem]) -> str | None:
    starts = _cluster_times(cluster_items, "event_time_start")
    return _to_iso_text(starts[0]) if starts else None


def _time_range_end(cluster_items: list[NewsItem]) -> str | None:
    ends = _cluster_times(cluster_items, "event_time_end")
    return _to_iso_text(ends[-1]) if ends else None


def _quality_metrics(cluster: EventCluster, cluster_items: list[NewsItem], expanded_items: list[NewsItem]) -> dict[str, object]:
    unique_title_count = _unique_normalized_title_count(expanded_items)
    total_title_count = max(len(expanded_items), 1)
    duplicate_ratio = max(0.0, 1.0 - (unique_title_count / total_title_count))
    time_span_days = _time_span_days(expanded_items)
    semantic_cohesion = cluster.average_similarity
    temporal_coherence = cluster.time_consistency
    graph_density = cluster.edge_density
    support_score = min(1.0, log2(unique_title_count + 1) / log2(6))

    return {
        "semantic_cohesion": None if semantic_cohesion is None else round(float(semantic_cohesion), 4),
        "temporal_coherence": None if temporal_coherence is None else round(float(temporal_coherence), 4),
        "support_score": round(float(support_score), 4),
        "graph_density": None if graph_density is None else round(float(graph_density), 4),
        "duplicate_ratio": round(float(duplicate_ratio), 4),
        "unique_title_count": unique_title_count,
        "article_count": total_title_count,
        "time_span_days": None if time_span_days is None else round(float(time_span_days), 3),
        "clustered_title_count": len(cluster_items),
    }


def _risk_flags(cluster: EventCluster, quality_metrics: dict[str, object], title_flags: list[str]) -> list[str]:
    flags = list(title_flags)
    seen = set(flags)

    def add(flag: str) -> None:
        if flag not in seen:
            seen.add(flag)
            flags.append(flag)

    time_span_days = quality_metrics.get("time_span_days")
    if isinstance(time_span_days, (int, float)) and float(time_span_days) > LONG_TIME_SPAN_DAYS:
        add("long_time_span")

    duplicate_ratio = float(quality_metrics.get("duplicate_ratio") or 0.0)
    if duplicate_ratio >= 0.4 and int(quality_metrics.get("article_count") or 0) >= 3:
        add("high_duplicate_ratio")

    graph_density = quality_metrics.get("graph_density")
    if isinstance(graph_density, (int, float)) and cluster.cluster_size >= 6 and float(graph_density) < 0.2:
        add("low_graph_density")

    temporal_coherence = quality_metrics.get("temporal_coherence")
    if isinstance(temporal_coherence, (int, float)) and cluster.cluster_size > 1 and float(temporal_coherence) < 0.4:
        add("low_temporal_coherence")

    return flags


def _confidence_score(cluster: EventCluster, quality_metrics: dict[str, object], risk_flags: list[str]) -> float:
    semantic_score = quality_metrics.get("semantic_cohesion")
    if not isinstance(semantic_score, (int, float)):
        semantic_score = 0.55

    temporal_score = quality_metrics.get("temporal_coherence")
    if not isinstance(temporal_score, (int, float)):
        temporal_score = 0.75

    support_score = float(quality_metrics.get("support_score") or 0.0)
    density_score = quality_metrics.get("graph_density")
    if not isinstance(density_score, (int, float)):
        density_score = 0.5 if cluster.cluster_size <= 1 else 0.65

    confidence = (
        (0.45 * float(semantic_score))
        + (0.25 * float(temporal_score))
        + (0.15 * support_score)
        + (0.15 * float(density_score))
    )

    if "rolling_coverage" in risk_flags:
        confidence -= 0.08
    if "long_time_span" in risk_flags:
        confidence -= 0.10
    if "high_duplicate_ratio" in risk_flags:
        confidence -= 0.04
    if "low_graph_density" in risk_flags:
        confidence -= 0.05

    return round(max(0.0, min(confidence, 1.0)), 4)


def _noise_hint(cluster: EventCluster, confidence: float) -> tuple[bool, str | None]:
    """Return a conservative SBERT-side noise hint for downstream LLM validation."""
    if confidence < 0.55:
        return True, "low_cluster_confidence"
    return False, None


def build_event_nodes(
    topic: str,
    clusters: list[EventCluster],
    news_items: list[NewsItem],
    similarity_matrix: np.ndarray,
) -> tuple[list[EventNode], list[dict[str, object]]]:
    """Materialize standardized events and news-to-event assignments."""
    sortable_events: list[tuple[tuple[str, str, str], EventNode, list[dict[str, object]]]] = []

    for cluster in clusters:
        member_indices = sorted(cluster.member_indices)
        cluster_items = [news_items[index] for index in member_indices]
        expanded_items = _expanded_cluster_items(cluster_items)
        representative_index = _choose_representative_index(member_indices, similarity_matrix)
        representative_item = news_items[representative_index]
        quality_metrics = _quality_metrics(cluster, cluster_items, expanded_items)
        risk_flags = _risk_flags(cluster, quality_metrics, _title_risk_flags(expanded_items))
        confidence = _confidence_score(cluster, quality_metrics, risk_flags)

        event_node = EventNode(
            event_id=cluster.event_id,
            topic=topic,
            member_news_ids=[item.news_id for item in expanded_items],
            cluster_size=len(expanded_items),
            canonical_title=representative_item.title,
            representative_news_id=representative_item.news_id,
            event_time_start=_time_range_start(expanded_items),
            event_time_end=_time_range_end(expanded_items),
            event_time_anchor=_median_anchor(expanded_items),
            source_count=len({item.source.strip() for item in expanded_items if item.source and item.source.strip()}),
            confidence=confidence,
            risk_flags=risk_flags,
            quality_metrics=quality_metrics,
        )
        event_node.system_is_noise, event_node.noise_reason = _noise_hint(cluster, event_node.confidence)

        assignments = [
            {
                "news_id": item.news_id,
                "event_id": cluster.event_id,
                "title": item.title,
                "source": item.source,
                "url": item.url,
                "event_time_anchor": item.event_time_anchor,
                "normalized_title": item.metadata.get("normalized_title") if item.metadata else None,
                "title_risk_flags": item.metadata.get("title_risk_flags") if item.metadata else [],
                "system_is_noise": event_node.system_is_noise,
                "noise_reason": event_node.noise_reason,
            }
            for item in expanded_items
        ]

        sort_key = (
            event_node.event_time_anchor or event_node.event_time_start or event_node.event_time_end or "9999-12-31 23:59:59",
            str(event_node.representative_news_id),
            event_node.canonical_title or "",
        )
        sortable_events.append((sort_key, event_node, assignments))

    sortable_events.sort(key=lambda item: item[0])

    topic_token = _sanitize_topic_token(topic)
    events: list[EventNode] = []
    assignments: list[dict[str, object]] = []
    for index, (_, event_node, event_assignments) in enumerate(sortable_events, start=1):
        event_id = f"{topic_token}_event_{index:03d}"
        event_node.event_id = event_id
        events.append(event_node)

        for assignment in event_assignments:
            assignment["event_id"] = event_id
            assignment["cluster_size"] = event_node.cluster_size
            assignment["canonical_title"] = event_node.canonical_title
            assignment["system_is_noise"] = event_node.system_is_noise
            assignment["noise_reason"] = event_node.noise_reason
            assignments.append(assignment)

    assignments.sort(key=lambda item: (str(item["event_id"]), str(item["news_id"])))
    return events, assignments
