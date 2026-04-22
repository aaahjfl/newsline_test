"""Convert graph clusters into standardized event objects."""

from __future__ import annotations

from datetime import datetime
from math import log2
import re

import numpy as np

from core.schemas import EventCluster, EventNode, NewsItem


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


def _confidence_score(cluster: EventCluster) -> float:
    if cluster.cluster_size <= 1:
        average_similarity = 0.7
    else:
        average_similarity = cluster.average_similarity if cluster.average_similarity is not None else 0.6

    size_score = min(1.0, log2(cluster.cluster_size + 1) / log2(6))
    time_score = cluster.time_consistency if cluster.time_consistency is not None else 0.75

    confidence = (0.55 * average_similarity) + (0.25 * size_score) + (0.20 * time_score)
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
        representative_index = _choose_representative_index(member_indices, similarity_matrix)
        representative_item = news_items[representative_index]

        event_node = EventNode(
            event_id=cluster.event_id,
            topic=topic,
            member_news_ids=[item.news_id for item in cluster_items],
            cluster_size=cluster.cluster_size,
            canonical_title=representative_item.title,
            representative_news_id=representative_item.news_id,
            event_time_start=_time_range_start(cluster_items),
            event_time_end=_time_range_end(cluster_items),
            event_time_anchor=_median_anchor(cluster_items),
            source_count=len({item.source.strip() for item in cluster_items if item.source and item.source.strip()}),
            confidence=_confidence_score(cluster),
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
                "system_is_noise": event_node.system_is_noise,
                "noise_reason": event_node.noise_reason,
            }
            for item in cluster_items
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
