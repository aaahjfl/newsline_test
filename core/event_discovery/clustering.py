"""Graph-link clustering for the formal event discovery pipeline."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import NamedTuple

import numpy as np

from core.schemas import EventCluster, EventEdge, NewsItem


SIMILARITY_THRESHOLD = 0.80
TIME_WINDOW_DAYS = 30.0
OVERRIDE_SIMILARITY_THRESHOLD = 0.92
OVERSIZED_COMPONENT_LIMIT = 120
COHESION_REFINEMENT_MIN_SIZE = 6
MIN_COMPONENT_EDGE_DENSITY = 0.35
MIN_COMPONENT_AVG_SIMILARITY = 0.84
REFINEMENT_STEP = 0.03
MAX_REFINEMENT_THRESHOLD = 0.95


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


def _time_gap_days(left: NewsItem, right: NewsItem) -> float | None:
    left_dt = _parse_iso_datetime(left.event_time_anchor)
    right_dt = _parse_iso_datetime(right.event_time_anchor)
    if left_dt is None or right_dt is None:
        return None
    return abs((left_dt - right_dt).total_seconds()) / 86400.0


def _mean_upper_triangle(matrix: np.ndarray) -> float | None:
    if matrix.shape[0] <= 1:
        return None
    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    if upper.size == 0:
        return None
    return float(np.mean(upper))


def _component_time_consistency(news_items: list[NewsItem]) -> float | None:
    anchors = [_parse_iso_datetime(item.event_time_anchor) for item in news_items]
    anchors = [anchor for anchor in anchors if anchor is not None]
    if len(anchors) <= 1:
        return None

    sorted_anchors = sorted(anchors)
    span_days = abs((sorted_anchors[-1] - sorted_anchors[0]).total_seconds()) / 86400.0
    return max(0.0, 1.0 - min(span_days / TIME_WINDOW_DAYS, 1.0))


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        if self.parent[index] != index:
            self.parent[index] = self.find(self.parent[index])
        return self.parent[index]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return

        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
            return
        if self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
            return

        self.parent[root_right] = root_left
        self.rank[root_left] += 1


class _EdgeCandidate(NamedTuple):
    left: int
    right: int
    similarity: float
    gap_days: float | None
    edge_reason: str


def _edge_candidate(
    left: int,
    right: int,
    news_items: list[NewsItem],
    similarity_matrix: np.ndarray,
    threshold: float,
) -> _EdgeCandidate | None:
    similarity = float(similarity_matrix[left, right])
    if similarity < threshold:
        return None

    gap_days = _time_gap_days(news_items[left], news_items[right])
    if gap_days is not None and gap_days > TIME_WINDOW_DAYS and similarity < OVERRIDE_SIMILARITY_THRESHOLD:
        return None

    if gap_days is None:
        edge_reason = "semantic_only"
    elif gap_days <= TIME_WINDOW_DAYS:
        edge_reason = "semantic_and_time"
    else:
        edge_reason = "semantic_override"

    return _EdgeCandidate(left, right, similarity, gap_days, edge_reason)


def _build_component_partition(
    member_indices: list[int],
    news_items: list[NewsItem],
    similarity_matrix: np.ndarray,
    threshold: float,
) -> tuple[list[list[int]], list[_EdgeCandidate]]:
    local_union_find = _UnionFind(len(member_indices))
    local_edges: list[_EdgeCandidate] = []

    for local_left, global_left in enumerate(member_indices):
        for local_right in range(local_left + 1, len(member_indices)):
            global_right = member_indices[local_right]
            edge = _edge_candidate(global_left, global_right, news_items, similarity_matrix, threshold)
            if edge is None:
                continue
            local_union_find.union(local_left, local_right)
            local_edges.append(edge)

    local_groups: dict[int, list[int]] = defaultdict(list)
    for local_index, global_index in enumerate(member_indices):
        local_groups[local_union_find.find(local_index)].append(global_index)

    return [sorted(group) for group in local_groups.values()], local_edges


def _component_edge_density(member_indices: list[int], group_edges: list[_EdgeCandidate]) -> float:
    size = len(member_indices)
    if size <= 1:
        return 1.0

    possible_edges = size * (size - 1) / 2
    return len(group_edges) / possible_edges if possible_edges else 1.0


def _component_average_similarity(member_indices: list[int], similarity_matrix: np.ndarray) -> float:
    if len(member_indices) <= 1:
        return 1.0
    submatrix = similarity_matrix[np.ix_(member_indices, member_indices)]
    mean_value = _mean_upper_triangle(submatrix)
    return 1.0 if mean_value is None else mean_value


def _should_refine_component(
    member_indices: list[int],
    group_edges: list[_EdgeCandidate],
    similarity_matrix: np.ndarray,
    threshold: float,
) -> bool:
    if threshold >= MAX_REFINEMENT_THRESHOLD:
        return False

    size = len(member_indices)
    if size > OVERSIZED_COMPONENT_LIMIT:
        return True

    if size < COHESION_REFINEMENT_MIN_SIZE:
        return False

    edge_density = _component_edge_density(member_indices, group_edges)
    average_similarity = _component_average_similarity(member_indices, similarity_matrix)
    return edge_density < MIN_COMPONENT_EDGE_DENSITY or average_similarity < MIN_COMPONENT_AVG_SIMILARITY


def _refine_components(
    member_indices: list[int],
    news_items: list[NewsItem],
    similarity_matrix: np.ndarray,
    threshold: float,
) -> tuple[list[list[int]], list[_EdgeCandidate]]:
    groups, edges = _build_component_partition(member_indices, news_items, similarity_matrix, threshold)

    refined_groups: list[list[int]] = []
    refined_edges: list[_EdgeCandidate] = []
    for group in groups:
        group_set = set(group)
        group_edges = [
            edge
            for edge in edges
            if edge.left in group_set and edge.right in group_set
        ]

        if _should_refine_component(group, group_edges, similarity_matrix, threshold):
            next_threshold = min(MAX_REFINEMENT_THRESHOLD, threshold + REFINEMENT_STEP)
            nested_groups, nested_edges = _refine_components(group, news_items, similarity_matrix, next_threshold)
            refined_groups.extend(nested_groups)
            refined_edges.extend(nested_edges)
            continue

        refined_groups.append(group)
        refined_edges.extend(group_edges)

    return refined_groups, refined_edges


def cluster_embeddings(
    news_items: list[NewsItem],
    embeddings: np.ndarray,
    *,
    topic: str,
) -> tuple[list[EventCluster], list[EventEdge], np.ndarray]:
    """Build a similarity graph and return connected components."""
    count = len(news_items)
    if count == 0:
        return [], [], np.empty((0, 0), dtype=np.float32)

    if embeddings.shape[0] != count:
        raise ValueError("Embedding count does not match news item count.")

    similarity_matrix = np.clip(np.matmul(embeddings, embeddings.T), -1.0, 1.0).astype(np.float32)
    grouped_indices, edge_candidates = _refine_components(
        list(range(count)),
        news_items,
        similarity_matrix,
        SIMILARITY_THRESHOLD,
    )
    edges = [
        EventEdge(
            left_index=edge.left,
            right_index=edge.right,
            left_news_id=news_items[edge.left].news_id,
            right_news_id=news_items[edge.right].news_id,
            similarity=round(edge.similarity, 6),
            time_gap_days=None if edge.gap_days is None else round(edge.gap_days, 3),
            edge_reason=edge.edge_reason,
        )
        for edge in edge_candidates
    ]

    clusters: list[EventCluster] = []
    for cluster_index, member_indices in enumerate(grouped_indices):
        sorted_indices = sorted(member_indices)
        submatrix = similarity_matrix[np.ix_(sorted_indices, sorted_indices)]
        cluster_items = [news_items[index] for index in sorted_indices]
        clusters.append(
            EventCluster(
                event_id=f"cluster_{cluster_index:03d}",
                topic=topic,
                member_indices=sorted_indices,
                member_news_ids=[item.news_id for item in cluster_items],
                cluster_size=len(sorted_indices),
                average_similarity=_mean_upper_triangle(submatrix),
                time_consistency=_component_time_consistency(cluster_items),
            )
        )

    clusters.sort(
        key=lambda cluster: (
            -cluster.cluster_size,
            str(cluster.member_news_ids[0]) if cluster.member_news_ids else "",
        )
    )
    return clusters, edges, similarity_matrix
