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
ROLLING_TIME_WINDOW_DAYS = 3.0
ROLLING_OVERRIDE_SIMILARITY_THRESHOLD = 0.97
SMALL_CLUSTER_MERGE_SOURCE_MAX_SIZE = 2
SMALL_CLUSTER_MERGE_RESULT_MAX_SIZE = 5
SMALL_CLUSTER_MERGE_TIME_WINDOW_DAYS = 7.0
SMALL_CLUSTER_MERGE_AVG_SIMILARITY = 0.86
SMALL_CLUSTER_MERGE_MAX_SIMILARITY = 0.90
SMALL_CLUSTER_MERGE_MISSING_TIME_AVG_SIMILARITY = 0.90
SMALL_CLUSTER_MERGE_MISSING_TIME_MAX_SIMILARITY = 0.94
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


def _risk_flags(item: NewsItem) -> set[str]:
    values = item.metadata.get("title_risk_flags") if item.metadata else None
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values}


def _edge_time_policy(left: NewsItem, right: NewsItem) -> tuple[float, float]:
    flags = _risk_flags(left).union(_risk_flags(right))
    if "rolling_coverage" in flags:
        return ROLLING_TIME_WINDOW_DAYS, ROLLING_OVERRIDE_SIMILARITY_THRESHOLD
    return TIME_WINDOW_DAYS, OVERRIDE_SIMILARITY_THRESHOLD


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


def _component_time_span_days(member_indices: list[int], news_items: list[NewsItem]) -> float | None:
    anchors = [_parse_iso_datetime(news_items[index].event_time_anchor) for index in member_indices]
    anchors = [anchor for anchor in anchors if anchor is not None]
    if len(anchors) <= 1:
        return None

    sorted_anchors = sorted(anchors)
    return abs((sorted_anchors[-1] - sorted_anchors[0]).total_seconds()) / 86400.0


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


class _MergeCandidate(NamedTuple):
    left_component: int
    right_component: int
    average_similarity: float
    max_similarity: float
    best_left: int
    best_right: int
    gap_days: float | None


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
    time_window_days, override_threshold = _edge_time_policy(news_items[left], news_items[right])
    if gap_days is not None and gap_days > time_window_days and similarity < override_threshold:
        return None

    if gap_days is None:
        edge_reason = "semantic_only"
    elif gap_days <= time_window_days:
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


def _component_contains_risk(member_indices: list[int], news_items: list[NewsItem], risk_flag: str) -> bool:
    return any(risk_flag in _risk_flags(news_items[index]) for index in member_indices)


def _merge_candidate(
    left_component: int,
    right_component: int,
    left_indices: list[int],
    right_indices: list[int],
    news_items: list[NewsItem],
    similarity_matrix: np.ndarray,
) -> _MergeCandidate | None:
    if len(left_indices) + len(right_indices) > SMALL_CLUSTER_MERGE_RESULT_MAX_SIZE:
        return None
    if _component_contains_risk(left_indices, news_items, "rolling_coverage"):
        return None
    if _component_contains_risk(right_indices, news_items, "rolling_coverage"):
        return None

    cross = similarity_matrix[np.ix_(left_indices, right_indices)]
    average_similarity = float(np.mean(cross))
    best_local_index = int(np.argmax(cross))
    left_local, right_local = np.unravel_index(best_local_index, cross.shape)
    max_similarity = float(cross[left_local, right_local])

    combined_indices = [*left_indices, *right_indices]
    combined_span = _component_time_span_days(combined_indices, news_items)
    if combined_span is None:
        if (
            average_similarity < SMALL_CLUSTER_MERGE_MISSING_TIME_AVG_SIMILARITY
            or max_similarity < SMALL_CLUSTER_MERGE_MISSING_TIME_MAX_SIMILARITY
        ):
            return None
    elif combined_span > SMALL_CLUSTER_MERGE_TIME_WINDOW_DAYS:
        return None
    elif average_similarity < SMALL_CLUSTER_MERGE_AVG_SIMILARITY or max_similarity < SMALL_CLUSTER_MERGE_MAX_SIMILARITY:
        return None

    best_left = left_indices[left_local]
    best_right = right_indices[right_local]
    return _MergeCandidate(
        left_component=left_component,
        right_component=right_component,
        average_similarity=average_similarity,
        max_similarity=max_similarity,
        best_left=best_left,
        best_right=best_right,
        gap_days=_time_gap_days(news_items[best_left], news_items[best_right]),
    )


def _merge_small_components(
    grouped_indices: list[list[int]],
    news_items: list[NewsItem],
    similarity_matrix: np.ndarray,
) -> tuple[list[list[int]], list[_EdgeCandidate]]:
    """Conservatively merge tiny over-split components into compact events."""
    if len(grouped_indices) <= 1:
        return grouped_indices, []

    eligible_components = [
        component_index
        for component_index, group in enumerate(grouped_indices)
        if len(group) <= SMALL_CLUSTER_MERGE_SOURCE_MAX_SIZE
    ]
    if len(eligible_components) <= 1:
        return grouped_indices, []

    candidates: list[_MergeCandidate] = []
    for left_offset, left_component in enumerate(eligible_components):
        for right_component in eligible_components[left_offset + 1 :]:
            candidate = _merge_candidate(
                left_component,
                right_component,
                grouped_indices[left_component],
                grouped_indices[right_component],
                news_items,
                similarity_matrix,
            )
            if candidate is not None:
                candidates.append(candidate)

    if not candidates:
        return grouped_indices, []

    candidates.sort(key=lambda item: (-item.average_similarity, -item.max_similarity, item.left_component, item.right_component))
    union_find = _UnionFind(len(grouped_indices))
    root_members = {index: list(group) for index, group in enumerate(grouped_indices)}
    merge_edges: list[_EdgeCandidate] = []

    for candidate in candidates:
        left_root = union_find.find(candidate.left_component)
        right_root = union_find.find(candidate.right_component)
        if left_root == right_root:
            continue

        left_indices = root_members[left_root]
        right_indices = root_members[right_root]
        refreshed = _merge_candidate(
            left_root,
            right_root,
            left_indices,
            right_indices,
            news_items,
            similarity_matrix,
        )
        if refreshed is None:
            continue

        union_find.union(left_root, right_root)
        merged_root = union_find.find(left_root)
        retired_root = right_root if merged_root == left_root else left_root
        root_members[merged_root] = sorted([*left_indices, *right_indices])
        root_members.pop(retired_root, None)
        merge_edges.append(
            _EdgeCandidate(
                refreshed.best_left,
                refreshed.best_right,
                refreshed.max_similarity,
                refreshed.gap_days,
                "small_cluster_merge",
            )
        )

    merged_groups: dict[int, list[int]] = defaultdict(list)
    for component_index, group in enumerate(grouped_indices):
        merged_groups[union_find.find(component_index)].extend(group)

    return [sorted(group) for group in merged_groups.values()], merge_edges


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
    grouped_indices, merge_edges = _merge_small_components(grouped_indices, news_items, similarity_matrix)
    edge_candidates.extend(merge_edges)
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
        member_set = set(sorted_indices)
        cluster_edges = [
            edge
            for edge in edge_candidates
            if edge.left in member_set and edge.right in member_set
        ]
        clusters.append(
            EventCluster(
                event_id=f"cluster_{cluster_index:03d}",
                topic=topic,
                member_indices=sorted_indices,
                member_news_ids=[item.news_id for item in cluster_items],
                cluster_size=len(sorted_indices),
                average_similarity=_mean_upper_triangle(submatrix),
                time_consistency=_component_time_consistency(cluster_items),
                edge_density=_component_edge_density(sorted_indices, cluster_edges),
            )
        )

    clusters.sort(
        key=lambda cluster: (
            -cluster.cluster_size,
            str(cluster.member_news_ids[0]) if cluster.member_news_ids else "",
        )
    )
    return clusters, edges, similarity_matrix
