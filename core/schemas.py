"""Shared domain schemas used across the formal project structure."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class NewsItem:
    """Normalized news row used by the formal event discovery layer."""

    news_id: int | str
    title: str
    source: str | None = None
    url: str | None = None
    publish_time: str | None = None
    event_time_anchor: str | None = None
    event_time_start: str | None = None
    event_time_end: str | None = None
    time_granularity: str | None = None
    is_noise: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> int | str:
        """Compatibility alias for older code paths that still access `.id`."""
        return self.news_id

    @property
    def event_timestamp(self) -> str | None:
        """Compatibility alias for the legacy DB column name."""
        return self.event_time_anchor

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ParsedNews(NewsItem):
    """Extended news item used by the migrated time-parsing layer."""

    normalized_title: str | None = None
    entities: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EventEdge:
    """Graph edge retained for event discovery debugging output."""

    left_index: int
    right_index: int
    left_news_id: int | str
    right_news_id: int | str
    similarity: float
    time_gap_days: float | None = None
    edge_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EventCluster:
    """Internal connected component before final event serialization."""

    event_id: str
    topic: str
    member_indices: list[int] = field(default_factory=list)
    member_news_ids: list[int | str] = field(default_factory=list)
    cluster_size: int = 0
    average_similarity: float | None = None
    time_consistency: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EventNode:
    """Standardized event object exposed to downstream stages."""

    event_id: str
    topic: str
    member_news_ids: list[int | str] = field(default_factory=list)
    cluster_size: int = 0
    canonical_title: str | None = None
    representative_news_id: int | str | None = None
    event_time_start: str | None = None
    event_time_end: str | None = None
    event_time_anchor: str | None = None
    source_count: int = 0
    confidence: float = 0.0
    system_is_noise: bool = False
    noise_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EventDiscoveryResult:
    """Top-level result returned by `run_event_discovery`."""

    topic: str
    run_id: str
    topic_aliases: list[str] = field(default_factory=list)
    topic_alias_details: list[dict[str, Any]] = field(default_factory=list)
    candidate_count: int = 0
    filtered_count: int = 0
    events: list[EventNode] = field(default_factory=list)
    assignments: list[dict[str, Any]] = field(default_factory=list)
    graph_edges: list[dict[str, Any]] = field(default_factory=list)
    output_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "run_id": self.run_id,
            "topic_aliases": list(self.topic_aliases),
            "topic_alias_details": list(self.topic_alias_details),
            "candidate_count": self.candidate_count,
            "filtered_count": self.filtered_count,
            "events": [event.to_dict() for event in self.events],
            "assignments": list(self.assignments),
            "graph_edges": list(self.graph_edges),
            "output_paths": dict(self.output_paths),
        }


@dataclass(slots=True)
class TimelineNode:
    event_id: str
    order_index: int
    reasoning_note: str | None = None
