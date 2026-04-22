"""Data models for the LLM-backed timeline reasoning layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class EventCard:
    """Compact event-level input passed through rules and, when needed, the LLM."""

    discovery_run_id: str
    topic: str
    event_id: str
    canonical_title: str | None = None
    cluster_size: int = 0
    source_count: int = 0
    confidence: float = 0.0
    system_is_noise: bool = False
    noise_reason: str | None = None
    event_time_start: str | None = None
    event_time_end: str | None = None
    event_time_anchor: str | None = None
    member_news_ids: list[int | str] = field(default_factory=list)
    member_titles_sample: list[str] = field(default_factory=list)
    articles: list[dict[str, Any]] = field(default_factory=list)
    semantic_override_edge_count: int = 0
    graph_edge_count: int = 0
    risk_flags: list[str] = field(default_factory=list)

    def to_llm_dict(self) -> dict[str, Any]:
        """Return the bounded payload intended for LLM decisions."""
        return {
            "event_id": self.event_id,
            "topic": self.topic,
            "canonical_title": self.canonical_title,
            "cluster_size": self.cluster_size,
            "source_count": self.source_count,
            "confidence": self.confidence,
            "system_is_noise": self.system_is_noise,
            "noise_reason": self.noise_reason,
            "event_time_start": self.event_time_start,
            "event_time_end": self.event_time_end,
            "event_time_anchor": self.event_time_anchor,
            "risk_flags": list(self.risk_flags),
            "member_titles_sample": list(self.member_titles_sample),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EventDecision:
    """Final event-level decision produced by rules or the LLM."""

    event_id: str
    decision_source: str
    keep_event: bool
    is_topic_relevant: bool
    final_is_noise: bool
    needs_split: bool = False
    needs_merge: bool = False
    display_title: str | None = None
    resolved_time_start: str | None = None
    resolved_time_end: str | None = None
    resolved_time_anchor: str | None = None
    decision_confidence: float = 0.0
    time_confidence: float = 0.0
    decision_reason: str | None = None
    raw_response_json: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TimelineRecord:
    """A fully materialized timeline node with its source articles attached."""

    reasoning_run_id: str
    discovery_run_id: str
    topic: str
    event_id: str
    order_index: int
    canonical_title: str | None
    display_title: str | None
    event_time_start: str | None
    event_time_end: str | None
    event_time_anchor: str | None
    resolved_time_start: str | None
    resolved_time_end: str | None
    resolved_time_anchor: str | None
    display_date: str | None
    cluster_size: int
    source_count: int
    member_news_ids: list[int | str]
    confidence: float
    system_is_noise: bool
    noise_reason: str | None
    decision_source: str
    keep_event: bool
    is_topic_relevant: bool
    final_is_noise: bool
    needs_split: bool
    needs_merge: bool
    decision_confidence: float
    time_confidence: float
    decision_reason: str | None
    risk_flags: list[str] = field(default_factory=list)
    articles: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TimelineReasoningResult:
    """Top-level result returned by the formal timeline reasoning pipeline."""

    topic: str
    discovery_run_id: str
    reasoning_run_id: str
    model_name: str
    mode: str
    prompt_version: str
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(sep=" ", timespec="seconds"))
    input_event_count: int = 0
    review_event_count: int = 0
    accepted_event_count: int = 0
    rejected_event_count: int = 0
    status: str = "completed"
    timeline: list[TimelineRecord] = field(default_factory=list)
    decisions: list[EventDecision] = field(default_factory=list)
    decision_contexts: dict[str, dict[str, Any]] = field(default_factory=dict)
    output_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "discovery_run_id": self.discovery_run_id,
            "reasoning_run_id": self.reasoning_run_id,
            "model_name": self.model_name,
            "mode": self.mode,
            "prompt_version": self.prompt_version,
            "generated_at": self.generated_at,
            "status": self.status,
            "summary": {
                "input_event_count": self.input_event_count,
                "review_event_count": self.review_event_count,
                "accepted_event_count": self.accepted_event_count,
                "rejected_event_count": self.rejected_event_count,
            },
            "timeline": [record.to_dict() for record in self.timeline],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "decision_contexts": dict(self.decision_contexts),
            "output_paths": dict(self.output_paths),
        }
