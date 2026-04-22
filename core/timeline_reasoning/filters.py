"""Rule-based routing and fallback decisions for timeline reasoning."""

from __future__ import annotations

from datetime import datetime
import re

from .models import EventCard, EventDecision


ROLLING_TITLE_RE = re.compile(
    r"\b(live|updates?|latest|timeline|breaking|rolling|as it happened)\b",
    flags=re.IGNORECASE,
)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:19] if fmt.endswith("%S") else text[:10], fmt)
        except ValueError:
            continue
    return None


def _time_span_days(card: EventCard) -> float | None:
    start = _parse_datetime(card.event_time_start)
    end = _parse_datetime(card.event_time_end)
    if not start or not end:
        return None
    return abs((end - start).total_seconds()) / 86400.0


def collect_risk_flags(card: EventCard) -> list[str]:
    """Collect generic risk flags used for routing and LLM context."""
    flags: list[str] = []
    title = card.canonical_title or ""
    span_days = _time_span_days(card)

    if not card.event_id:
        flags.append("missing_event_id")
    if not title.strip():
        flags.append("missing_canonical_title")
    if card.cluster_size <= 0:
        flags.append("empty_cluster")
    if card.system_is_noise:
        flags.append("system_noise")
    if card.confidence < 0.65:
        flags.append("low_confidence")
    elif card.confidence < 0.75:
        flags.append("medium_confidence")
    if not (card.event_time_anchor or card.event_time_start or card.event_time_end):
        flags.append("missing_time")
    if span_days is not None and span_days > 45:
        flags.append("long_time_span")
    if card.cluster_size >= 10:
        flags.append("large_cluster")
    if card.source_count <= 1 and card.confidence < 0.75:
        flags.append("low_source_support")
    if ROLLING_TITLE_RE.search(title):
        flags.append("rolling_coverage_title")
    if card.semantic_override_edge_count >= 2:
        flags.append("semantic_override_edges")

    return flags


def route_event_card(card: EventCard, *, mode: str = "standard") -> str:
    """Return `rule_reject`, `llm_review`, or `auto_accept`."""
    card.risk_flags = collect_risk_flags(card)
    fatal = {"missing_event_id", "missing_canonical_title", "empty_cluster"}
    if fatal.intersection(card.risk_flags):
        return "rule_reject"

    normalized_mode = mode.strip().casefold()
    if normalized_mode == "full":
        return "llm_review"

    fast_review_flags = {
        "system_noise",
        "low_confidence",
        "missing_time",
        "long_time_span",
        "rolling_coverage_title",
    }
    standard_review_flags = fast_review_flags.union(
        {
            "medium_confidence",
            "large_cluster",
            "low_source_support",
            "semantic_override_edges",
        }
    )

    review_flags = fast_review_flags if normalized_mode == "fast" else standard_review_flags
    if review_flags.intersection(card.risk_flags):
        return "llm_review"
    return "auto_accept"


def build_rule_decision(card: EventCard, *, route: str) -> EventDecision:
    """Build a deterministic decision for events not sent to the LLM."""
    keep_event = route != "rule_reject"
    title = card.canonical_title
    if keep_event:
        reason = "Accepted by rule-based routing; no high-risk flags required LLM review."
        decision_confidence = max(0.0, min(1.0, card.confidence or 0.75))
        final_is_noise = False
        is_topic_relevant = True
    else:
        reason = "Rejected by rule-based routing because required event fields are missing or empty."
        decision_confidence = 1.0
        final_is_noise = True
        is_topic_relevant = False

    return EventDecision(
        event_id=card.event_id,
        decision_source="rule",
        keep_event=keep_event,
        is_topic_relevant=is_topic_relevant,
        final_is_noise=final_is_noise,
        display_title=title,
        resolved_time_start=card.event_time_start,
        resolved_time_end=card.event_time_end,
        resolved_time_anchor=card.event_time_anchor,
        decision_confidence=decision_confidence,
        time_confidence=1.0 if card.event_time_anchor else 0.5,
        decision_reason=reason,
        raw_response_json={"route": route, "risk_flags": list(card.risk_flags)},
    )
