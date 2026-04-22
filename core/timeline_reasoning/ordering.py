"""Final deterministic ordering for timeline records."""

from __future__ import annotations

from datetime import datetime

from .models import EventCard, EventDecision, TimelineRecord


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


def _display_date(*values: str | None) -> str | None:
    for value in values:
        parsed = _parse_datetime(value)
        if parsed:
            return parsed.strftime("%Y-%m-%d")
    return None


def _sort_key(card: EventCard, decision: EventDecision) -> tuple[datetime, str]:
    parsed = (
        _parse_datetime(decision.resolved_time_anchor)
        or _parse_datetime(decision.resolved_time_start)
        or _parse_datetime(decision.resolved_time_end)
        or _parse_datetime(card.event_time_anchor)
        or _parse_datetime(card.event_time_start)
        or _parse_datetime(card.event_time_end)
        or datetime.max
    )
    return parsed, card.event_id


def build_timeline_records(
    *,
    reasoning_run_id: str,
    cards: list[EventCard],
    decisions: list[EventDecision],
) -> list[TimelineRecord]:
    """Attach decisions to cards and return kept events in final order."""
    cards_by_id = {card.event_id: card for card in cards}
    kept_pairs = [
        (cards_by_id[decision.event_id], decision)
        for decision in decisions
        if decision.keep_event and decision.event_id in cards_by_id
    ]
    kept_pairs.sort(key=lambda pair: _sort_key(pair[0], pair[1]))

    records: list[TimelineRecord] = []
    for index, (card, decision) in enumerate(kept_pairs, start=1):
        records.append(
            TimelineRecord(
                reasoning_run_id=reasoning_run_id,
                discovery_run_id=card.discovery_run_id,
                topic=card.topic,
                event_id=card.event_id,
                order_index=index,
                canonical_title=card.canonical_title,
                display_title=decision.display_title or card.canonical_title,
                event_time_start=card.event_time_start,
                event_time_end=card.event_time_end,
                event_time_anchor=card.event_time_anchor,
                resolved_time_start=decision.resolved_time_start,
                resolved_time_end=decision.resolved_time_end,
                resolved_time_anchor=decision.resolved_time_anchor,
                display_date=_display_date(
                    decision.resolved_time_anchor,
                    decision.resolved_time_start,
                    decision.resolved_time_end,
                    card.event_time_anchor,
                    card.event_time_start,
                    card.event_time_end,
                ),
                cluster_size=card.cluster_size,
                source_count=card.source_count,
                member_news_ids=list(card.member_news_ids),
                confidence=card.confidence,
                system_is_noise=card.system_is_noise,
                noise_reason=card.noise_reason,
                decision_source=decision.decision_source,
                keep_event=decision.keep_event,
                is_topic_relevant=decision.is_topic_relevant,
                final_is_noise=decision.final_is_noise,
                needs_split=decision.needs_split,
                needs_merge=decision.needs_merge,
                decision_confidence=decision.decision_confidence,
                time_confidence=decision.time_confidence,
                decision_reason=decision.decision_reason,
                risk_flags=list(card.risk_flags),
                articles=list(card.articles),
            )
        )
    return records
