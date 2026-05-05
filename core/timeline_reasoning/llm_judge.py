"""LLM-backed event decisions for timeline reasoning."""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from configs.model_config import MODEL_CONFIG
from configs.pipeline_config import PIPELINE_CONFIG
from core.llm.ollama_client import OllamaRequestError, check_ollama_available, generate_with_ollama

from .models import EventCard, EventDecision
from .prompts import build_event_decision_prompt


def _clamp_float(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", "null", "none", ""}:
            return False
    return default


def _payload_value_or_card(payload: dict[str, Any], key: str, fallback: str | None) -> str | None:
    if key in payload:
        value = payload.get(key)
        return str(value).strip() if value not in (None, "") else None
    return fallback


def _derived_decision_confidence(card: EventCard, keep_event: bool) -> float:
    base = _clamp_float(card.confidence, default=0.75)
    if keep_event:
        return max(0.5, min(0.9, base))
    return 0.75


def _derived_time_confidence(resolved_time_anchor: str | None) -> float:
    return 0.8 if resolved_time_anchor else 0.0


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"<think>.*?</think>", "", stripped, flags=re.IGNORECASE | re.DOTALL).strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def _decision_from_payload(payload: dict[str, Any], card: EventCard, raw: dict[str, Any]) -> EventDecision:
    keep_event = _as_bool(payload.get("keep_event"), default=True)
    is_topic_relevant = _as_bool(payload.get("is_topic_relevant"), default=keep_event)
    final_is_noise = _as_bool(payload.get("final_is_noise"), default=not keep_event)
    if final_is_noise or not is_topic_relevant:
        keep_event = False

    resolved_time_anchor = _payload_value_or_card(payload, "resolved_time_anchor", card.event_time_anchor)

    return EventDecision(
        event_id=card.event_id,
        decision_source="llm",
        keep_event=keep_event,
        is_topic_relevant=is_topic_relevant,
        final_is_noise=final_is_noise,
        needs_split=False,
        needs_merge=False,
        split_reason=None,
        merge_reason=None,
        display_title=payload.get("display_title") or card.canonical_title,
        resolved_time_start=card.event_time_start,
        resolved_time_end=card.event_time_end,
        resolved_time_anchor=resolved_time_anchor,
        decision_confidence=_derived_decision_confidence(card, keep_event),
        time_confidence=_derived_time_confidence(resolved_time_anchor),
        decision_reason=payload.get("decision_reason") or "LLM decision returned without a detailed reason.",
        raw_response_json=raw,
    )


def _fallback_decision(card: EventCard, reason: str) -> EventDecision:
    """Return a conservative decision when the LLM cannot answer one card."""
    final_is_noise = bool(
        card.system_is_noise
        or "rolling_coverage" in card.risk_flags
        or "rolling_coverage_title" in card.risk_flags
        or "missing_canonical_title" in card.risk_flags
        or "empty_cluster" in card.risk_flags
    )
    keep_event = not final_is_noise
    return EventDecision(
        event_id=card.event_id,
        decision_source="llm_fallback",
        keep_event=keep_event,
        is_topic_relevant=True,
        final_is_noise=final_is_noise,
        needs_split=False,
        needs_merge=False,
        split_reason=None,
        merge_reason=None,
        display_title=card.canonical_title,
        resolved_time_start=card.event_time_start,
        resolved_time_end=card.event_time_end,
        resolved_time_anchor=card.event_time_anchor,
        decision_confidence=0.25,
        time_confidence=0.5 if card.event_time_anchor else 0.0,
        decision_reason=f"LLM fallback used because model request failed: {reason}",
        raw_response_json={"error": reason, "risk_flags": list(card.risk_flags)},
    )


def _judge_batch(
    batch: list[EventCard],
    *,
    model: str,
    url: str,
    timeout_seconds: int,
    progress_label: str,
) -> list[EventDecision]:
    prompt = build_event_decision_prompt(batch)
    print(
        f"[timeline-llm] {progress_label}: sending {len(batch)} event(s) to {model}; timeout={timeout_seconds}s",
        file=sys.stderr,
        flush=True,
    )
    try:
        response_text = generate_with_ollama(
            prompt,
            model=model,
            url=url,
            timeout_seconds=timeout_seconds,
            keep_alive=str(PIPELINE_CONFIG.get("topic_alias_ollama_keep_alive", "0s")),
            think=False,
            options={
                "num_ctx": 8192,
                "num_predict": 1024,
                "temperature": 0.0,
            },
        )
        raw = _extract_json_object(response_text)
    except (OllamaRequestError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        if len(batch) > 1:
            midpoint = len(batch) // 2
            return [
                *_judge_batch(
                    batch[:midpoint],
                    model=model,
                    url=url,
                    timeout_seconds=timeout_seconds,
                    progress_label=f"{progress_label}.split-a",
                ),
                *_judge_batch(
                    batch[midpoint:],
                    model=model,
                    url=url,
                    timeout_seconds=timeout_seconds,
                    progress_label=f"{progress_label}.split-b",
                ),
            ]
        return [_fallback_decision(batch[0], str(exc))]

    payload_by_id = {
        str(item.get("event_id")): item
        for item in raw.get("decisions", [])
        if isinstance(item, dict) and item.get("event_id") is not None
    }
    decisions: list[EventDecision] = []
    for card in batch:
        payload = payload_by_id.get(card.event_id)
        if payload is None:
            payload = {
                "event_id": card.event_id,
                "keep_event": False,
                "is_topic_relevant": False,
                "final_is_noise": True,
                "decision_reason": "LLM response omitted this event_id.",
            }
        decisions.append(_decision_from_payload(payload, card, raw))
    return decisions


def judge_event_cards_with_llm(
    cards: list[EventCard],
    *,
    model_name: str | None = None,
    batch_size: int = 4,
    timeout_seconds: int = 300,
) -> list[EventDecision]:
    """Ask Ollama to judge event cards in bounded batches."""
    if not cards:
        return []

    model = model_name or str(MODEL_CONFIG.get("reasoning_model", "qwen3.5:9b"))
    batch_size = max(1, int(batch_size or 1))
    url = str(
        PIPELINE_CONFIG.get(
            "timeline_reasoning_ollama_url",
            PIPELINE_CONFIG.get("topic_alias_ollama_url", "http://127.0.0.1:11434/api/generate"),
        )
    )
    try:
        check_ollama_available(url, timeout_seconds=min(10, max(1, timeout_seconds)))
    except OllamaRequestError as exc:
        print(f"[timeline-llm] {exc}", file=sys.stderr, flush=True)
        return [_fallback_decision(card, str(exc)) for card in cards]

    decisions: list[EventDecision] = []
    total_batches = (len(cards) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(cards), batch_size), start=1):
        batch = cards[start : start + batch_size]
        decisions.extend(
            _judge_batch(
                batch,
                model=model,
                url=url,
                timeout_seconds=timeout_seconds,
                progress_label=f"batch {batch_index}/{total_batches}",
            )
        )

    return decisions
