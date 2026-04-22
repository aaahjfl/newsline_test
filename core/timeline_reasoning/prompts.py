"""Prompt templates for generic LLM timeline decisions."""

from __future__ import annotations

import json

from .models import EventCard


PROMPT_VERSION = "timeline_reasoning_v1"


def build_event_decision_prompt(cards: list[EventCard]) -> str:
    """Build a topic-agnostic prompt for event validity and time decisions."""
    payload = [card.to_llm_dict() for card in cards]
    return (
        "You are a generic news timeline decision layer. Judge candidate event clusters "
        "using only the structured input below.\n\n"
        "Task:\n"
        "1. Decide whether each candidate is a concrete real-world event suitable for a timeline.\n"
        "2. Decide whether it is relevant to the provided topic string.\n"
        "3. Decide whether it should be treated as final noise.\n"
        "4. Choose the most reliable time fields from the provided time fields. If the input does "
        "not support a precise date or time, return null instead of inventing one.\n"
        "5. Produce a concise display title. Prefer preserving the canonical title unless a shorter "
        "neutral title is obvious from the input.\n\n"
        "Important constraints:\n"
        "- This prompt must be applied to any topic; do not use topic-specific special cases.\n"
        "- system_is_noise is an upstream reference signal, not a final verdict. You may overturn it.\n"
        "- Do not add facts, dates, sources, or entities that are not supported by the input.\n"
        "- Do not automatically reject single-source or singleton events if they appear concrete and relevant.\n"
        "- Mark needs_split=true for broad rolling coverage or mixed events, but do not split them yourself.\n"
        "- Mark needs_merge=true only when the candidate clearly appears duplicated by another input item.\n"
        "- Return strict JSON only. No markdown, no comments, no extra prose.\n\n"
        "Return this exact shape:\n"
        "{\n"
        '  "decisions": [\n'
        "    {\n"
        '      "event_id": "string",\n'
        '      "keep_event": true,\n'
        '      "is_topic_relevant": true,\n'
        '      "final_is_noise": false,\n'
        '      "needs_split": false,\n'
        '      "needs_merge": false,\n'
        '      "display_title": "string or null",\n'
        '      "resolved_time_start": "YYYY-MM-DD HH:MM:SS or null",\n'
        '      "resolved_time_end": "YYYY-MM-DD HH:MM:SS or null",\n'
        '      "resolved_time_anchor": "YYYY-MM-DD HH:MM:SS or null",\n'
        '      "decision_confidence": 0.0,\n'
        '      "time_confidence": 0.0,\n'
        '      "decision_reason": "short reason"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Candidate event cards:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
