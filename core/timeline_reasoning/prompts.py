"""Prompt templates for generic LLM timeline decisions."""

from __future__ import annotations

import json

from .models import EventCard


PROMPT_VERSION = "timeline_reasoning_v7"


def build_event_decision_prompt(cards: list[EventCard]) -> str:
    """Build a topic-agnostic prompt for event validity and time decisions."""
    payload = [card.to_llm_dict() for card in cards]
    return (
        "You are a generic news timeline decision layer. Judge candidate event clusters "
        "using only the structured input below.\n\n"
        "Task:\n"
        "1. Decide whether each candidate is a concrete real-world event suitable for a timeline.\n"
        "2. Decide whether it is relevant to the provided topic and topic_profile. Do not rely on "
        "literal string overlap alone; use the title evidence to judge the intended entity, word sense, "
        "or subject.\n"
        "3. Decide whether it should be treated as final noise.\n"
        "4. Choose the most reliable timeline anchor date from the provided time fields. If the input "
        "does not support a precise date, return null instead of inventing one.\n"
        "5. Optionally produce a better display title when the canonical title is unsuitable, noisy, "
        "too broad, or not neutral enough for a timeline.\n\n"
        "Important constraints:\n"
        "- This prompt must be applied to any topic; do not use topic-specific special cases.\n"
        "- system_noise is an upstream reference signal, not a final verdict. You may overturn it.\n"
        "- Do not add facts, dates, sources, or entities that are not supported by the input.\n"
        "- Do not automatically reject single-source or singleton events if they appear concrete and relevant.\n"
        "- topic_profile is a relevance hint. If strict_entity=true, a candidate is "
        "topic-relevant when the named entity is a clear actor, speaker, decision-maker, target, "
        "counterparty, subject of a concrete claim, or otherwise materially involved in the event. "
        "The topic does not need to be the only actor or the only entity in the title.\n"
        "- Reject candidates only when the title evidence uses the surface word or a translated alias "
        "in a clearly different common-noun, cultural, product-category, generic analogy, pure category "
        "label, or unrelated sense.\n"
        "- A translated alias is not by itself a problem. If the translated name clearly refers to the "
        "same person, organization, brand, place, or product as the topic, treat it as relevant.\n"
        "- system_noise, low_confidence, low_source_support, and translated_topic_alias_risk are "
        "diagnostic signals only. They are not sufficient to reject a concrete topic-relevant event.\n"
        "- If a whole cluster is about the wrong topic sense, set is_topic_relevant=false and "
        "final_is_noise=true.\n"
        "- Use quality_hints and risk_flags as diagnostic evidence, not as automatic verdicts.\n"
        "- time.start, time.end, and time.anchor are date-level inputs. Return resolved_time_anchor as "
        "YYYY-MM-DD 00:00:00 when a date is supported, or null when unresolved.\n"
        "- Return display_title only when it improves or corrects the input title. Otherwise return null.\n"
        "- display_title must be concise, neutral, factual, and supported by title/evidence only.\n"
        "- Low source support alone is not enough to reject a concrete event.\n"
        "- Return strict JSON only. No markdown, no comments, no extra prose.\n\n"
        "Return this exact shape:\n"
        "{\n"
        '  "decisions": [\n'
        "    {\n"
        '      "event_id": "string",\n'
        '      "keep_event": true,\n'
        '      "is_topic_relevant": true,\n'
        '      "final_is_noise": false,\n'
        '      "display_title": "string or null",\n'
        '      "resolved_time_anchor": "YYYY-MM-DD HH:MM:SS or null",\n'
        '      "decision_reason": "very short reason"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Candidate event cards:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
