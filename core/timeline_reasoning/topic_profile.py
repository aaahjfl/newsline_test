"""Lightweight topic intent hints for timeline reasoning."""

from __future__ import annotations

import re
from typing import Any


LATIN_RE = re.compile(r"[A-Za-z]")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _latin_tokens(topic: str) -> list[str]:
    return TOKEN_RE.findall(topic)


def _surface_form_type(topic: str, tokens: list[str]) -> str:
    if not topic.strip():
        return "empty"
    if not LATIN_RE.search(topic):
        return "non_latin_or_translated"
    if len(tokens) == 1 and tokens[0].isupper() and len(tokens[0]) <= 8:
        return "acronym_or_code"
    if any(any(ch.islower() for ch in token) and any(ch.isupper() for ch in token[1:]) for token in tokens):
        return "mixed_case_named_entity"
    if tokens and all(token[:1].isupper() for token in tokens):
        return "proper_noun_like"
    return "common_phrase_or_keyword"


def build_topic_profile(topic: str) -> dict[str, Any]:
    """Build generic topic-disambiguation hints without topic-specific rules."""
    normalized = " ".join(str(topic or "").strip().split())
    tokens = _latin_tokens(normalized)
    surface_type = _surface_form_type(normalized, tokens)
    token_count = len(tokens) if tokens else (1 if normalized else 0)
    single_surface = token_count == 1
    strict_named_entity = surface_type in {
        "acronym_or_code",
        "mixed_case_named_entity",
        "proper_noun_like",
    }
    ambiguity_level = "high" if single_surface else "medium" if strict_named_entity else "low"

    if strict_named_entity:
        guidance = (
            "Treat the topic as a likely named entity, brand, organization, person, place, product, "
            "or code-like surface form. A translated alias can be relevant when it clearly refers to "
            "the same entity. Reject evidence only when it clearly uses another word sense, a common "
            "noun meaning, a generic analogy, or an unrelated cultural phrase."
        )
    else:
        guidance = (
            "Treat the topic as a literal user query. Require the candidate event to match the intended "
            "subject described by the topic, not merely a broad or accidental lexical overlap."
        )

    return {
        "topic": normalized,
        "surface_form_type": surface_type,
        "token_count": token_count,
        "single_surface_form": single_surface,
        "ambiguity_level": ambiguity_level,
        "strict_named_entity_relevance": strict_named_entity,
        "relevance_guidance": guidance,
    }


def title_contains_topic_surface(topic: str, title: str | None) -> bool:
    """Return whether the original topic surface appears in a title-like string."""
    text = str(title or "")
    normalized_topic = " ".join(str(topic or "").strip().split())
    if not normalized_topic:
        return False
    if LATIN_RE.search(normalized_topic):
        pattern = r"(?<![A-Za-z0-9])" + re.escape(normalized_topic) + r"(?![A-Za-z0-9])"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    return normalized_topic in text
