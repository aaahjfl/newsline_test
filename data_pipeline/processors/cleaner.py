"""Lightweight text cleaning helpers for the formal pipeline."""

import re


def clean_text(text: str | None) -> str:
    """Normalize whitespace without changing business semantics."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()
