"""Generic title normalization and news-title shape diagnostics."""

from __future__ import annotations

import re
import unicodedata


ROLLING_COVERAGE_RE = re.compile(
    r"(?i)(?:^|\b)(live|updates?|latest|timeline|breaking|as it happened|rolling)(?:\b|[：:])|直播|快讯"
)
EXPLAINER_RE = re.compile(r"(?i)(?:^|\b)(explainer|analysis|opinion|factbox|what to know)(?:\b|[：:])|解读|评论")
LEADING_LABEL_RE = re.compile(
    r"(?i)^\s*(?:live|updates?|latest|timeline|breaking|as it happened|rolling|explainer|analysis|opinion|factbox)"
    r"\s*[:：|-]\s*"
)
BRACKET_LABEL_RE = re.compile(r"(?i)\s*[\[(（【]\s*(?:video|live|update|updates|photos?|图集|视频)\s*[\])）】]\s*")
MEDIA_SUFFIX_RE = re.compile(
    r"\s*(?:\|| - | – | — )\s*"
    r"(?:[A-Z][A-Za-z0-9 ._-]*(?:news|times|post|daily|reuters|ap|afp|bbc|cnn|dw|xinhua|al jazeera).*)$",
    flags=re.IGNORECASE,
)
DW_DATE_SUFFIX_RE = re.compile(r"\s*[–-]\s*DW\s*[–-]\s*\d{1,2}\s*[./]\s*\d{1,2}\s*[./]\s*\d{4}\s*$", re.IGNORECASE)
PUNCT_RE = re.compile(r"[^\w\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7a3\u0400-\u04ff]+", re.UNICODE)


def normalize_title_for_matching(title: str | None) -> str:
    """Return a conservative event-core title key used for dedupe and diagnostics."""
    text = unicodedata.normalize("NFKC", str(title or "")).strip()
    if not text:
        return ""

    text = BRACKET_LABEL_RE.sub(" ", text)
    text = LEADING_LABEL_RE.sub("", text)
    text = DW_DATE_SUFFIX_RE.sub("", text)
    text = MEDIA_SUFFIX_RE.sub("", text)
    text = PUNCT_RE.sub(" ", text.casefold())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_title_risk_flags(title: str | None) -> list[str]:
    """Detect generic news-title shapes that are risky for event clustering."""
    text = unicodedata.normalize("NFKC", str(title or "")).strip()
    flags: list[str] = []
    if ROLLING_COVERAGE_RE.search(text):
        flags.append("rolling_coverage")
    if EXPLAINER_RE.search(text):
        flags.append("analysis_or_explainer")
    return flags
