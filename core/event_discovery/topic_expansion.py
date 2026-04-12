"""Topic alias expansion for multilingual candidate recall."""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Any
import unicodedata

from configs.model_config import MODEL_CONFIG
from configs.pipeline_config import PIPELINE_CONFIG

try:
    from langdetect import detect as _detect_language
except Exception:  # pragma: no cover - dependency exists in runtime environment.
    _detect_language = None


TOPIC_EXPANSION_LANGS = list(PIPELINE_CONFIG.get("topic_expansion_langs", []))
TOPIC_TRANSLATION_MODEL = MODEL_CONFIG["topic_translation_model"]

NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn",
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "es": "spa_Latn",
    "ko": "kor_Hang",
    "fr": "fra_Latn",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "sw": "swh_Latn",
}


def _normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text).strip()


def _contains_latin(text: str) -> bool:
    return any("A" <= char <= "Z" or "a" <= char <= "z" for char in text)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _contains_hangul(text: str) -> bool:
    return any("\uac00" <= char <= "\ud7a3" for char in text)


def _contains_cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04ff" for char in text)


def detect_topic_language(topic: str) -> str:
    """Best-effort language detection for short topic strings."""
    text = _normalize_text(topic)
    if not text:
        return "en"

    if _contains_cjk(text):
        return "zh-cn"
    if _contains_hangul(text):
        return "ko"
    if _contains_cyrillic(text):
        if any(char in text for char in "іїєґІЇЄҐ"):
            return "uk"
        return "ru"

    if _detect_language is None:
        return "en"

    try:
        detected = _detect_language(text)
    except Exception:
        return "en"

    language_aliases = {
        "zh-cn": "zh-cn",
        "zh-tw": "zh-tw",
        "zh": "zh-cn",
        "en": "en",
        "es": "es",
        "ko": "ko",
        "fr": "fr",
        "ru": "ru",
        "uk": "uk",
        "sw": "sw",
    }
    return language_aliases.get(detected, "en")


@lru_cache(maxsize=1)
def _load_translation_stack():
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ModuleNotFoundError:
        return None

    model_name = TOPIC_TRANSLATION_MODEL
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True)
    except Exception:
        # Topic expansion is a recall improvement, not a hard dependency for the
        # main event-discovery path. Fall back to the original topic when the
        # translation model is unavailable or cannot be downloaded.
        return None

    return tokenizer, model


def _translate_topic_once(topic: str, src_lang: str, tgt_lang: str) -> str | None:
    src_code = NLLB_LANGUAGE_CODES.get(src_lang)
    tgt_code = NLLB_LANGUAGE_CODES.get(tgt_lang)
    if not src_code or not tgt_code:
        return None
    if src_code == tgt_code:
        return _normalize_text(topic)

    translation_stack = _load_translation_stack()
    if translation_stack is None:
        return None

    tokenizer, model = translation_stack
    tokenizer.src_lang = src_code
    inputs = tokenizer(topic, return_tensors="pt")
    generated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),
        max_new_tokens=32,
    )
    translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    normalized = _normalize_text(translated)
    return normalized or None


def _dedupe_aliases(aliases: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        normalized = _normalize_text(alias)
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def _looks_like_latin_named_entity(topic: str) -> bool:
    text = _normalize_text(topic)
    if not re.fullmatch(r"[A-Za-z0-9 .,&'’/-]+", text):
        return False
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    if not tokens:
        return False
    return all(token[:1].isupper() for token in tokens if token[:1].isalpha())


def _is_valid_alias(source_topic: str, alias: str) -> bool:
    normalized_source = _normalize_text(source_topic)
    normalized_alias = _normalize_text(alias)
    if not normalized_alias:
        return False

    if normalized_alias.casefold() == normalized_source.casefold():
        return False

    # Drop obviously broken duplications like "Apple Apple".
    alias_tokens = re.findall(r"[A-Za-z0-9]+", normalized_alias.casefold())
    if len(alias_tokens) >= 2 and len(set(alias_tokens)) == 1:
        return False

    # One-character aliases are usually too noisy for candidate recall.
    compact_alias = re.sub(r"\s+", "", normalized_alias)
    if len(compact_alias) <= 1:
        return False

    # Avoid aliases that still contain the original Latin token plus extra filler,
    # e.g. "Apple Apple" or particle-attached forms that are poor recall keys.
    if _contains_latin(normalized_source) and _contains_latin(normalized_alias):
        source_tokens = re.findall(r"[A-Za-z0-9]+", normalized_source.casefold())
        alias_text = normalized_alias.casefold()
        if source_tokens and any(token in alias_text for token in source_tokens):
            if alias_text != normalized_source.casefold():
                return False

    return True


def expand_topic_aliases(topic: str, target_langs: list[str] | None = None) -> list[str]:
    """Expand a topic into multilingual aliases for candidate recall."""
    normalized_topic = _normalize_text(topic)
    if not normalized_topic:
        return []

    aliases = [normalized_topic]
    if not PIPELINE_CONFIG.get("topic_expansion_enabled", True):
        return aliases

    source_lang = detect_topic_language(normalized_topic)
    target_languages = target_langs or TOPIC_EXPANSION_LANGS

    # Proper names like Apple/Trump are too ambiguous as single-token translation
    # inputs. For those topics, automatic multilingual expansion hurts recall more
    # than it helps, so we keep only the original surface form for now.
    skip_translation_for_named_entity = _looks_like_latin_named_entity(normalized_topic)
    if skip_translation_for_named_entity:
        return aliases

    for target_lang in target_languages:
        if target_lang == source_lang:
            continue
        translated = _translate_topic_once(normalized_topic, source_lang, target_lang)
        if translated and _is_valid_alias(normalized_topic, translated):
            aliases.append(translated)

    return _dedupe_aliases(aliases)
