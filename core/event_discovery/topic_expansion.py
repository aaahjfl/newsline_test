"""Topic alias expansion for multilingual candidate recall."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import re
from typing import Any
import unicodedata

from configs.model_config import MODEL_CONFIG
from configs.pipeline_config import PIPELINE_CONFIG
from core.llm.ollama_client import generate_with_ollama

try:
    from langdetect import detect as _detect_language
except Exception:  # pragma: no cover - dependency exists in runtime environment.
    _detect_language = None


TOPIC_EXPANSION_LANGS = list(PIPELINE_CONFIG.get("topic_expansion_langs", []))
TOPIC_TRANSLATION_MODEL = MODEL_CONFIG["topic_translation_model"]
TOPIC_ALIAS_MODEL = MODEL_CONFIG.get("topic_alias_model", MODEL_CONFIG.get("reasoning_model", "qwen3.5:9b"))
STRONG_PRIORITY = "strong"
WEAK_PRIORITY = "weak"

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

LANGUAGE_NAMES = {
    "en": "English",
    "zh-cn": "Simplified Chinese",
    "es": "Spanish",
    "ko": "Korean",
    "fr": "French",
    "ru": "Russian",
    "uk": "Ukrainian",
    "sw": "Swahili",
}

ALIAS_SPLIT_PATTERN = re.compile(r"\s*(?:[,;；、/|]+|\bor\b)\s*", flags=re.IGNORECASE)
ARTICLE_PREFIX_PATTERN = re.compile(r"^(?:the|a|an|la|le|el|los|las|les)\s+", flags=re.IGNORECASE)
RELATED_PHRASE_MARKERS = {
    "news",
    "headline",
    "headlines",
    "topic",
    "topics",
    "event",
    "events",
    "related",
    "about",
    "latest",
    "stock",
    "stocks",
    "share",
    "shares",
    "technology",
    "tech",
    "新闻",
    "报道",
    "报导",
    "话题",
    "事件",
    "相关",
    "最新",
    "股票",
    "股",
    "科技",
    "ニュース",
    "뉴스",
    "주식",
}


@dataclass(frozen=True, slots=True)
class TopicAlias:
    """A cleaned topic alias with recall priority metadata."""

    text: str
    lang: str
    priority: str = STRONG_PRIORITY
    notes: tuple[str, ...] = ()


def _normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text).strip()
    normalized = normalized.strip("\"'`“”‘’")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


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

    alias_tokens = re.findall(r"[A-Za-z0-9]+", normalized_alias.casefold())
    if len(alias_tokens) >= 2 and len(set(alias_tokens)) == 1:
        return False

    compact_alias = re.sub(r"\s+", "", normalized_alias)
    if len(compact_alias) <= 1:
        return False

    if _looks_like_related_phrase(normalized_alias):
        return False

    if normalized_alias.count(" ") >= 8:
        return False

    # A very long CJK alias is usually an explanatory phrase, not a search key.
    if _contains_cjk(normalized_alias) and len(compact_alias) > 12:
        return False

    return bool(normalized_source)


def _alias_notes(source_topic: str, alias: str) -> tuple[str, ...]:
    normalized_source = _normalize_text(source_topic)
    normalized_alias = _normalize_text(alias)
    if not normalized_source or not normalized_alias:
        return ()

    notes: list[str] = []
    source_key = normalized_source.casefold()
    alias_key = normalized_alias.casefold()
    compact_alias = re.sub(r"\s+", "", normalized_alias)

    if _looks_like_latin_named_entity(normalized_source) and alias_key != source_key:
        source_tokens = [
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9]+", normalized_source)
            if len(token) > 1
        ]
        if source_tokens and not any(token in alias_key for token in source_tokens):
            notes.append("possible_translated_named_entity")

    if len(compact_alias) <= 2 and alias_key != source_key:
        notes.append("very_short_alias")

    return tuple(dict.fromkeys(notes))


def _looks_like_related_phrase(alias: str) -> bool:
    lowered = alias.casefold()
    return any(marker in lowered for marker in RELATED_PHRASE_MARKERS)


def _extract_parenthetical_parts(text: str) -> list[str]:
    parts = [text]
    for match in re.finditer(r"[\(（]([^\)）]+)[\)）]", text):
        parts.append(match.group(1))
    without_parenthetical = re.sub(r"[\(（][^\)）]+[\)）]", "", text).strip()
    if without_parenthetical and without_parenthetical != text:
        parts.append(without_parenthetical)
    return parts


def _split_alias_text(alias: Any) -> list[str]:
    text = _normalize_text(alias)
    if not text:
        return []

    split_parts: list[str] = []
    for part in _extract_parenthetical_parts(text):
        for piece in ALIAS_SPLIT_PATTERN.split(part):
            normalized = _normalize_text(piece)
            normalized = ARTICLE_PREFIX_PATTERN.sub("", normalized).strip()
            if normalized:
                split_parts.append(normalized)
    return split_parts


def _alias_items_from_value(value: Any) -> list[str]:
    if isinstance(value, str):
        return _split_alias_text(value)
    if isinstance(value, list):
        items: list[str] = []
        for entry in value:
            items.extend(_alias_items_from_value(entry))
        return items
    if isinstance(value, dict):
        items = []
        for candidate_key in ("alias", "text", "name", "value"):
            if candidate_key in value:
                items.extend(_alias_items_from_value(value[candidate_key]))
        return items
    return []


def _priority_key(priority: str) -> int:
    return 0 if priority == STRONG_PRIORITY else 1


def _dedupe_topic_aliases(aliases: list[TopicAlias]) -> list[TopicAlias]:
    by_key: dict[str, TopicAlias] = {}
    for alias in aliases:
        normalized = _normalize_text(alias.text)
        if not normalized:
            continue
        key = f"{alias.lang}:{normalized.casefold()}"
        existing = by_key.get(key)
        candidate = TopicAlias(normalized, alias.lang, alias.priority, alias.notes)
        if existing is None or _priority_key(candidate.priority) < _priority_key(existing.priority):
            by_key[key] = candidate
    return list(by_key.values())


def _limit_aliases_by_language(aliases: list[TopicAlias]) -> list[TopicAlias]:
    per_language_limit = int(PIPELINE_CONFIG.get("topic_alias_per_language_limit", 4))
    total_limit = int(PIPELINE_CONFIG.get("topic_alias_total_limit", 24))
    counters: dict[str, int] = {}
    limited: list[TopicAlias] = []

    for alias in aliases:
        if counters.get(alias.lang, 0) >= per_language_limit:
            continue
        counters[alias.lang] = counters.get(alias.lang, 0) + 1
        limited.append(alias)
        if len(limited) >= total_limit:
            break
    return limited


def _build_llm_alias_prompt(topic: str, target_langs: list[str]) -> str:
    language_lines = "\n".join(
        f"- {lang}: {LANGUAGE_NAMES.get(lang, lang)}"
        for lang in target_langs
    )
    return f"""/no_think
You are generating multilingual search aliases for a news event-discovery system.

Topic: {topic}

Target languages:
{language_lines}

Return only valid JSON, with exactly this schema:
{{
  "aliases": {{
    "en": ["..."],
    "zh-cn": ["..."]
  }}
}}

Rules:
- aliases must be names, aliases, translations, or transliterations for the same real-world entity/topic.
- Do not include related concepts, categories, roles, descriptions, news/event words, or explanations.
- Do not translate news titles. Only translate or transliterate the topic itself.
- Prefer concise SQL LIKE search keys, not full sentences.
- Include 1 to 4 aliases per language.
- Use every exact language key listed above.
- If a target language commonly uses the same Latin-script name, include that same name under that language key.
- If a named entity is commonly written in Latin script in a target language, repeat that Latin name for that language.
- For company, brand, product, person, organization, and place names, preserve common original-script names when local news commonly uses them.
- Do not turn a proper name into only a generic category, role, product type, or literal common-noun meaning.
- Example: for Fed, use "Fed", "Federal Reserve", "美联储", "Reserva Federal"; do not include "US government" or generic "central bank".
- Example: for Trump, use "Donald Trump", "Trump", "特朗普", and "川普"; do not include "US president" or "Republican Party".
"""


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("LLM alias response must be a JSON object.")
    return payload


def _load_llm_alias_payload(topic: str, target_langs: list[str]) -> dict[str, Any]:
    prompt = _build_llm_alias_prompt(topic, target_langs)
    text = generate_with_ollama(
        prompt,
        model=TOPIC_ALIAS_MODEL,
        url=str(PIPELINE_CONFIG.get("topic_alias_ollama_url", "http://localhost:11434/api/generate")),
        timeout_seconds=int(PIPELINE_CONFIG.get("topic_alias_request_timeout_seconds", 45)),
        keep_alive=str(PIPELINE_CONFIG.get("topic_alias_ollama_keep_alive", "0s")),
        think=bool(PIPELINE_CONFIG.get("topic_alias_ollama_think", False)),
        options={
            "temperature": 0,
            "num_ctx": int(PIPELINE_CONFIG.get("topic_alias_ollama_num_ctx", 2048)),
            "num_predict": int(PIPELINE_CONFIG.get("topic_alias_ollama_num_predict", 256)),
        },
    )
    return _extract_json_object(text)


def _aliases_from_payload(topic: str, target_langs: list[str], payload: dict[str, Any]) -> list[TopicAlias]:
    aliases: list[TopicAlias] = [TopicAlias(_normalize_text(topic), detect_topic_language(topic), STRONG_PRIORITY)]

    if isinstance(payload.get("aliases"), dict):
        seen_langs: set[str] = set()
        for lang in target_langs:
            for alias_text in _alias_items_from_value(payload["aliases"].get(lang)):
                if _is_valid_alias(topic, alias_text):
                    aliases.append(TopicAlias(alias_text, lang, STRONG_PRIORITY, _alias_notes(topic, alias_text)))
                    seen_langs.add(lang)
        if _looks_like_latin_named_entity(topic):
            for lang in target_langs:
                if lang not in seen_langs:
                    aliases.append(TopicAlias(topic, lang, STRONG_PRIORITY))
        return _limit_aliases_by_language(_dedupe_topic_aliases(aliases))

    # Backward-compatible parser for older prompts/tests.
    seen_langs: set[str] = set()
    for field_name, priority in (("strong_aliases", STRONG_PRIORITY), ("weak_aliases", STRONG_PRIORITY)):
        field_value = payload.get(field_name)
        if not isinstance(field_value, dict):
            continue
        for lang in target_langs:
            for alias_text in _alias_items_from_value(field_value.get(lang)):
                if _is_valid_alias(topic, alias_text):
                    aliases.append(TopicAlias(alias_text, lang, priority, _alias_notes(topic, alias_text)))
                    seen_langs.add(lang)
    if _looks_like_latin_named_entity(topic):
        for lang in target_langs:
            if lang not in seen_langs:
                aliases.append(TopicAlias(topic, lang, STRONG_PRIORITY))

    return _limit_aliases_by_language(_dedupe_topic_aliases(aliases))


def _expand_topic_aliases_with_nllb(topic: str, target_langs: list[str]) -> list[TopicAlias]:
    normalized_topic = _normalize_text(topic)
    if not normalized_topic:
        return []

    source_lang = detect_topic_language(normalized_topic)
    aliases = [TopicAlias(normalized_topic, source_lang, STRONG_PRIORITY)]
    if _looks_like_latin_named_entity(normalized_topic):
        return aliases

    for target_lang in target_langs:
        if target_lang == source_lang:
            continue
        translated = _translate_topic_once(normalized_topic, source_lang, target_lang)
        if translated and _is_valid_alias(normalized_topic, translated):
            aliases.append(
                TopicAlias(
                    translated,
                    target_lang,
                    STRONG_PRIORITY,
                    _alias_notes(normalized_topic, translated),
                )
            )

    return _limit_aliases_by_language(_dedupe_topic_aliases(aliases))


def expand_topic_alias_candidates(topic: str, target_langs: list[str] | None = None) -> list[TopicAlias]:
    """Expand a topic into cleaned multilingual aliases."""
    normalized_topic = _normalize_text(topic)
    if not normalized_topic:
        return []

    source_lang = detect_topic_language(normalized_topic)
    fallback = [TopicAlias(normalized_topic, source_lang, STRONG_PRIORITY)]
    if not PIPELINE_CONFIG.get("topic_expansion_enabled", True):
        return fallback

    target_languages = target_langs or TOPIC_EXPANSION_LANGS
    if not target_languages:
        return fallback

    backend = str(PIPELINE_CONFIG.get("topic_alias_backend", "ollama")).casefold()
    if backend == "nllb":
        return _expand_topic_aliases_with_nllb(normalized_topic, target_languages)

    try:
        payload = _load_llm_alias_payload(normalized_topic, target_languages)
        aliases = _aliases_from_payload(normalized_topic, target_languages, payload)
    except Exception:
        return fallback

    return aliases or fallback


def split_aliases_by_priority(aliases: list[TopicAlias]) -> tuple[list[str], list[str]]:
    """Return strong and weak alias texts, preserving cleaned priority."""
    strong = [alias.text for alias in aliases if alias.priority == STRONG_PRIORITY]
    weak = [alias.text for alias in aliases if alias.priority == WEAK_PRIORITY]
    return _dedupe_aliases(strong), _dedupe_aliases(weak)


def expand_topic_aliases(topic: str, target_langs: list[str] | None = None) -> list[str]:
    """Expand a topic into alias strings for candidate recall."""
    aliases = expand_topic_alias_candidates(topic, target_langs=target_langs)
    strong, weak = split_aliases_by_priority(aliases)
    return _dedupe_aliases(strong + weak)
