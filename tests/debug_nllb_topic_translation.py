"""Debug NLLB topic translation outputs.

This is an operator script, not a pytest test. Run it directly, for example:

    python tests/debug_nllb_topic_translation.py --topic Apple --local-files-only

It prints one row per target language so short-topic translation failures are
visible before aliases enter candidate recall.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
import unicodedata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import MODEL_CONFIG
from configs.pipeline_config import PIPELINE_CONFIG
from core.event_discovery.topic_expansion import (
    NLLB_LANGUAGE_CODES,
    _is_valid_alias,
    _looks_like_latin_named_entity,
    _normalize_text,
    detect_topic_language,
)


def _dedupe_key(text: str) -> str:
    return _normalize_text(text).casefold()


def _alias_decision(source_topic: str, alias: str | None, seen: set[str]) -> tuple[bool, str]:
    if alias is None:
        return False, "translation_failed"

    normalized_source = _normalize_text(source_topic)
    normalized_alias = _normalize_text(alias)
    if not normalized_alias:
        return False, "empty"

    if normalized_alias.casefold() == normalized_source.casefold():
        return False, "same_as_source"

    alias_tokens = re.findall(r"[A-Za-z0-9]+", normalized_alias.casefold())
    if len(alias_tokens) >= 2 and len(set(alias_tokens)) == 1:
        return False, "repeated_latin_token"

    compact_alias = re.sub(r"\s+", "", normalized_alias)
    if len(compact_alias) <= 1:
        return False, "too_short"

    if not _is_valid_alias(normalized_source, normalized_alias):
        return False, "invalid_by_pipeline_filter"

    key = _dedupe_key(normalized_alias)
    if key in seen:
        return False, "duplicate_alias"

    seen.add(key)
    return True, "kept"


def _load_nllb_stack(
    model_name: str,
    *,
    local_files_only: bool,
    use_safetensors: bool,
) -> tuple[Any, Any]:
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        use_safetensors=use_safetensors,
    )
    return tokenizer, model


def _translate_once(tokenizer: Any, model: Any, text: str, src_code: str, tgt_code: str) -> str:
    tokenizer.src_lang = src_code
    inputs = tokenizer(text, return_tensors="pt")
    generated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),
        max_new_tokens=32,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def debug_topic_translation(
    topic: str,
    *,
    source_lang: str | None,
    target_langs: list[str],
    model_name: str,
    local_files_only: bool,
    use_safetensors: bool,
) -> dict[str, Any]:
    normalized_topic = _normalize_text(topic)
    if not normalized_topic:
        raise ValueError("topic must be a non-empty string")

    resolved_source_lang = source_lang or detect_topic_language(normalized_topic)
    source_code = NLLB_LANGUAGE_CODES.get(resolved_source_lang)
    if source_code is None:
        raise ValueError(f"unsupported source language: {resolved_source_lang}")

    payload: dict[str, Any] = {
        "topic": normalized_topic,
        "source_lang": resolved_source_lang,
        "source_code": source_code,
        "model": model_name,
        "local_files_only": local_files_only,
        "use_safetensors": use_safetensors,
        "looks_like_latin_named_entity": _looks_like_latin_named_entity(normalized_topic),
        "target_langs": target_langs,
        "load_error": None,
        "rows": [],
    }

    try:
        tokenizer, model = _load_nllb_stack(
            model_name,
            local_files_only=local_files_only,
            use_safetensors=use_safetensors,
        )
    except Exception as exc:  # noqa: BLE001 - this script is for surfacing setup failures.
        payload["load_error"] = f"{type(exc).__name__}: {exc}"
        return payload

    seen = {_dedupe_key(normalized_topic)}
    for target_lang in target_langs:
        target_code = NLLB_LANGUAGE_CODES.get(target_lang)
        row: dict[str, Any] = {
            "target_lang": target_lang,
            "target_code": target_code,
            "raw_translation": None,
            "normalized_alias": None,
            "kept": False,
            "decision": None,
            "error": None,
        }

        if target_code is None:
            row["decision"] = "unsupported_target_language"
            payload["rows"].append(row)
            continue

        if target_code == source_code:
            raw_translation = normalized_topic
        else:
            try:
                raw_translation = _translate_once(tokenizer, model, normalized_topic, source_code, target_code)
            except Exception as exc:  # noqa: BLE001 - keep per-language failures visible.
                row["error"] = f"{type(exc).__name__}: {exc}"
                raw_translation = None

        normalized_alias = _normalize_text(raw_translation)
        kept, decision = _alias_decision(normalized_topic, normalized_alias, seen)
        row["raw_translation"] = raw_translation
        row["normalized_alias"] = normalized_alias
        row["kept"] = kept
        row["decision"] = decision
        payload["rows"].append(row)

    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug NLLB topic translation for multilingual alias expansion.")
    parser.add_argument("--topic", default="Apple", help="Topic string to translate.")
    parser.add_argument("--source-lang", default=None, help="Override detected source language, e.g. en.")
    parser.add_argument(
        "--target-langs",
        nargs="+",
        default=list(PIPELINE_CONFIG.get("topic_expansion_langs", [])),
        help="Target language keys, e.g. zh-cn es ko fr ru uk sw.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_CONFIG["topic_translation_model"],
        help="Hugging Face model id or local path.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local Hugging Face cache; helpful for reproducible/offline debugging.",
    )
    parser.add_argument(
        "--allow-pytorch-bin",
        action="store_true",
        help="Allow loading pytorch_model.bin when model.safetensors is unavailable.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON payload instead of a compact table.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = debug_topic_translation(
        args.topic,
        source_lang=args.source_lang,
        target_langs=args.target_langs,
        model_name=args.model,
        local_files_only=args.local_files_only,
        use_safetensors=not args.allow_pytorch_bin,
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1 if payload.get("load_error") else 0

    print(f"topic: {payload['topic']}")
    print(f"source: {payload['source_lang']} ({payload['source_code']})")
    print(f"model: {payload['model']}")
    print(f"local_files_only: {payload['local_files_only']}")
    print(f"use_safetensors: {payload['use_safetensors']}")
    print(f"looks_like_latin_named_entity: {payload['looks_like_latin_named_entity']}")
    if payload.get("load_error"):
        print(f"load_error: {payload['load_error']}")
        return 1

    print()
    print(f"{'lang':<8} {'code':<10} {'kept':<5} {'decision':<28} raw_translation")
    print("-" * 88)
    for row in payload["rows"]:
        raw_translation = row["raw_translation"] if row["raw_translation"] is not None else ""
        error = f" | {row['error']}" if row.get("error") else ""
        print(
            f"{row['target_lang']:<8} "
            f"{str(row['target_code']):<10} "
            f"{str(row['kept']):<5} "
            f"{str(row['decision']):<28} "
            f"{raw_translation}{error}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
