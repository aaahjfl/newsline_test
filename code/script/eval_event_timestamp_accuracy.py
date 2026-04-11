import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import dateparser
import pymysql
from dateparser.search import search_dates
from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "database": "news_db",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}

TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

MONTH_NAME_PATTERN = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)

FUTURE_HINT_PATTERN = re.compile(
    r"(将于|将会|即将|计划于|预计|拟于|to be held|will|upcoming|is set to|next)",
    flags=re.IGNORECASE,
)
PAST_HINT_PATTERN = re.compile(
    r"(此前|去年|上月|昨日|昨天|last|previous|ago|earlier)",
    flags=re.IGNORECASE,
)

DAY_TOKEN_PATTERN = re.compile(
    rf"(\d{{1,2}}[/-]\d{{1,2}}|\d{{1,2}}月\d{{1,2}}[日号]?|"
    rf"\b{MONTH_NAME_PATTERN}\s+\d{{1,2}}(?:,\s*\d{{4}})?\b|"
    r"today|yesterday|tomorrow|今天|昨日|昨天|明天|后天|前天)",
    flags=re.IGNORECASE,
)

MONTH_TOKEN_PATTERN = re.compile(
    rf"(\d{{4}}年\d{{1,2}}月|\d{{1,2}}月|\b{MONTH_NAME_PATTERN}\b|"
    r"Q[1-4]|第?[一二三四1-4]季度|上半年|下半年|this month|last month|next month|本月|上月|下月)",
    flags=re.IGNORECASE,
)

YEAR_TOKEN_PATTERN = re.compile(
    r"((?:19|20)\d{2}|this year|last year|next year|今年|去年|明年)",
    flags=re.IGNORECASE,
)

DATE_CUE_PATTERN = re.compile(
    rf"({DAY_TOKEN_PATTERN.pattern}|{MONTH_TOKEN_PATTERN.pattern}|{YEAR_TOKEN_PATTERN.pattern})",
    flags=re.IGNORECASE,
)

CURRENCY_AROUND_PATTERN = re.compile(
    r"([$¥￥€£₹]|USD|CNY|RMB|HKD|S\$|元|万元|亿元|万亿|美元|million|billion|trillion)",
    re.IGNORECASE,
)
PERCENT_AROUND_PATTERN = re.compile(r"\d+(?:\s*[.,]\s*\d+)?\s*%")
YEAR_CONTEXT_HINT_PATTERN = re.compile(
    r"\b(in|by|during|since|from|until|before|after|for|fiscal|fy|season)\b|(?:于|在|截至|到|自|至|年|年度|财年|季度)",
    re.IGNORECASE,
)
YEAR_RANGE_PATTERN = re.compile(r"(19\d{2}|20\d{2})\s*[-/~]\s*(19\d{2}|20\d{2})")
BAND_NAME_YEAR_PATTERN = re.compile(r"\bthe\s+1975\b", re.IGNORECASE)

DATEPARSER_LANGUAGE_MAP = {
    "en": ["en"],
    "zh-cn": ["zh"],
    "zh-tw": ["zh"],
    "zh": ["zh"],
    "es": ["es"],
    "fr": ["fr"],
    "ru": ["ru"],
    "ko": ["ko"],
    "uk": ["uk"],
    "sw": ["sw"],
}


@dataclass
class PseudoGold:
    value: datetime
    range_start: datetime
    range_end: datetime
    granularity: str  # day|month|year
    source_text: str
    pattern: str
    confidence: str  # high|medium|low


@dataclass
class Candidate:
    text: str
    start: int
    pattern: str
    granularity: str
    confidence: str


def get_db_connection(db_config: Dict) -> pymysql.connections.Connection:
    return pymysql.connect(**db_config)


def normalize_dt(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return dateparser.parse(text)
    return None


def detect_lang(text: str) -> str:
    if not text or not text.strip():
        return "empty"
    try:
        return detect(text)
    except Exception:
        return "unknown"


def detect_prefer_dates_from(title: str) -> str:
    if FUTURE_HINT_PATTERN.search(title):
        return "future"
    if PAST_HINT_PATTERN.search(title):
        return "past"
    return "current_period"


def safe_replace_year(dt_obj: datetime, year: int) -> Optional[datetime]:
    try:
        return dt_obj.replace(year=year)
    except ValueError:
        return None


def infer_year_for_month_day(month: int, day: int, base_time: datetime, title: str) -> Optional[int]:
    try:
        base_candidate = datetime(base_time.year, month, day)
    except ValueError:
        return None

    prefer = detect_prefer_dates_from(title)
    if prefer == "future" and base_candidate < (base_time - timedelta(days=1)):
        return base_time.year + 1
    if prefer == "past" and base_candidate > (base_time + timedelta(days=1)):
        return base_time.year - 1

    nearby = [base_candidate]
    prev_year = safe_replace_year(base_candidate, base_candidate.year - 1)
    next_year = safe_replace_year(base_candidate, base_candidate.year + 1)
    if prev_year:
        nearby.append(prev_year)
    if next_year:
        nearby.append(next_year)
    nearby.sort(key=lambda item: abs(item - base_time))
    return nearby[0].year


def has_currency_or_percent_around(text: str, start: int, end: int, window: int = 6) -> bool:
    left = max(0, start - window)
    right = min(len(text), end + window)
    around = text[left:right]
    return CURRENCY_AROUND_PATTERN.search(around) is not None or PERCENT_AROUND_PATTERN.search(around) is not None


def is_noisy_year_candidate(title: str, matched_text: str, match_obj: Optional[re.Match]) -> bool:
    text = (matched_text or "").strip()
    if not text:
        return True

    # "The 1975" 常见为乐队名，不应作为年份事件
    if BAND_NAME_YEAR_PATTERN.search(text):
        return True
    if BAND_NAME_YEAR_PATTERN.search(title) and re.fullmatch(r"(?:the\s+)?1975", text, flags=re.IGNORECASE):
        return True

    pure_year_match = re.fullmatch(r"(19\d{2}|20\d{2})", text)
    if pure_year_match is None:
        return False

    # 区间年份（如 2021-2024）弱监督歧义较大，避免给 parser 造成错误惩罚
    if match_obj:
        left = max(0, match_obj.start() - 12)
        right = min(len(title), match_obj.end() + 12)
        around = title[left:right]

        if YEAR_RANGE_PATTERN.search(around):
            return True

    return False


def normalize_by_granularity(dt_value: datetime, granularity: str) -> datetime:
    if granularity == "day":
        return dt_value.replace(hour=0, minute=0, second=0, microsecond=0)
    if granularity == "month":
        return dt_value.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if granularity == "year":
        return dt_value.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return dt_value


def day_floor(dt_value: datetime) -> datetime:
    return dt_value.replace(hour=0, minute=0, second=0, microsecond=0)


def month_last_day(year: int, month: int) -> int:
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    return (next_month - timedelta(days=1)).day


def range_by_granularity(dt_value: datetime, granularity: str):
    value = day_floor(dt_value)
    if granularity == "day":
        return value, value
    if granularity == "month":
        start = datetime(value.year, value.month, 1)
        end = datetime(value.year, value.month, month_last_day(value.year, value.month))
        return start, end
    if granularity == "year":
        return datetime(value.year, 1, 1), datetime(value.year, 12, 31)
    return value, value


def normalize_range(start_dt: Optional[datetime], end_dt: Optional[datetime]):
    if start_dt is None or end_dt is None:
        return None, None
    start = day_floor(start_dt)
    end = day_floor(end_dt)
    if end < start:
        start, end = end, start
    return start, end


def ranges_overlap(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return a_start <= b_end and b_start <= a_end


def parse_candidate_text(candidate: str, base_time: datetime, lang: str, title: str) -> Optional[datetime]:
    settings = {
        "RELATIVE_BASE": base_time,
        "TIMEZONE": "Asia/Shanghai",
        "RETURN_AS_TIMEZONE_AWARE": False,
        "PREFER_DAY_OF_MONTH": "first",
        "DATE_ORDER": "YMD",
        "PREFER_DATES_FROM": detect_prefer_dates_from(title),
    }
    langs = DATEPARSER_LANGUAGE_MAP.get(lang)
    try:
        if langs:
            return dateparser.parse(candidate, settings=settings, languages=langs)
        return dateparser.parse(candidate, settings=settings)
    except Exception:
        return None


def normalize_quarter(quarter_text: str, year_text: str) -> Optional[datetime]:
    q_map = {
        "1": 1,
        "2": 4,
        "3": 7,
        "4": 10,
        "一": 1,
        "二": 4,
        "三": 7,
        "四": 10,
    }
    month = q_map.get(quarter_text)
    if month is None:
        return None
    try:
        return datetime(int(year_text), month, 1)
    except ValueError:
        return None


def extract_regex_candidates(title: str, mode: str) -> List[Candidate]:
    patterns = [
        ("ymd_numeric", r"\b(19\d{2}|20\d{2})[/-](\d{1,2})[/-](\d{1,2})\b", "day", "high"),
        ("zh_ymd", r"(19\d{2}|20\d{2})年(\d{1,2})月(\d{1,2})[日号]?", "day", "high"),
        ("en_mdy", rf"\b({MONTH_NAME_PATTERN})\s+\d{{1,2}},?\s+(19\d{{2}}|20\d{{2}})\b", "day", "high"),
        ("zh_md", r"(?<!\d)(\d{1,2})月(\d{1,2})[日号]?", "day", "high"),
        ("en_my", rf"\b({MONTH_NAME_PATTERN})\s+(19\d{{2}}|20\d{{2}})\b", "month", "high"),
        ("zh_ym", r"(19\d{2}|20\d{2})年(\d{1,2})月", "month", "high"),
    ]

    if mode in {"balanced", "loose"}:
        patterns.extend([
            ("relative_day", r"\b(today|yesterday|tomorrow)\b|今天|昨日|昨天|明天|后天|前天", "day", "medium"),
            ("relative_month", r"\b(this month|last month|next month)\b|本月|上月|下月", "month", "medium"),
            ("relative_year", r"\b(this year|last year|next year)\b|今年|去年|明年", "year", "medium"),
            ("zh_half", r"(?:(19\d{2}|20\d{2})年)?(上半年|下半年)", "month", "medium"),
            ("en_half", r"\b(?:H([12])\s+(19\d{2}|20\d{2})|(first|second)\s+half\s+of\s+(19\d{2}|20\d{2}))\b", "month", "medium"),
            ("zh_quarter", r"(19\d{2}|20\d{2})年?第?([一二三四1-4])季度", "month", "medium"),
            ("en_quarter", r"\b(?:Q([1-4])\s+(19\d{2}|20\d{2})|(19\d{2}|20\d{2})\s+Q([1-4]))\b", "month", "medium"),
            ("year_context", r"\b(?:in|by|during|since|from)\s+(19\d{2}|20\d{2})\b|(?:于|在|截至|到|自)(19\d{2}|20\d{2})年", "year", "low"),
            ("zh_month_only", r"(?:于|在|到|截至|本|上|下)?\s*(\d{1,2})月(?!\d)", "month", "low"),
            ("en_month_only", rf"\b(?:in|during|for|by|this|last|next)\s+({MONTH_NAME_PATTERN})\b", "month", "low"),
        ])

    candidates: List[Candidate] = []
    for pattern_name, pattern, granularity, confidence in patterns:
        for match in re.finditer(pattern, title, flags=re.IGNORECASE):
            start, end = match.start(), match.end()
            text = match.group(0).strip()
            if not text:
                continue
            if has_currency_or_percent_around(title, start, end):
                continue
            candidates.append(Candidate(text=text, start=start, pattern=pattern_name, granularity=granularity, confidence=confidence))

    return candidates


def extract_search_dates_candidate(title: str, base_time: datetime, lang: str, mode: str) -> Optional[Candidate]:
    if mode == "strict":
        return None

    settings = {
        "RELATIVE_BASE": base_time,
        "TIMEZONE": "Asia/Shanghai",
        "RETURN_AS_TIMEZONE_AWARE": False,
        "PREFER_DAY_OF_MONTH": "first",
        "DATE_ORDER": "YMD",
        "PREFER_DATES_FROM": detect_prefer_dates_from(title),
    }
    langs = DATEPARSER_LANGUAGE_MAP.get(lang)

    try:
        extracted = search_dates(title, languages=langs, settings=settings) if langs else search_dates(title, settings=settings)
    except Exception:
        extracted = None

    if not extracted:
        return None

    for matched_text, _ in extracted:
        if not matched_text:
            continue
        normalized_text = matched_text.strip()
        if re.fullmatch(MONTH_NAME_PATTERN, normalized_text, flags=re.IGNORECASE):
            continue
        match_obj = re.search(re.escape(matched_text), title, flags=re.IGNORECASE)
        if match_obj:
            if has_currency_or_percent_around(title, match_obj.start(), match_obj.end()):
                continue
        if mode == "balanced" and DATE_CUE_PATTERN.search(matched_text) is None:
            continue

        granularity = "day"
        if DAY_TOKEN_PATTERN.search(matched_text):
            granularity = "day"
        elif MONTH_TOKEN_PATTERN.search(matched_text):
            granularity = "month"
        elif YEAR_TOKEN_PATTERN.search(matched_text):
            granularity = "year"
        elif mode == "balanced":
            continue

        if granularity == "year" and is_noisy_year_candidate(title, matched_text, match_obj):
            continue

        return Candidate(
            text=matched_text.strip(),
            start=match_obj.start() if match_obj else 999,
            pattern="search_dates",
            granularity=granularity,
            confidence="low" if mode == "balanced" else "medium",
        )

    return None


def candidate_sort_key(candidate: Candidate):
    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    granularity_rank = {"day": 0, "month": 1, "year": 2}
    pattern_rank = {
        "ymd_numeric": 0,
        "zh_ymd": 1,
        "en_mdy": 2,
        "zh_md": 3,
        "en_my": 4,
        "zh_ym": 5,
        "relative_day": 6,
        "relative_month": 7,
        "relative_year": 8,
        "zh_half": 9,
        "en_half": 10,
        "zh_quarter": 11,
        "en_quarter": 12,
        "year_context": 13,
        "zh_month_only": 14,
        "en_month_only": 15,
        "search_dates": 16,
    }
    return (
        confidence_rank.get(candidate.confidence, 9),
        granularity_rank.get(candidate.granularity, 9),
        pattern_rank.get(candidate.pattern, 99),
        candidate.start,
        len(candidate.text),
    )


def parse_candidate_to_gold(candidate: Candidate, title: str, base_time: datetime, lang: str) -> Optional[PseudoGold]:
    text = candidate.text

    if candidate.pattern == "zh_md":
        m = re.search(r"(?<!\d)(\d{1,2})月(\d{1,2})[日号]?", text)
        if not m:
            return None
        month = int(m.group(1))
        day = int(m.group(2))
        year = infer_year_for_month_day(month, day, base_time, title)
        if year is None:
            return None
        try:
            dt_value = datetime(year, month, day)
        except ValueError:
            return None
        normalized = normalize_by_granularity(dt_value, "day")
        start, end = range_by_granularity(normalized, "day")
        return PseudoGold(normalized, start, end, "day", text, candidate.pattern, candidate.confidence)

    if candidate.pattern == "zh_half":
        m = re.search(r"(?:(19\d{2}|20\d{2})年)?(上半年|下半年)", text)
        if not m:
            return None
        year = int(m.group(1)) if m.group(1) else base_time.year
        half_text = m.group(2)
        if half_text == "上半年":
            start = datetime(year, 1, 1)
            end = datetime(year, 6, 30)
        else:
            start = datetime(year, 7, 1)
            end = datetime(year, 12, 31)
        return PseudoGold(start, start, end, "month", text, candidate.pattern, candidate.confidence)

    if candidate.pattern == "en_half":
        m = re.search(r"H([12])\s+(19\d{2}|20\d{2})", text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"(first|second)\s+half\s+of\s+(19\d{2}|20\d{2})", text, flags=re.IGNORECASE)
            if not m:
                return None
            half = "1" if m.group(1).lower() == "first" else "2"
            year = m.group(2)
        else:
            half = m.group(1)
            year = m.group(2)
        if half == "1":
            start = datetime(int(year), 1, 1)
            end = datetime(int(year), 6, 30)
        else:
            start = datetime(int(year), 7, 1)
            end = datetime(int(year), 12, 31)
        return PseudoGold(start, start, end, "month", text, candidate.pattern, candidate.confidence)

    if candidate.pattern == "zh_quarter":
        m = re.search(r"(19\d{2}|20\d{2})年?第?([一二三四1-4])季度", text)
        if not m:
            return None
        dt_value = normalize_quarter(m.group(2), m.group(1))
        if dt_value is None:
            return None
        start = dt_value
        end_month = start.month + 2
        end = datetime(start.year, end_month, month_last_day(start.year, end_month))
        return PseudoGold(start, start, end, "month", text, candidate.pattern, candidate.confidence)

    if candidate.pattern == "en_quarter":
        m = re.search(r"Q([1-4])\s+(19\d{2}|20\d{2})", text, flags=re.IGNORECASE)
        if m:
            dt_value = normalize_quarter(m.group(1), m.group(2))
        else:
            m = re.search(r"(19\d{2}|20\d{2})\s+Q([1-4])", text, flags=re.IGNORECASE)
            if not m:
                return None
            dt_value = normalize_quarter(m.group(2), m.group(1))
        if dt_value is None:
            return None
        start = dt_value
        end_month = start.month + 2
        end = datetime(start.year, end_month, month_last_day(start.year, end_month))
        return PseudoGold(start, start, end, "month", text, candidate.pattern, candidate.confidence)

    if candidate.pattern == "year_context":
        m = re.search(r"(19\d{2}|20\d{2})", text)
        if not m:
            return None
        year = int(m.group(1))
        dt_value = datetime(year, 1, 1)
        start, end = range_by_granularity(dt_value, "year")
        return PseudoGold(dt_value, start, end, "year", text, candidate.pattern, candidate.confidence)

    parsed = parse_candidate_text(text, base_time, lang, title)
    if not parsed:
        return None

    normalized = normalize_by_granularity(parsed, candidate.granularity)

    if normalized.year < 1900 or normalized.year > 2100:
        return None

    start, end = range_by_granularity(normalized, candidate.granularity)
    return PseudoGold(normalized, start, end, candidate.granularity, text, candidate.pattern, candidate.confidence)


def extract_pseudo_gold(title: str, base_time: datetime, lang: str, mode: str = "balanced") -> Optional[PseudoGold]:
    mode = (mode or "balanced").lower()
    if mode not in {"strict", "balanced", "loose"}:
        mode = "balanced"

    if not title or not base_time:
        return None

    candidates = extract_regex_candidates(title, mode)
    search_candidate = extract_search_dates_candidate(title, base_time, lang, mode)
    if search_candidate:
        candidates.append(search_candidate)

    if not candidates:
        return None

    candidates.sort(key=candidate_sort_key)

    for candidate in candidates:
        gold = parse_candidate_to_gold(candidate, title, base_time, lang)
        if gold:
            return gold

    return None


def same_by_granularity(event_time: datetime, gold: PseudoGold) -> bool:
    if gold.granularity == "day":
        return event_time.date() == gold.value.date()
    if gold.granularity == "month":
        return event_time.year == gold.value.year and event_time.month == gold.value.month
    if gold.granularity == "year":
        return event_time.year == gold.value.year
    return False


def within_days(event_time: datetime, gold: PseudoGold, max_days: int) -> bool:
    if gold.granularity != "day":
        return same_by_granularity(event_time, gold)
    return abs((event_time.date() - gold.value.date()).days) <= max_days


def build_strata_key(record: Dict, mode: str) -> str:
    lang = record["lang"]
    parse_mode = record["parse_mode"]
    if mode == "language":
        return lang
    return f"{lang}__{parse_mode}"


def stratified_sample(records: List[Dict], sample_size: int, mode: str, seed: int) -> List[Dict]:
    if sample_size >= len(records):
        return list(records)

    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        groups[build_strata_key(r, mode)].append(r)

    group_items = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
    rng = random.Random(seed)

    target = min(sample_size, len(records))
    allocation = {k: 0 for k in groups}

    for idx, (group_key, _) in enumerate(group_items):
        if idx >= target:
            break
        allocation[group_key] = 1

    used = sum(allocation.values())
    remaining = target - used

    if remaining > 0:
        capacities = {k: len(v) - allocation[k] for k, v in groups.items()}
        total_capacity = sum(max(c, 0) for c in capacities.values())

        if total_capacity > 0:
            fractional = []
            for k, cap in capacities.items():
                if cap <= 0:
                    continue
                exact = remaining * (cap / total_capacity)
                add = int(math.floor(exact))
                add = min(add, cap)
                allocation[k] += add
                fractional.append((exact - add, k))

            used_after_floor = sum(allocation.values())
            left = target - used_after_floor

            if left > 0:
                fractional.sort(reverse=True)
                for _, k in fractional:
                    if left <= 0:
                        break
                    if allocation[k] < len(groups[k]):
                        allocation[k] += 1
                        left -= 1

    sampled: List[Dict] = []
    for group_key, rows in groups.items():
        n = min(allocation[group_key], len(rows))
        if n <= 0:
            continue
        sampled.extend(rng.sample(rows, n))

    return sampled


def evaluate_sample(sampled_records: List[Dict], day_tolerance: int, gold_mode: str = "balanced"):
    detail_rows = []

    total = len(sampled_records)
    evaluable = 0
    exact_hit = 0
    relaxed_hit = 0
    range_hit = 0
    anchor_in_gold_hit = 0
    interval_evaluable = 0
    interval_range_hit = 0

    pattern_counter = Counter()
    confidence_counter = Counter()

    per_lang = defaultdict(
        lambda: {
            "sample_count": 0,
            "evaluable_count": 0,
            "exact_hit": 0,
            "relaxed_hit": 0,
            "range_hit": 0,
            "anchor_in_gold_hit": 0,
            "interval_evaluable": 0,
            "interval_range_hit": 0,
        }
    )

    for r in sampled_records:
        lang = r["lang"]
        title = r["title"]
        standard_ts = r["standard_timestamp"]
        event_ts = r["event_timestamp"]
        event_start = r.get("event_time_start")
        event_end = r.get("event_time_end")
        time_granularity = r.get("time_granularity") or ""

        per_lang[lang]["sample_count"] += 1

        gold = extract_pseudo_gold(title, standard_ts, lang, mode=gold_mode)

        is_evaluable = gold is not None and event_ts is not None
        is_exact = False
        is_relaxed = False
        is_range_hit = False
        is_anchor_in_gold = False
        is_interval_gold = False

        if is_evaluable:
            evaluable += 1
            per_lang[lang]["evaluable_count"] += 1
            is_exact = same_by_granularity(event_ts, gold)
            is_relaxed = within_days(event_ts, gold, day_tolerance)
            normalized_event_start, normalized_event_end = normalize_range(
                event_start if event_start is not None else event_ts,
                event_end if event_end is not None else event_ts,
            )
            gold_start, gold_end = normalize_range(gold.range_start, gold.range_end)
            if normalized_event_start and normalized_event_end and gold_start and gold_end:
                is_range_hit = ranges_overlap(normalized_event_start, normalized_event_end, gold_start, gold_end)
                is_anchor_in_gold = gold_start <= day_floor(event_ts) <= gold_end
                is_interval_gold = gold_start != gold_end
                if is_range_hit:
                    range_hit += 1
                    per_lang[lang]["range_hit"] += 1
                if is_anchor_in_gold:
                    anchor_in_gold_hit += 1
                    per_lang[lang]["anchor_in_gold_hit"] += 1
                if is_interval_gold:
                    interval_evaluable += 1
                    per_lang[lang]["interval_evaluable"] += 1
                    if is_range_hit:
                        interval_range_hit += 1
                        per_lang[lang]["interval_range_hit"] += 1
            pattern_counter[gold.pattern] += 1
            confidence_counter[gold.confidence] += 1

            if is_exact:
                exact_hit += 1
                per_lang[lang]["exact_hit"] += 1
            if is_relaxed:
                relaxed_hit += 1
                per_lang[lang]["relaxed_hit"] += 1

        detail_rows.append(
            {
                "id": r["id"],
                "lang": lang,
                "parse_mode": r["parse_mode"],
                "title": title,
                "standard_timestamp": standard_ts.strftime("%Y-%m-%d %H:%M:%S") if standard_ts else "",
                "event_timestamp": event_ts.strftime("%Y-%m-%d %H:%M:%S") if event_ts else "",
                "event_time_start": event_start.strftime("%Y-%m-%d %H:%M:%S") if event_start else "",
                "event_time_end": event_end.strftime("%Y-%m-%d %H:%M:%S") if event_end else "",
                "time_granularity": time_granularity,
                "pseudo_time": gold.value.strftime("%Y-%m-%d %H:%M:%S") if gold else "",
                "pseudo_range_start": gold.range_start.strftime("%Y-%m-%d %H:%M:%S") if gold else "",
                "pseudo_range_end": gold.range_end.strftime("%Y-%m-%d %H:%M:%S") if gold else "",
                "pseudo_granularity": gold.granularity if gold else "",
                "pseudo_source_text": gold.source_text if gold else "",
                "pseudo_pattern": gold.pattern if gold else "",
                "pseudo_confidence": gold.confidence if gold else "",
                "is_evaluable": int(is_evaluable),
                "is_exact": int(is_exact),
                "is_relaxed": int(is_relaxed),
                "is_range_hit": int(is_range_hit),
                "is_anchor_in_gold_range": int(is_anchor_in_gold),
                "is_interval_gold": int(is_interval_gold),
            }
        )

    summary = {
        "sample_count": total,
        "evaluable_count": evaluable,
        "evaluable_coverage": round(evaluable / total, 4) if total else 0.0,
        "exact_accuracy": round(exact_hit / evaluable, 4) if evaluable else None,
        "relaxed_accuracy": round(relaxed_hit / evaluable, 4) if evaluable else None,
        "range_hit_accuracy": round(range_hit / evaluable, 4) if evaluable else None,
        "anchor_in_gold_range_accuracy": round(anchor_in_gold_hit / evaluable, 4) if evaluable else None,
        "interval_evaluable_count": interval_evaluable,
        "interval_range_hit_accuracy": round(interval_range_hit / interval_evaluable, 4) if interval_evaluable else None,
        "day_tolerance": day_tolerance,
        "gold_mode": gold_mode,
        "pseudo_pattern_distribution": dict(pattern_counter.most_common()),
        "pseudo_confidence_distribution": dict(confidence_counter.most_common()),
    }

    lang_summary = {}
    for lang, stats in sorted(per_lang.items(), key=lambda item: item[1]["sample_count"], reverse=True):
        evaluable_count = stats["evaluable_count"]
        lang_summary[lang] = {
            "sample_count": stats["sample_count"],
            "evaluable_count": evaluable_count,
            "evaluable_coverage": round(evaluable_count / stats["sample_count"], 4) if stats["sample_count"] else 0.0,
            "exact_accuracy": round(stats["exact_hit"] / evaluable_count, 4) if evaluable_count else None,
            "relaxed_accuracy": round(stats["relaxed_hit"] / evaluable_count, 4) if evaluable_count else None,
            "range_hit_accuracy": round(stats["range_hit"] / evaluable_count, 4) if evaluable_count else None,
            "anchor_in_gold_range_accuracy": round(stats["anchor_in_gold_hit"] / evaluable_count, 4) if evaluable_count else None,
            "interval_evaluable_count": stats["interval_evaluable"],
            "interval_range_hit_accuracy": round(stats["interval_range_hit"] / stats["interval_evaluable"], 4) if stats["interval_evaluable"] else None,
        }

    return summary, detail_rows, lang_summary


def fetch_parser_newsdata(db_config: Dict, table_name: str, limit: Optional[int]) -> List[Dict]:
    if not TABLE_NAME_PATTERN.fullmatch(table_name):
        raise ValueError(f"非法表名: {table_name}")

    conn = get_db_connection(db_config)
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"SHOW COLUMNS FROM {table_name}")
            table_columns = {row["Field"] for row in cursor.fetchall()}

            select_cols = ["id", "title", "standard_timestamp", "event_timestamp"]
            if "event_time_start" in table_columns:
                select_cols.append("event_time_start")
            if "event_time_end" in table_columns:
                select_cols.append("event_time_end")
            if "time_granularity" in table_columns:
                select_cols.append("time_granularity")
            if "parse_mode" in table_columns:
                select_cols.append("parse_mode")

            sql = f"""
                SELECT {", ".join(select_cols)}
                FROM {table_name}
                WHERE title IS NOT NULL AND title != ''
                  AND standard_timestamp IS NOT NULL
            """
            if limit and limit > 0:
                sql += f" LIMIT {int(limit)}"
            cursor.execute(sql)
            rows = cursor.fetchall()

        records = []
        for row in rows:
            title = row["title"]
            standard_ts = normalize_dt(row["standard_timestamp"])
            event_ts = normalize_dt(row["event_timestamp"])
            event_start = normalize_dt(row.get("event_time_start")) if "event_time_start" in row else None
            event_end = normalize_dt(row.get("event_time_end")) if "event_time_end" in row else None
            if standard_ts is None:
                continue

            if event_ts is not None:
                if event_start is None:
                    event_start = event_ts
                if event_end is None:
                    event_end = event_ts
            event_start, event_end = normalize_range(event_start, event_end)

            lang = detect_lang(title)
            parse_mode = row.get("parse_mode") if "parse_mode" in row else None
            if not parse_mode:
                parse_mode = "missing_event"
                if event_ts is not None:
                    parse_mode = "fallback_like" if event_ts == standard_ts else "parsed_like"

            records.append(
                {
                    "id": row["id"],
                    "title": title,
                    "standard_timestamp": standard_ts,
                    "event_timestamp": event_ts,
                    "event_time_start": event_start,
                    "event_time_end": event_end,
                    "time_granularity": row.get("time_granularity", "") if "time_granularity" in row else "",
                    "lang": lang,
                    "parse_mode": parse_mode,
                }
            )

        return records
    finally:
        conn.close()


def save_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="分层抽样 + 弱监督时间准确率评估（基于 parser_newsdata）")
    parser.add_argument("--db-host", default=DB_CONFIG["host"], help="MySQL host")
    parser.add_argument("--db-port", type=int, default=DB_CONFIG["port"], help="MySQL port")
    parser.add_argument("--db-user", default=DB_CONFIG["user"], help="MySQL user")
    parser.add_argument("--db-password", default=DB_CONFIG["password"], help="MySQL password")
    parser.add_argument("--db-name", default=DB_CONFIG["database"], help="MySQL database")
    parser.add_argument("--table", default="parser_newsdata", help="目标表名，默认 parser_newsdata")
    parser.add_argument("--sample-size", type=int, default=1200, help="分层抽样样本量")
    parser.add_argument("--strata-mode", choices=["language", "language_parse_mode"], default="language_parse_mode", help="分层维度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--day-tolerance", type=int, default=1, help="日粒度 relaxed 允许误差天数")
    parser.add_argument("--gold-mode", choices=["strict", "balanced", "loose"], default="balanced", help="伪标注规则强度")
    parser.add_argument("--max-rows", type=int, default=0, help="仅调试用：限制读取行数，0 表示全量")
    parser.add_argument("--output-dir", default="code/script/reports", help="输出目录")
    args = parser.parse_args()

    max_rows = args.max_rows if args.max_rows > 0 else None
    db_config = {
        "host": args.db_host,
        "port": args.db_port,
        "user": args.db_user,
        "password": args.db_password,
        "database": args.db_name,
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor,
    }

    print("开始读取 parser_newsdata...")
    records = fetch_parser_newsdata(db_config, args.table, max_rows)
    if not records:
        raise RuntimeError("未读取到可用数据，请检查表名和筛选条件。")

    print(f"读取完成：{len(records)} 条。开始分层抽样...")
    sampled = stratified_sample(records, args.sample_size, args.strata_mode, args.seed)
    print(f"抽样完成：{len(sampled)} 条。开始自动评估...")

    summary, detail_rows, lang_summary = evaluate_sample(sampled, args.day_tolerance, gold_mode=args.gold_mode)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = out_dir / f"event_time_eval_summary_{ts}.json"
    detail_path = out_dir / f"event_time_eval_detail_{ts}.csv"

    payload = {
        "config": {
            "table": args.table,
            "sample_size": args.sample_size,
            "strata_mode": args.strata_mode,
            "seed": args.seed,
            "day_tolerance": args.day_tolerance,
            "gold_mode": args.gold_mode,
            "max_rows": args.max_rows,
        },
        "overall": summary,
        "by_language": lang_summary,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    save_csv(detail_path, detail_rows)

    print("\n===== 自动评估完成 =====")
    print(json.dumps(payload["overall"], ensure_ascii=False, indent=2))
    print(f"按语种统计已写入: {summary_path}")
    print(f"样本明细已写入: {detail_path}")
    print("提示：该结果是弱监督估计，不等价于人工标注的真实准确率。")


if __name__ == "__main__":
    main()
