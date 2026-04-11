import pymysql
import spacy
import dateparser
import re
import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta
from dateparser.search import search_dates
from langdetect import detect, DetectorFactory

# 保证每次语言检测结果的一致性
DetectorFactory.seed = 0

# ==========================================
# 1. 数据库配置
# ==========================================
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '123456', 
    'database': 'news_db',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor 
}

# ==========================================
# 2. 预加载第一梯队 NLP 模型 (驻留内存)
# ==========================================
print("正在加载主要语言 spaCy 模型...")
MODEL_NAME_MAP = {
    'en': "en_core_web_sm",
    'zh-cn': "zh_core_web_sm",
    'zh-tw': "zh_core_web_sm", # 繁体中文复用简体模型进行实体识别
    'es': "es_core_news_sm",
    'fr': "fr_core_news_sm",
    'ru': "ru_core_news_sm",
    'ko': "ko_core_news_sm",
    'uk': "uk_core_news_sm"
}


def load_spacy_models():
    loaded_models = {}
    for lang, model_name in MODEL_NAME_MAP.items():
        try:
            loaded_models[lang] = spacy.load(model_name)
        except Exception as model_err:
            print(f"[警告] spaCy 模型加载失败 ({lang} -> {model_name}): {model_err}")
    return loaded_models


nlp_models = load_spacy_models()
print("模型加载完毕，小语种将自动执行安全降级。")

DATEPARSER_LANGUAGE_MAP = {
    'en': ['en'],
    'zh-cn': ['zh'],
    'zh-tw': ['zh'],
    'zh': ['zh'],
    'es': ['es'],
    'fr': ['fr'],
    'ru': ['ru'],
    'ko': ['ko'],
    'uk': ['uk'],
    'sw': ['sw']
}

MONTH_NAME_PATTERN = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
MONTH_NAME_TO_NUM = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
FULLWIDTH_TRANSLATION = str.maketrans({
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "／": "/", "－": "-", "：": ":", "，": ",", "．": ".", "～": "~",
    "（": "(", "）": ")", "　": " ",
})
YEAR_CONTEXT_WORD_PATTERN = re.compile(
    r"\b(in|by|during|since|from|until|before|after|fiscal|fy|season|for|to)\b|(?:于|在|截至|到|自|至)",
    flags=re.IGNORECASE
)
QUARTER_HALF_PATTERN = re.compile(
    r"(?:Q[1-4]|(?:19\d{2}|20\d{2})\s*Q[1-4]|第?[一二三四1-4]季度|前[一二三四1-4]季度|上半年|下半年|前\d{1,2}月|\d{1,2}\s*[-~到至]\s*\d{1,2}月|H[12]\s+(?:19\d{2}|20\d{2})|(?:first|second)\s+half(?:\s+of)?\s+(?:19\d{2}|20\d{2}))",
    flags=re.IGNORECASE
)
EN_MONTH_ONLY_WITH_CONTEXT_PATTERN = re.compile(
    rf"\b(?:in|during|for|by|this|last|next|on)\s+({MONTH_NAME_PATTERN})\b",
    flags=re.IGNORECASE
)
DATE_SIGNAL_PATTERN = re.compile(
    rf"(\d{{4}}[年/\-.]\d{{1,2}}([月/\-.]\d{{1,2}})?|\d{{1,2}}月\d{{1,2}}[日号]?|"
    rf"\b{MONTH_NAME_PATTERN}\b|"
    r"(今天|今日|昨天|昨日|昨晚|明天|明日|后天|今年|去年|明年|上周|下周|本周|上月|下月|本月|上半年|下半年|前[一二三四1-4]季度|前\d{1,2}月|"
    r"today|yesterday|tomorrow|last|next|this\s+(week|month|year)))",
    flags=re.IGNORECASE
)

MONEY_PATTERN = re.compile(
    r"([$¥￥€£₹]|USD|CNY|RMB|HKD|S\$|元|万元|亿元|万亿|美元|欧元|英镑|港元|million|billion|trillion)",
    flags=re.IGNORECASE
)
PERCENT_PATTERN = re.compile(r"\d+(?:\s*[.,]\s*\d+)?\s*%")
PURE_NUMBER_PATTERN = re.compile(r"^\d+(?:[.,]\d+)*$")

FUTURE_HINT_PATTERN = re.compile(
    r"(将于|将会|即将|计划于|预计|拟于|to be held|will|upcoming|is set to|next)",
    flags=re.IGNORECASE
)
PAST_HINT_PATTERN = re.compile(
    r"(此前|去年|上月|昨日|昨天|last|previous|ago|earlier)",
    flags=re.IGNORECASE
)
SOURCE_TAIL_MARKER_PATTERN = re.compile(
    r"(?:DW|Reuters|Xinhua|BBC|AP|AFP|新华网|中新社|路透|彭博)",
    flags=re.IGNORECASE
)
DEADLINE_HINT_PATTERN = re.compile(r"\b(by|until|till|before|through)\b|(?:截至|到|至)", flags=re.IGNORECASE)
START_HINT_PATTERN = re.compile(r"\b(from|since|starting|start)\b|(?:自|从)", flags=re.IGNORECASE)


@dataclass
class ParsedEventTime:
    anchor: datetime
    start: datetime
    end: datetime
    granularity: str
    candidate_text: str


def normalize_datetime_text(text):
    if text is None:
        return ""
    return text.translate(FULLWIDTH_TRANSLATION).strip()


def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

def normalize_base_time(base_time):
    if isinstance(base_time, datetime):
        return base_time

    if isinstance(base_time, str):
        base_time = base_time.strip()
        if not base_time:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(base_time, fmt)
            except ValueError:
                continue
        try:
            return dateparser.parse(base_time)
        except Exception:
            return None

    return None

def detect_prefer_dates_from(title):
    if FUTURE_HINT_PATTERN.search(title):
        return "future"
    if PAST_HINT_PATTERN.search(title):
        return "past"
    return "current_period"

def day_start(dt_obj):
    return dt_obj.replace(hour=0, minute=0, second=0, microsecond=0)

def month_last_day(year, month):
    return calendar.monthrange(year, month)[1]

def choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text=""):
    start_dt = day_start(start_dt)
    end_dt = day_start(end_dt)
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt

    prefer = detect_prefer_dates_from(title)
    candidate_text = (candidate_text or "").strip()
    if DEADLINE_HINT_PATTERN.search(candidate_text):
        return end_dt
    if START_HINT_PATTERN.search(candidate_text):
        return start_dt

    base_day = day_start(base_time)
    if start_dt <= base_day <= end_dt:
        return base_day
    if prefer == "future" and base_day < start_dt:
        return start_dt
    if prefer == "past" and base_day > end_dt:
        return end_dt

    return start_dt if abs((base_day - start_dt).days) <= abs((base_day - end_dt).days) else end_dt

def quarter_range(year, quarter):
    start_month = (quarter - 1) * 3 + 1
    end_month = start_month + 2
    start_dt = datetime(year, start_month, 1)
    end_dt = datetime(year, end_month, month_last_day(year, end_month))
    return start_dt, end_dt

def build_result(anchor_dt, start_dt, end_dt, granularity, candidate_text, base_time, title):
    anchor = day_start(anchor_dt)
    start = day_start(start_dt)
    end = day_start(end_dt)
    if end < start:
        start, end = end, start

    if start <= anchor <= end:
        final_anchor = anchor
    else:
        final_anchor = choose_anchor_from_range(start, end, base_time, title, candidate_text)

    return ParsedEventTime(
        anchor=final_anchor,
        start=start,
        end=end,
        granularity=granularity,
        candidate_text=candidate_text,
    )

def infer_range_from_candidate(candidate_text, anchor_dt, base_time, title):
    text = normalize_datetime_text(candidate_text)
    year_fallback = anchor_dt.year

    range_match = re.search(r"(?:(19\d{2}|20\d{2})年)?(\d{1,2})\s*[-~到至]\s*(\d{1,2})月", text)
    if range_match:
        year = int(range_match.group(1)) if range_match.group(1) else year_fallback
        m1 = int(range_match.group(2))
        m2 = int(range_match.group(3))
        if 1 <= m1 <= 12 and 1 <= m2 <= 12:
            start_month, end_month = sorted([m1, m2])
            return datetime(year, start_month, 1), datetime(year, end_month, month_last_day(year, end_month)), "month_span"

    prefix_month_match = re.search(r"(?:(19\d{2}|20\d{2})年)?前(\d{1,2})月", text)
    if prefix_month_match:
        year = int(prefix_month_match.group(1)) if prefix_month_match.group(1) else year_fallback
        end_month = int(prefix_month_match.group(2))
        if 1 <= end_month <= 12:
            return datetime(year, 1, 1), datetime(year, end_month, month_last_day(year, end_month)), "month_span"

    quarter_prefix = re.search(r"(?:(19\d{2}|20\d{2})年)?前([一二三四1-4])季度", text)
    if quarter_prefix:
        year = int(quarter_prefix.group(1)) if quarter_prefix.group(1) else year_fallback
        q = parse_quarter_token(quarter_prefix.group(2))
        if q is not None:
            end_month = q * 3
            return datetime(year, 1, 1), datetime(year, end_month, month_last_day(year, end_month)), "quarter_span"

    quarter_zh = re.search(r"(?:(19\d{2}|20\d{2})年)?第?([一二三四1-4])季度", text)
    if quarter_zh:
        year = int(quarter_zh.group(1)) if quarter_zh.group(1) else year_fallback
        q = parse_quarter_token(quarter_zh.group(2))
        if q is not None:
            start_dt, end_dt = quarter_range(year, q)
            return start_dt, end_dt, "quarter"

    quarter_en = re.search(r"\bQ([1-4])\s*(?:FY)?\s*(19\d{2}|20\d{2})\b", text, flags=re.IGNORECASE)
    if quarter_en:
        q = int(quarter_en.group(1))
        year = int(quarter_en.group(2))
        start_dt, end_dt = quarter_range(year, q)
        return start_dt, end_dt, "quarter"

    quarter_en_reverse = re.search(r"\b(19\d{2}|20\d{2})\s*Q([1-4])\b", text, flags=re.IGNORECASE)
    if quarter_en_reverse:
        year = int(quarter_en_reverse.group(1))
        q = int(quarter_en_reverse.group(2))
        start_dt, end_dt = quarter_range(year, q)
        return start_dt, end_dt, "quarter"

    half_zh = re.search(r"(?:(19\d{2}|20\d{2})年)?(上半年|下半年)", text)
    if half_zh:
        year = int(half_zh.group(1)) if half_zh.group(1) else year_fallback
        if half_zh.group(2) == "上半年":
            return datetime(year, 1, 1), datetime(year, 6, 30), "half"
        return datetime(year, 7, 1), datetime(year, 12, 31), "half"

    half_en = re.search(r"\bH([12])\s+(19\d{2}|20\d{2})\b", text, flags=re.IGNORECASE)
    if half_en:
        half = int(half_en.group(1))
        year = int(half_en.group(2))
        if half == 1:
            return datetime(year, 1, 1), datetime(year, 6, 30), "half"
        return datetime(year, 7, 1), datetime(year, 12, 31), "half"

    half_en_words = re.search(r"\b(first|second)\s+half(?:\s+of)?\s+(19\d{2}|20\d{2})\b", text, flags=re.IGNORECASE)
    if half_en_words:
        year = int(half_en_words.group(2))
        if half_en_words.group(1).lower() == "first":
            return datetime(year, 1, 1), datetime(year, 6, 30), "half"
        return datetime(year, 7, 1), datetime(year, 12, 31), "half"

    year_range = re.search(r"(?<!\d)(19\d{2}|20\d{2})\s*(?:/|-|~|到|至)\s*(\d{2,4})(?!\d)", text)
    if year_range and not re.search(r"\d{1,2}\s*(?:/|-|~|到|至)\s*\d{1,2}月", text):
        start_year = int(year_range.group(1))
        end_token = year_range.group(2)
        if len(end_token) == 2:
            century = start_year // 100
            end_year = century * 100 + int(end_token)
            if end_year < start_year:
                end_year += 100
        else:
            end_year = int(end_token)
        return datetime(start_year, 1, 1), datetime(end_year, 12, 31), "year_span"

    if re.search(r"\d{4}年\d{1,2}月(?!\d{1,2}[日号]?)", text):
        m = re.search(r"(19\d{2}|20\d{2})年(\d{1,2})月", text)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            if 1 <= month <= 12:
                return datetime(year, month, 1), datetime(year, month, month_last_day(year, month)), "month"

    en_my = re.search(rf"\b({MONTH_NAME_PATTERN})\s+(19\d{{2}}|20\d{{2}})\b", text, flags=re.IGNORECASE)
    if en_my and not re.search(rf"\b{MONTH_NAME_PATTERN}\s+\d{{1,2}},?\s+(19\d{{2}}|20\d{{2}})\b", text, flags=re.IGNORECASE):
        month = MONTH_NAME_TO_NUM.get(en_my.group(1).lower())
        year = int(en_my.group(2))
        if month:
            return datetime(year, month, 1), datetime(year, month, month_last_day(year, month)), "month"

    if EN_MONTH_ONLY_WITH_CONTEXT_PATTERN.search(text) or re.search(r"(?<!\d)\d{1,2}月(?!\d)", text):
        return datetime(anchor_dt.year, anchor_dt.month, 1), datetime(anchor_dt.year, anchor_dt.month, month_last_day(anchor_dt.year, anchor_dt.month)), "month"

    is_year_only = (
        re.fullmatch(r"(?:于|在|截至|到|自|至)?\s*(19\d{2}|20\d{2})年?", text) is not None
        or re.fullmatch(r"(19\d{2}|20\d{2})", text) is not None
        or re.search(r"\b(?:in|by|during|since|from|until|fiscal|fy|for)\s+(19\d{2}|20\d{2})(?:/\d{2,4})?\b", text, flags=re.IGNORECASE) is not None
    )
    if is_year_only and not re.search(r"\d{1,2}\s*[-/.]\s*\d{1,2}", text) and not re.search(r"\d{1,2}月", text):
        year_match = re.search(r"(19\d{2}|20\d{2})", text)
        if year_match:
            year = int(year_match.group(1))
            return datetime(year, 1, 1), datetime(year, 12, 31), "year"

    if re.search(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}", text) or re.search(r"\d{4}\.\d{1,2}\.\d{1,2}", text):
        point = day_start(anchor_dt)
        return point, point, "day"
    if re.search(r"\d{1,2}月\d{1,2}[日号]?", text):
        point = day_start(anchor_dt)
        return point, point, "day"
    if re.search(rf"\b{MONTH_NAME_PATTERN}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*(19\d{{2}}|20\d{{2}}))?\b", text, flags=re.IGNORECASE):
        point = day_start(anchor_dt)
        return point, point, "day"
    if re.search(r"\b(today|yesterday|tomorrow)\b|今天|昨日|昨天|明天|后天|前天", text, flags=re.IGNORECASE):
        point = day_start(anchor_dt)
        return point, point, "day"

    point = day_start(anchor_dt)
    return point, point, "day"

def to_parsed_event(candidate_text, parsed_dt, base_time, title):
    start_dt, end_dt, granularity = infer_range_from_candidate(candidate_text, parsed_dt, base_time, title)
    return build_result(parsed_dt, start_dt, end_dt, granularity, candidate_text, base_time, title)

def parse_quarter_token(token):
    if token is None:
        return None
    token = token.strip()
    q_map = {"一": 1, "二": 2, "三": 3, "四": 4}
    if token in q_map:
        return q_map[token]
    if token.isdigit():
        q = int(token)
        if 1 <= q <= 4:
            return q
    return None

def has_date_signal(text):
    return DATE_SIGNAL_PATTERN.search(text) is not None

def is_noise_candidate(text):
    text = normalize_datetime_text(text)
    if not text:
        return True

    normalized = re.sub(r"\s+", "", text)
    if not normalized:
        return True

    if re.search(r"\b\d{4}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{1,2}\b", text):
        return False
    if re.search(r"\b\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*(19\d{2}|20\d{2})\b", text):
        return False
    if re.search(rf"\b\d{{1,2}}\s+{MONTH_NAME_PATTERN}\s+(19\d{{2}}|20\d{{2}})\b", text, flags=re.IGNORECASE):
        return False

    if re.fullmatch(r"第[一二三四五六七八九十百千万\d]+届", normalized):
        return True

    if not has_date_signal(text):
        if PERCENT_PATTERN.search(text):
            return True
        if MONEY_PATTERN.search(text):
            return True
        if PURE_NUMBER_PATTERN.fullmatch(normalized):
            try:
                year = int(float(normalized.replace(",", "")))
                return year < 1900 or year > 2100
            except Exception:
                return True

    if PURE_NUMBER_PATTERN.fullmatch(normalized):
        try:
            year = int(float(normalized.replace(",", "")))
            if year < 1900 or year > 2100:
                return True
        except Exception:
            return True

    return False

def is_likely_source_tail_date(title, start_idx):
    if not title:
        return False
    if start_idx < int(len(title) * 0.6):
        return False

    left = max(0, start_idx - 24)
    prefix = title[left:start_idx]
    if SOURCE_TAIL_MARKER_PATTERN.search(prefix) and re.search(r"[-–—|]", prefix):
        return True
    if re.search(r"(?:DW|Reuters|Xinhua|BBC|AP|AFP)\s*[-–—]\s*$", prefix, flags=re.IGNORECASE):
        return True
    return False

def has_prior_date_signal(title, start_idx):
    if not title or start_idx <= 0:
        return False
    head = title[:max(0, start_idx - 1)]
    return re.search(
        rf"(19\d{{2}}|20\d{{2}}|\d{{1,2}}月|Q[1-4]|{MONTH_NAME_PATTERN}|上半年|下半年)",
        head,
        flags=re.IGNORECASE
    ) is not None

def candidate_score(text, start_idx, title=""):
    text = normalize_datetime_text(text)
    score = 0

    if re.search(r"\d{4}年\d{1,2}月\d{1,2}[日号]?", text) or re.search(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", text):
        score += 60
    elif re.search(r"\b\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*(19\d{2}|20\d{2})\b", text):
        score += 56
    elif re.search(rf"\b\d{{1,2}}\s+{MONTH_NAME_PATTERN}\s+(19\d{{2}}|20\d{{2}})\b", text, flags=re.IGNORECASE):
        score += 56
    elif re.search(rf"\b{MONTH_NAME_PATTERN}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*(19\d{{2}}|20\d{{2}}))?\b", text, flags=re.IGNORECASE):
        score += 55
    elif re.fullmatch(r"(19\d{2}|20\d{2})", text):
        score += 28
    elif re.search(r"\d{1,2}月\d{1,2}[日号]?", text):
        score += 50
    elif QUARTER_HALF_PATTERN.search(text):
        score += 45
    elif re.search(r"\d{4}年\d{1,2}月", text) or re.search(rf"\b{MONTH_NAME_PATTERN}\s+\d{{4}}\b", text, flags=re.IGNORECASE):
        score += 40
    elif re.search(r"\b(?:in|by|during|since|from|until|fiscal|fy|for)\s+(19\d{2}|20\d{2})(?:/\d{2,4})?\b", text, flags=re.IGNORECASE) or re.search(r"(?:于|在|截至|到|自|至)\s*(19\d{2}|20\d{2})年?", text):
        score += 35
    elif EN_MONTH_ONLY_WITH_CONTEXT_PATTERN.search(text) or re.search(r"(?<!\d)\d{1,2}月(?!\d)", text):
        score += 30
    elif has_date_signal(text):
        score += 30

    if FUTURE_HINT_PATTERN.search(text) or PAST_HINT_PATTERN.search(text):
        score += 10

    # 规避 "may" 作为情态动词造成的误识别
    if text.lower() == "may":
        score -= 40
    if re.search(r"\bthe\s+1975\b", text, flags=re.IGNORECASE):
        score -= 40

    if is_noise_candidate(text):
        score -= 100
    if is_likely_source_tail_date(title, start_idx):
        score -= 45 if has_prior_date_signal(title, start_idx) else 18

    score -= min(start_idx, 100) // 20
    return score

def extract_regex_candidates(title, lang):
    normalized_title = normalize_datetime_text(title)
    patterns = [
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        r"\d{4}[.]\d{1,2}[.]\d{1,2}",
        r"\b\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*(?:19\d{2}|20\d{2})\b",
        r"\b(?:19\d{2}|20\d{2})\b",
        r"\d{4}年\d{1,2}月\d{1,2}[日号]?",
        r"\d{1,2}月\d{1,2}[日号]?",
        r"\d{4}年\d{1,2}月",
        r"(?<!\d)\d{1,2}月(?!\d)",
        r"(?:(?:于|在|截至|到|自|至)\s*)?(?:19\d{2}|20\d{2})年(?!\d{1,2}月)",
        r"(今天|今日|昨天|昨日|昨晚|明天|明日|后天|今年|去年|明年|上周|下周|本周|上月|下月|本月|上半年|下半年|第?[一二三四1-4]季度|前[一二三四1-4]季度|前\d{1,2}月|\d{1,2}\s*[-~到至]\s*\d{1,2}月)",
        rf"\b{MONTH_NAME_PATTERN}\s+\d{{1,2}},?\s+\d{{4}}\b",
        rf"\b\d{{1,2}}\s+{MONTH_NAME_PATTERN}\s+(?:19\d{{2}}|20\d{{2}})\b",
        rf"\b{MONTH_NAME_PATTERN}\s+\d{{1,2}}(?:st|nd|rd|th)?\b",
        rf"\b{MONTH_NAME_PATTERN}\s+\d{{4}}\b",
        rf"\b(?:in|during|for|by|this|last|next|on)\s+{MONTH_NAME_PATTERN}\b",
        r"\b(?:Q[1-4]\s*(?:FY)?\s*(?:19\d{2}|20\d{2})|(?:19\d{2}|20\d{2})\s*Q[1-4])\b",
        r"\b(?:H[12]\s+(?:19\d{2}|20\d{2})|(?:first|second)\s+half(?:\s+of)?\s+(?:19\d{2}|20\d{2}))\b",
        r"\b(?:in|by|during|since|from|until|fiscal|fy|for)\s+(?:19\d{2}|20\d{2})(?:/\d{2,4})?\b",
        r"\b(?:19\d{2}|20\d{2})\s*/\s*\d{2,4}\b",
        r"\b(today|yesterday|tomorrow|last\s+\w+|next\s+\w+)\b"
    ]

    # 中文标题优先走中文规则；英文标题优先走英文规则
    if lang and lang.startswith("zh"):
        patterns = [patterns[0], patterns[1], patterns[2], patterns[3], patterns[4], patterns[5], patterns[6], patterns[7], patterns[8], patterns[9], patterns[10], patterns[11], patterns[13]]
    elif lang == 'en':
        patterns = [patterns[0], patterns[1], patterns[2], patterns[3], patterns[10], patterns[11], patterns[12], patterns[13], patterns[14], patterns[15], patterns[16], patterns[17], patterns[18], patterns[19]]

    candidates = []
    for pattern in patterns:
        try:
            for match in re.finditer(pattern, normalized_title, flags=re.IGNORECASE):
                span_text = match.group(0).strip()
                if span_text:
                    candidates.append((span_text, match.start()))
        except Exception:
            continue
    return candidates

def collect_time_candidates(title, lang, nlp_model):
    candidates = []

    if nlp_model:
        try:
            doc = nlp_model(title)
            for ent in doc.ents:
                if ent.label_ in ['DATE', 'TIME']:
                    candidates.append((ent.text.strip(), ent.start_char))
        except Exception:
            pass

    candidates.extend(extract_regex_candidates(title, lang))

    dedup = {}
    for text, start_idx in candidates:
        normalized_text = normalize_datetime_text(text)
        key = (re.sub(r"\s+", "", normalized_text), start_idx)
        if key not in dedup:
            dedup[key] = (normalized_text, start_idx)

    ranked = sorted(
        dedup.values(),
        key=lambda item: (-candidate_score(item[0], item[1], title), item[1])
    )
    return ranked

def safe_replace_year(dt_obj, year):
    try:
        return dt_obj.replace(year=year)
    except ValueError:
        return None

def infer_year_for_month(month, base_time, title):
    try:
        parsed = datetime(base_time.year, month, 1)
    except ValueError:
        return None

    prefer = detect_prefer_dates_from(title)
    if prefer == "current_period":
        return base_time.year

    if prefer == "future" and parsed < (base_time - timedelta(days=1)):
        adjusted = safe_replace_year(parsed, parsed.year + 1)
        return adjusted.year if adjusted else parsed.year
    if prefer == "past" and parsed > (base_time + timedelta(days=1)):
        adjusted = safe_replace_year(parsed, parsed.year - 1)
        return adjusted.year if adjusted else parsed.year

    nearby = [parsed]
    prev_year = safe_replace_year(parsed, parsed.year - 1)
    next_year = safe_replace_year(parsed, parsed.year + 1)
    if prev_year:
        nearby.append(prev_year)
    if next_year:
        nearby.append(next_year)
    nearby.sort(key=lambda item: abs(item - base_time))
    return nearby[0].year

def infer_year_for_month_day(month, day, base_time, title):
    try:
        parsed = datetime(base_time.year, month, day)
    except ValueError:
        return None

    prefer = detect_prefer_dates_from(title)
    if prefer == "future" and parsed < (base_time - timedelta(days=1)):
        adjusted = safe_replace_year(parsed, parsed.year + 1)
        return adjusted.year if adjusted else parsed.year
    if prefer == "past" and parsed > (base_time + timedelta(days=1)):
        adjusted = safe_replace_year(parsed, parsed.year - 1)
        return adjusted.year if adjusted else parsed.year

    nearby = [parsed]
    prev_year = safe_replace_year(parsed, parsed.year - 1)
    next_year = safe_replace_year(parsed, parsed.year + 1)
    if prev_year:
        nearby.append(prev_year)
    if next_year:
        nearby.append(next_year)
    nearby.sort(key=lambda item: abs(item - base_time))
    return nearby[0].year

def parse_zh_month_day(candidate_text, base_time, title):
    match = re.search(r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})[日号]?", candidate_text)
    if not match:
        return None

    year_text, month_text, day_text = match.groups()
    month = int(month_text)
    day = int(day_text)
    year = int(year_text) if year_text else base_time.year

    try:
        parsed = datetime(year, month, day)
    except ValueError:
        return None

    if year_text:
        return parsed

    inferred_year = infer_year_for_month_day(month, day, base_time, title)
    if inferred_year is None:
        return parsed

    adjusted = safe_replace_year(parsed, inferred_year)
    return adjusted if adjusted else parsed

def parse_zh_quarter_prefix(candidate_text, base_time, title):
    match = re.search(r"(?:(19\d{2}|20\d{2})年)?前([一二三四1-4])季度", candidate_text)
    if not match:
        return None

    year = int(match.group(1)) if match.group(1) else base_time.year
    quarter = parse_quarter_token(match.group(2))
    if quarter is None:
        return None

    end_month = quarter * 3
    start_dt = datetime(year, 1, 1)
    end_dt = datetime(year, end_month, month_last_day(year, end_month))
    return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)

def parse_zh_month_span(candidate_text, base_time, title):
    # 形如 "1-11月"、"1到11月"
    range_match = re.search(r"(?:(19\d{2}|20\d{2})年)?(\d{1,2})\s*[-~到至]\s*(\d{1,2})月", candidate_text)
    if range_match:
        year = int(range_match.group(1)) if range_match.group(1) else base_time.year
        m1, m2 = int(range_match.group(2)), int(range_match.group(3))
        if not (1 <= m1 <= 12 and 1 <= m2 <= 12):
            return None
        start_month, end_month = sorted([m1, m2])
        start_dt = datetime(year, start_month, 1)
        end_dt = datetime(year, end_month, month_last_day(year, end_month))
        return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)

    # 形如 "前8月"
    prefix_match = re.search(r"(?:(19\d{2}|20\d{2})年)?前(\d{1,2})月", candidate_text)
    if prefix_match:
        year = int(prefix_match.group(1)) if prefix_match.group(1) else base_time.year
        end_month = int(prefix_match.group(2))
        if not (1 <= end_month <= 12):
            return None
        start_dt = datetime(year, 1, 1)
        end_dt = datetime(year, end_month, month_last_day(year, end_month))
        return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)

    return None

def parse_zh_month_only(candidate_text, base_time, title):
    match = re.search(r"(?<!\d)(\d{1,2})月(?!\d)", candidate_text)
    if not match:
        return None

    month = int(match.group(1))
    year = infer_year_for_month(month, base_time, title)
    if year is None:
        return None

    try:
        start_dt = datetime(year, month, 1)
        end_dt = datetime(year, month, month_last_day(year, month))
    except ValueError:
        return None

    return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)

def parse_en_month_day(candidate_text, base_time, title):
    match = re.search(
        rf"\b(?:on|in)?\s*({MONTH_NAME_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*(19\d{{2}}|20\d{{2}}))?\b",
        candidate_text,
        flags=re.IGNORECASE
    )
    if not match:
        return None

    month_name, day_text, year_text = match.groups()
    month = MONTH_NAME_TO_NUM.get(month_name.lower())
    if not month:
        return None

    day = int(day_text)
    year = int(year_text) if year_text else infer_year_for_month_day(month, day, base_time, title)
    if year is None:
        return None

    try:
        return datetime(year, month, day)
    except ValueError:
        return None

def parse_en_day_month_year(candidate_text):
    match = re.search(
        rf"\b(?:mon|tue|wed|thu|fri|sat|sun)?\.?\s*(\d{{1,2}})\s+({MONTH_NAME_PATTERN})\s+(19\d{{2}}|20\d{{2}})\b",
        candidate_text,
        flags=re.IGNORECASE
    )
    if not match:
        return None

    day_text, month_name, year_text = match.groups()
    month = MONTH_NAME_TO_NUM.get(month_name.lower())
    if not month:
        return None

    try:
        return datetime(int(year_text), month, int(day_text))
    except ValueError:
        return None

def parse_numeric_day_month_year(candidate_text, lang):
    match = re.search(r"\b(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(19\d{2}|20\d{2})\b", candidate_text)
    if not match:
        return None

    first, second, year = int(match.group(1)), int(match.group(2)), int(match.group(3))

    if first > 12:
        day, month = first, second
    elif second > 12:
        day, month = second, first
    elif lang == "en":
        month, day = first, second
    else:
        day, month = first, second

    try:
        return datetime(year, month, day)
    except ValueError:
        return None

def parse_quarter_or_half(candidate_text, base_time, title):
    text = candidate_text.strip()

    quarter_en = re.search(r"\bQ([1-4])\s*(?:FY)?\s*(19\d{2}|20\d{2})\b", text, flags=re.IGNORECASE)
    if quarter_en:
        q, year = int(quarter_en.group(1)), int(quarter_en.group(2))
        start_month = (q - 1) * 3 + 1
        end_month = start_month + 2
        start_dt = datetime(year, start_month, 1)
        end_dt = datetime(year, end_month, month_last_day(year, end_month))
        return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)

    quarter_en_reverse = re.search(r"\b(19\d{2}|20\d{2})\s*Q([1-4])\b", text, flags=re.IGNORECASE)
    if quarter_en_reverse:
        year, q = int(quarter_en_reverse.group(1)), int(quarter_en_reverse.group(2))
        start_month = (q - 1) * 3 + 1
        end_month = start_month + 2
        start_dt = datetime(year, start_month, 1)
        end_dt = datetime(year, end_month, month_last_day(year, end_month))
        return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)

    quarter_zh = re.search(r"(?:(19\d{2}|20\d{2})年)?第?([一二三四1-4])季度", text)
    if quarter_zh:
        year = int(quarter_zh.group(1)) if quarter_zh.group(1) else base_time.year
        q = parse_quarter_token(quarter_zh.group(2))
        if q is None:
            return None
        start_month = (q - 1) * 3 + 1
        end_month = start_month + 2
        start_dt = datetime(year, start_month, 1)
        end_dt = datetime(year, end_month, month_last_day(year, end_month))
        return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)

    quarter_zh_prefix = re.search(r"(?:(19\d{2}|20\d{2})年)?前([一二三四1-4])季度", text)
    if quarter_zh_prefix:
        year = int(quarter_zh_prefix.group(1)) if quarter_zh_prefix.group(1) else base_time.year
        q = parse_quarter_token(quarter_zh_prefix.group(2))
        if q is None:
            return None
        end_month = q * 3
        start_dt = datetime(year, 1, 1)
        end_dt = datetime(year, end_month, month_last_day(year, end_month))
        return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)

    half_zh = re.search(r"(?:(19\d{2}|20\d{2})年)?(上半年|下半年)", text)
    if half_zh:
        year = int(half_zh.group(1)) if half_zh.group(1) else base_time.year
        if half_zh.group(2) == "上半年":
            start_dt = datetime(year, 1, 1)
            end_dt = datetime(year, 6, 30)
        else:
            start_dt = datetime(year, 7, 1)
            end_dt = datetime(year, 12, 31)
        return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)

    half_en = re.search(r"\bH([12])\s+(19\d{2}|20\d{2})\b", text, flags=re.IGNORECASE)
    if half_en:
        half, year = int(half_en.group(1)), int(half_en.group(2))
        if half == 1:
            start_dt = datetime(year, 1, 1)
            end_dt = datetime(year, 6, 30)
        else:
            start_dt = datetime(year, 7, 1)
            end_dt = datetime(year, 12, 31)
        return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)

    half_en_words = re.search(r"\b(first|second)\s+half(?:\s+of)?\s+(19\d{2}|20\d{2})\b", text, flags=re.IGNORECASE)
    if half_en_words:
        half_word, year = half_en_words.group(1).lower(), int(half_en_words.group(2))
        if half_word == "first":
            start_dt = datetime(year, 1, 1)
            end_dt = datetime(year, 6, 30)
        else:
            start_dt = datetime(year, 7, 1)
            end_dt = datetime(year, 12, 31)
        return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)

    return None

def parse_year_only(candidate_text, base_time, title):
    text = candidate_text.strip()
    if re.search(r"\d{1,2}\s*[-/.]\s*\d{1,2}", text) or re.search(r"\d{1,2}月\d{1,2}", text):
        return None
    if re.search(rf"\b{MONTH_NAME_PATTERN}\b", text, flags=re.IGNORECASE):
        return None
    if re.search(r"\d{1,2}月", text):
        return None
    if QUARTER_HALF_PATTERN.search(text):
        return None

    range_match = re.search(r"(?<!\d)(19\d{2}|20\d{2})\s*/\s*(\d{2,4})(?!\d)", text)
    if range_match:
        start_year = int(range_match.group(1))
        end_token = range_match.group(2)
        if len(end_token) == 2:
            century = start_year // 100
            end_year = century * 100 + int(end_token)
            if end_year < start_year:
                end_year += 100
        else:
            end_year = int(end_token)

        if abs(start_year - base_time.year) > 4 and abs(end_year - base_time.year) > 4 and YEAR_CONTEXT_WORD_PATTERN.search(text) is None:
            return None
        start_dt = datetime(start_year, 1, 1)
        end_dt = datetime(end_year, 12, 31)
        return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)

    year_match = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", text)
    if not year_match:
        return None

    year = int(year_match.group(1))
    if abs(year - base_time.year) > 4 and YEAR_CONTEXT_WORD_PATTERN.search(text) is None:
        return None

    start_dt = datetime(year, 1, 1)
    end_dt = datetime(year, 12, 31)
    return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)

def normalize_granularity(candidate_text, parsed, base_time, title):
    candidate_text = normalize_datetime_text(candidate_text)

    if re.fullmatch(r"\d{4}年?", candidate_text.strip()):
        start_dt = datetime(parsed.year, 1, 1)
        end_dt = datetime(parsed.year, 12, 31)
        return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)

    if re.search(r"\d{4}年\d{1,2}月(?!\d{1,2}[日号]?)", candidate_text.strip()):
        start_dt = datetime(parsed.year, parsed.month, 1)
        end_dt = datetime(parsed.year, parsed.month, month_last_day(parsed.year, parsed.month))
        return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)

    if re.search(rf"\b{MONTH_NAME_PATTERN}\s+\d{{4}}\b", candidate_text, flags=re.IGNORECASE) and not re.search(
        rf"\b{MONTH_NAME_PATTERN}\s+\d{{1,2}},?\s+\d{{4}}\b", candidate_text, flags=re.IGNORECASE
    ):
        start_dt = datetime(parsed.year, parsed.month, 1)
        end_dt = datetime(parsed.year, parsed.month, month_last_day(parsed.year, parsed.month))
        return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)

    if EN_MONTH_ONLY_WITH_CONTEXT_PATTERN.search(candidate_text) or re.search(r"(?<!\d)\d{1,2}月(?!\d)", candidate_text):
        start_dt = datetime(parsed.year, parsed.month, 1)
        end_dt = datetime(parsed.year, parsed.month, month_last_day(parsed.year, parsed.month))
        return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)

    if QUARTER_HALF_PATTERN.search(candidate_text):
        quarter = re.search(r"Q([1-4])", candidate_text, flags=re.IGNORECASE)
        if quarter:
            q = int(quarter.group(1))
            start_month = (q - 1) * 3 + 1
            end_month = start_month + 2
            start_dt = datetime(parsed.year, start_month, 1)
            end_dt = datetime(parsed.year, end_month, month_last_day(parsed.year, end_month))
            return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)
        if "上半年" in candidate_text or re.search(r"\bH1\b|first\s+half", candidate_text, flags=re.IGNORECASE):
            start_dt = datetime(parsed.year, 1, 1)
            end_dt = datetime(parsed.year, 6, 30)
            return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)
        if "下半年" in candidate_text or re.search(r"\bH2\b|second\s+half", candidate_text, flags=re.IGNORECASE):
            start_dt = datetime(parsed.year, 7, 1)
            end_dt = datetime(parsed.year, 12, 31)
            return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)

    if re.search(r"\b(?:in|by|during|since|from|until|fiscal|fy|for)\s+(19\d{2}|20\d{2})(?:/\d{2,4})?\b", candidate_text, flags=re.IGNORECASE):
        start_dt = datetime(parsed.year, 1, 1)
        end_dt = datetime(parsed.year, 12, 31)
        return choose_anchor_from_range(start_dt, end_dt, base_time, title, candidate_text)

    return parsed

def is_plausible_result(parsed, candidate_text, base_time):
    candidate_text = normalize_datetime_text(candidate_text)
    if parsed.year < 1900 or parsed.year > 2100:
        return False

    explicit_year = re.search(r"(19|20)\d{2}", candidate_text) is not None or "年" in candidate_text
    if not explicit_year and abs(parsed.year - base_time.year) > 2:
        return False
    has_month_or_day_signal = (
        re.search(r"\d{1,2}\s*[-/.]\s*\d{1,2}", candidate_text) is not None
        or re.search(r"\d{1,2}月(?:\d{1,2})?", candidate_text) is not None
        or re.search(rf"\b{MONTH_NAME_PATTERN}\b", candidate_text, flags=re.IGNORECASE) is not None
    )
    if explicit_year and abs(parsed.year - base_time.year) > 4 and YEAR_CONTEXT_WORD_PATTERN.search(candidate_text) is None and not has_month_or_day_signal:
        return False

    return True

def parse_candidate_time(candidate_text, base_time, lang, title):
    candidate_text = normalize_datetime_text(candidate_text)
    if not candidate_text or is_noise_candidate(candidate_text):
        return None
    if candidate_text.lower() == "may":
        return None

    numeric_dmy = parse_numeric_day_month_year(candidate_text, lang)
    if numeric_dmy and is_plausible_result(numeric_dmy, candidate_text, base_time):
        return to_parsed_event(candidate_text, numeric_dmy, base_time, title)

    en_dmy = parse_en_day_month_year(candidate_text)
    if en_dmy and is_plausible_result(en_dmy, candidate_text, base_time):
        return to_parsed_event(candidate_text, en_dmy, base_time, title)

    zh_md = parse_zh_month_day(candidate_text, base_time, title)
    if zh_md:
        if is_plausible_result(zh_md, candidate_text, base_time):
            return to_parsed_event(candidate_text, zh_md, base_time, title)
        return None

    zh_q_prefix = parse_zh_quarter_prefix(candidate_text, base_time, title)
    if zh_q_prefix and is_plausible_result(zh_q_prefix, candidate_text, base_time):
        return to_parsed_event(candidate_text, zh_q_prefix, base_time, title)

    zh_m_span = parse_zh_month_span(candidate_text, base_time, title)
    if zh_m_span and is_plausible_result(zh_m_span, candidate_text, base_time):
        return to_parsed_event(candidate_text, zh_m_span, base_time, title)

    en_md = parse_en_month_day(candidate_text, base_time, title)
    if en_md and is_plausible_result(en_md, candidate_text, base_time):
        return to_parsed_event(candidate_text, en_md, base_time, title)

    zh_m = parse_zh_month_only(candidate_text, base_time, title)
    if zh_m and is_plausible_result(zh_m, candidate_text, base_time):
        return to_parsed_event(candidate_text, zh_m, base_time, title)

    quarter_or_half = parse_quarter_or_half(candidate_text, base_time, title)
    if quarter_or_half and is_plausible_result(quarter_or_half, candidate_text, base_time):
        return to_parsed_event(candidate_text, quarter_or_half, base_time, title)

    year_only = parse_year_only(candidate_text, base_time, title)
    if year_only and is_plausible_result(year_only, candidate_text, base_time):
        return to_parsed_event(candidate_text, year_only, base_time, title)

    settings = {
        'RELATIVE_BASE': base_time,
        'TIMEZONE': 'Asia/Shanghai',
        'RETURN_AS_TIMEZONE_AWARE': False,
        'PREFER_DAY_OF_MONTH': 'first',
        'DATE_ORDER': 'YMD',
        'PREFER_DATES_FROM': detect_prefer_dates_from(title)
    }

    lang_hints = DATEPARSER_LANGUAGE_MAP.get(lang)

    parsed = None
    try:
        if lang_hints:
            parsed = dateparser.parse(candidate_text, settings=settings, languages=lang_hints)
        else:
            parsed = dateparser.parse(candidate_text, settings=settings)
    except Exception:
        parsed = None

    if not parsed:
        return None

    parsed = normalize_granularity(candidate_text, parsed, base_time, title)
    if not is_plausible_result(parsed, candidate_text, base_time):
        return None

    return to_parsed_event(candidate_text, parsed, base_time, title)

def extract_event_time(title, base_time):
    """
    三级降维分流提取逻辑
    """
    if not title:
        return None, "fallback"

    base_time = normalize_base_time(base_time)
    if base_time is None:
        return None, "fallback"

    try:
        lang = detect(title)
    except Exception:
        return None, "fallback" # 乱码或纯符号直接走兜底

    # ---------------------------------------------------------
    # 第一梯队：spaCy 精准圈词 + dateparser 计算
    # ---------------------------------------------------------
    nlp_model = nlp_models.get(lang)
    ranked_candidates = collect_time_candidates(title, lang, nlp_model)
    for candidate_text, _ in ranked_candidates:
        parsed_result = parse_candidate_time(candidate_text, base_time, lang, title)
        if parsed_result:
            return parsed_result, "tier1"

    # ---------------------------------------------------------
    # 第二梯队：dateparser 搜索模式兜底（优先保留 sw 专属能力）
    # ---------------------------------------------------------
    lang_hints = DATEPARSER_LANGUAGE_MAP.get(lang)
    try:
        settings = {
            'RELATIVE_BASE': base_time,
            'TIMEZONE': 'Asia/Shanghai',
            'RETURN_AS_TIMEZONE_AWARE': False,
            'PREFER_DAY_OF_MONTH': 'first',
            'DATE_ORDER': 'YMD',
            'PREFER_DATES_FROM': detect_prefer_dates_from(title)
        }
        extracted_dates = search_dates(title, languages=lang_hints, settings=settings) if lang_hints else search_dates(title, settings=settings)
        if extracted_dates:
            for matched_text, parsed_date in extracted_dates:
                matched_text = normalize_datetime_text(matched_text)
                if is_noise_candidate(matched_text):
                    continue
                if matched_text.lower() == "may":
                    continue

                direct_parsed = parse_candidate_time(matched_text, base_time, lang, title)
                if direct_parsed and is_plausible_result(direct_parsed.anchor, matched_text, base_time):
                    return direct_parsed, "tier2"

                normalized = normalize_granularity(matched_text, parsed_date, base_time, title)
                if is_plausible_result(normalized, matched_text, base_time):
                    return to_parsed_event(matched_text, normalized, base_time, title), "tier2"
    except Exception:
        pass

    # ---------------------------------------------------------
    # 第三梯队：其余长尾小语种，直接返回 None 触发兜底
    # ---------------------------------------------------------
    return None, "fallback"

def ensure_parser_newsdata_schema(cursor):
    cursor.execute("SHOW COLUMNS FROM parser_newsdata")
    columns = {row["Field"] for row in cursor.fetchall()}

    ddl_statements = []
    if "event_time_start" not in columns:
        ddl_statements.append("ALTER TABLE parser_newsdata ADD COLUMN event_time_start DATETIME NULL AFTER event_timestamp")
    if "event_time_end" not in columns:
        ddl_statements.append("ALTER TABLE parser_newsdata ADD COLUMN event_time_end DATETIME NULL AFTER event_time_start")
    if "time_granularity" not in columns:
        ddl_statements.append("ALTER TABLE parser_newsdata ADD COLUMN time_granularity VARCHAR(32) NULL AFTER event_time_end")
    if "parse_mode" not in columns:
        ddl_statements.append("ALTER TABLE parser_newsdata ADD COLUMN parse_mode VARCHAR(16) NULL AFTER time_granularity")

    for ddl in ddl_statements:
        cursor.execute(ddl)
    return len(ddl_statements)

def process_news_pipeline():
    print("开始执行三级混合 OSINT 时间清洗流水线...")
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cursor:
            added_column_count = ensure_parser_newsdata_schema(cursor)
            if added_column_count > 0:
                conn.commit()
                print(f"已自动补齐 parser_newsdata 字段 {added_column_count} 个。")

            # 读取原始数据表 (包含所有必备字段)
            select_sql = """
                SELECT id, title, raw_time, standard_timestamp, source, url, true_order, is_noise 
                FROM raw_news_data 
            """
            cursor.execute(select_sql)
            records = cursor.fetchall()
            
            if not records:
                print("数据库中未找到符合条件的数据。")
                return

            insert_values = []
            
            # 统计雷达
            stats = {
                'tier1_success': 0,
                'tier2_success': 0,
                'fallback_used': 0
            }

            for row in records:
                title = row['title']
                base_time = row['standard_timestamp']
                parse_mode = "fallback"
                
                # 安全检查：如果连基准时间都缺失，强制走兜底
                if base_time is None:
                    event_result = None
                    stats['fallback_used'] += 1
                else:
                    # 核心分流处理
                    event_result, strategy = extract_event_time(title, base_time)
                    
                    if strategy == "tier1":
                        stats['tier1_success'] += 1
                    elif strategy == "tier2":
                        stats['tier2_success'] += 1
                    else:
                        # 触发兜底逻辑
                        event_result = build_result(
                            base_time,
                            base_time,
                            base_time,
                            "day",
                            "fallback",
                            base_time,
                            title or "",
                        )
                        stats['fallback_used'] += 1
                    parse_mode = strategy

                # 准备待入库的数据 (确保字段一一对应)
                event_anchor = event_result.anchor if event_result else None
                event_start = event_result.start if event_result else None
                event_end = event_result.end if event_result else None
                event_granularity = event_result.granularity if event_result else None
                insert_values.append((
                    row['id'], row['title'], row['raw_time'], row['standard_timestamp'],
                    event_anchor, event_start, event_end, event_granularity, parse_mode,
                    row['source'], row['url'], row['true_order'], row['is_noise']
                ))

            # 执行大批量事务写入 (完美 UPSERT 镜像写入)
            if insert_values:
                insert_sql = """
                    INSERT INTO parser_newsdata 
                    (id, title, raw_time, standard_timestamp, event_timestamp, event_time_start, event_time_end, time_granularity, parse_mode, source, url, true_order, is_noise)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE 
                        title = VALUES(title),
                        raw_time = VALUES(raw_time),
                        standard_timestamp = VALUES(standard_timestamp),
                        event_timestamp = VALUES(event_timestamp),
                        event_time_start = VALUES(event_time_start),
                        event_time_end = VALUES(event_time_end),
                        time_granularity = VALUES(time_granularity),
                        parse_mode = VALUES(parse_mode),
                        source = VALUES(source),
                        url = VALUES(url),
                        true_order = VALUES(true_order),
                        is_noise = VALUES(is_noise)
                """
                cursor.executemany(insert_sql, insert_values)
                conn.commit()
                
                print(f"\nETL 处理完毕，共流转入库 {len(insert_values)} 条数据。")
                print(f"================ 引擎执行报告 ================")
                print(f"spaCy提取: {stats['tier1_success']} 条")
                print(f"search提取: {stats['tier2_success']} 条")
                print(f"无特征/小语种/无基准: {stats['fallback_used']} 条")
                print(f"==============================================")

    except Exception as e:
        print(f"MySQL 事务执行异常，已自动回滚: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    process_news_pipeline()
