import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import dateparser

ANCHOR_TIME = datetime(2026, 3, 20, 13, 19, 34, tzinfo=ZoneInfo("Asia/Shanghai"))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "newsdata_for_test"
INPUT_FILE = DATA_DIR / "newsdata_test1.json"
OUTPUT_FILE = DATA_DIR / "newsdata_test1_parser.json"


def clean_time_string(raw_string):
    """预处理不规则时间字符串"""
    if raw_string is None:
        return ""
    if not isinstance(raw_string, str):
        raw_string = str(raw_string)

    text = raw_string.strip()
    if not text:
        return ""

    publish_patterns = [
        r"Published\s*[:\-]?\s*([^|]+)",
        r"发布时间\s*[:：]?\s*([^|]+)",
    ]
    for pattern in publish_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    if "|" in text:
        parts = [p.strip() for p in text.split("|") if p.strip()]
        if parts:
            text = parts[0]

    return re.sub(
        r"^(published|last\s*updated|updated|update\s*time)\s*[:\-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def parse_to_iso(raw_time, anchor):
    """将异构时间解析为 ISO-8601"""
    cleaned_time = clean_time_string(raw_time)
    if not cleaned_time:
        return None

    settings = {
        "RELATIVE_BASE": anchor,
        "TIMEZONE": "Asia/Shanghai",
        "TO_TIMEZONE": "Asia/Shanghai",
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "past",
        "DATE_ORDER": "YMD",
    }

    candidates = [cleaned_time]
    if isinstance(raw_time, str) and raw_time.strip() and raw_time.strip() != cleaned_time:
        candidates.append(raw_time.strip())

    for candidate in candidates:
        parsed_date = dateparser.parse(candidate, settings=settings, languages=["en", "zh"])
        if parsed_date:
            return parsed_date.isoformat()

    match = re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?", cleaned_time)
    if match:
        parsed_date = dateparser.parse(match.group(0), settings=settings, languages=["en", "zh"])
        if parsed_date:
            return parsed_date.isoformat()

    return None


def process_news_times(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    if not Path(input_file).exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        news_data = json.load(f)

    if not isinstance(news_data, list):
        raise ValueError("输入 JSON 顶层必须是列表")

    success_count = 0
    print("=== 开始时间时序标准化 ===")

    for item in news_data:
        original = item.get("raw_time")
        iso_time = parse_to_iso(original, ANCHOR_TIME)
        item["parsed_time"] = iso_time

        if iso_time is not None:
            success_count += 1

        item_id = item.get("id", "N/A")
        print(f"ID: {item_id}")
        print(f"  原始输入: {original}")
        print(f"  ISO 标准: {iso_time}\\n")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(news_data, f, ensure_ascii=False, indent=2)

    total_count = len(news_data)
    fail_count = total_count - success_count
    print(f"=== 解析完成！已保存为 {output_file} ===")
    print(f"总数: {total_count}，成功: {success_count}，失败: {fail_count}")


if __name__ == "__main__":
    process_news_times()