"""Current RSS dataset builder under the formal architecture."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from configs.dataset_config import DATASET_CONFIG
from configs.path_config import NEWSDATA_DIR

try:
    import feedparser
except ImportError:  # pragma: no cover - optional until runtime.
    feedparser = None

try:
    import requests
except ImportError:  # pragma: no cover - optional until runtime.
    requests = None


RSS_SOURCES = DATASET_CONFIG["rss_sources"]
MAX_ITEMS_PER_SOURCE = DATASET_CONFIG["rss_max_items_per_source"]
DAYS_LOOKBACK = DATASET_CONFIG["rss_days_lookback"]
HTTP_HEADERS = DATASET_CONFIG["http_headers"]


def get_rss_output_path() -> Path:
    """Return the canonical dataset file for RSS news snapshots."""
    return NEWSDATA_DIR / DATASET_CONFIG["rss_output_filename"]


def load_existing_data(filepath: Path | str | None = None) -> list[dict]:
    """Read the existing RSS dataset for incremental deduplication."""
    path = Path(filepath) if filepath is not None else get_rss_output_path()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError:
            return []
    return []


def save_rss_dataset(records: list[dict], filepath: Path | str | None = None) -> Path:
    """Persist the RSS dataset into the historical dataset directory."""
    path = Path(filepath) if filepath is not None else get_rss_output_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
    return path


def is_within_time_window(raw_time_str: str | None, days: int = DAYS_LOOKBACK) -> bool:
    """Return whether an RSS item falls inside the freshness window."""
    if not raw_time_str:
        return True

    try:
        pub_date = parsedate_to_datetime(raw_time_str)
        now = datetime.now(timezone.utc)
        return (now - pub_date) <= timedelta(days=days)
    except Exception:
        return True


def fetch_rss_news(sources: dict[str, str] | None = None, *, filepath: Path | str | None = None) -> tuple[list[dict], int]:
    """Fetch RSS entries and merge them into the existing dataset snapshot."""
    if requests is None or feedparser is None:
        raise RuntimeError("RSS dataset building requires both requests and feedparser to be installed.")

    if sources is None:
        sources = RSS_SOURCES

    existing_data = load_existing_data(filepath)
    existing_urls = {item["url"] for item in existing_data if "url" in item}
    new_data_count = 0

    for source_name, rss_url in sources.items():
        print(f"正在抓取: {source_name} ...")
        try:
            response = requests.get(rss_url, headers=HTTP_HEADERS, timeout=10)
            if response.status_code != 200:
                print(f"  -> 抓取失败: HTTP 状态码 {response.status_code}")
                continue

            feed = feedparser.parse(response.content)
            if not feed.entries:
                print("  -> 警告：抓取成功但未发现文章，可能 RSS 源为空或页面结构改变。")
                continue

            source_count = 0
            for entry in feed.entries:
                if source_count >= MAX_ITEMS_PER_SOURCE:
                    break

                url = entry.get("link", "")
                if url in existing_urls:
                    continue

                title = entry.get("title", "").strip()
                raw_time = entry.get("published", entry.get("updated", ""))
                if not title:
                    continue
                if not is_within_time_window(raw_time, DAYS_LOOKBACK):
                    continue

                news_item = {
                    "id": f"rss_{str(uuid.uuid4())[:8]}",
                    "title": title,
                    "raw_time": raw_time,
                    "standard_timestamp": None,
                    "source": source_name,
                    "url": url,
                    "true_order": None,
                    "is_noise": None,
                }
                existing_data.append(news_item)
                existing_urls.add(url)
                source_count += 1
                new_data_count += 1

            print(f"  -> 抓取并保留了 {source_count} 条新数据。")
        except Exception as exc:
            print(f"  -> 抓取失败: {exc}")

        time.sleep(2)

    return existing_data, new_data_count


def build_rss_dataset(sources: dict[str, str] | None = None, *, filepath: Path | str | None = None) -> tuple[list[dict], int, Path]:
    """Run the active RSS dataset-building flow and persist the merged snapshot."""
    records, added_count = fetch_rss_news(sources, filepath=filepath)
    output_path = Path(filepath) if filepath is not None else get_rss_output_path()
    if added_count > 0:
        save_rss_dataset(records, output_path)
    return records, added_count, output_path
