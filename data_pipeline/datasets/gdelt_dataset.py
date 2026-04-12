"""Current GDELT incremental dataset builder under the formal architecture."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone

from configs.dataset_config import DATASET_CONFIG
from configs.db_config import get_db_config

try:
    import pymysql
except ImportError:  # pragma: no cover - optional until runtime.
    pymysql = None

try:
    import requests
except ImportError:  # pragma: no cover - optional until runtime.
    requests = None


DOMAINS = DATASET_CONFIG["gdelt_domains"]
DEFAULT_START_DATE = DATASET_CONFIG["gdelt_default_start_date"]
END_DATE = DATASET_CONFIG["gdelt_end_date"]
DAYS_PER_STEP = DATASET_CONFIG["gdelt_days_per_step"]
HEADERS = DATASET_CONFIG["http_headers"]


def get_db_connection():
    """Return a database connection that preserves the legacy cursor behavior."""
    if pymysql is None:
        raise RuntimeError("GDELT dataset building requires PyMySQL to be installed.")
    return pymysql.connect(**get_db_config())


def normalize_gdelt_time(raw_time):
    if not raw_time:
        return ""
    raw_time = raw_time.strip()
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
        try:
            dt = datetime.strptime(raw_time, fmt).replace(tzinfo=timezone.utc)
            return dt.strftime("%Y%m%dT%H%M%SZ")
        except ValueError:
            continue
    return ""


def parse_gdelt_time(raw_time):
    normalized = normalize_gdelt_time(raw_time)
    if not normalized:
        return None
    return datetime.strptime(normalized, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def ensure_checkpoint_table():
    """Preserve the legacy checkpoint-table behavior for the active GDELT flow."""
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS gdelt_checkpoints (
                    source VARCHAR(255) PRIMARY KEY,
                    next_start_time VARCHAR(16) NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) CHARACTER SET utf8mb4
                """
            )
        connection.commit()
    finally:
        connection.close()


def get_checkpoint(source_name):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT next_start_time FROM gdelt_checkpoints WHERE source = %s", (source_name,))
            result = cursor.fetchone()
            if result and result[0]:
                checkpoint = parse_gdelt_time(result[0])
                if checkpoint:
                    return checkpoint

            sql = (
                "SELECT MAX(raw_time) FROM raw_news_data "
                "WHERE source = %s AND id LIKE 'gdelt_%%' AND raw_time IS NOT NULL AND raw_time <> ''"
            )
            cursor.execute(sql, (source_name,))
            result = cursor.fetchone()[0]
            checkpoint = parse_gdelt_time(result)
            if checkpoint:
                return checkpoint + timedelta(seconds=1)
            return DEFAULT_START_DATE
    finally:
        connection.close()


def update_checkpoint(source_name, next_start_time):
    checkpoint_str = next_start_time.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO gdelt_checkpoints (source, next_start_time)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE next_start_time = VALUES(next_start_time)
                """,
                (source_name, checkpoint_str),
            )
        connection.commit()
    finally:
        connection.close()


def save_to_mysql(articles, source_name):
    """Preserve the current incremental GDELT-to-MySQL write behavior."""
    if not articles:
        return 0

    connection = get_db_connection()
    inserted_count = 0
    try:
        with connection.cursor() as cursor:
            sql = """
                INSERT IGNORE INTO raw_news_data 
                (id, title, raw_time, source, url) 
                VALUES (%s, %s, %s, %s, %s)
            """
            values = []
            for article in articles:
                uid = f"gdelt_{str(uuid.uuid4())[:8]}"
                title = article.get("title", "")
                raw_time = normalize_gdelt_time(article.get("seendate", ""))
                url = article.get("url", "")
                if title and url:
                    values.append((uid, title, raw_time, source_name, url))

            if values:
                inserted_count = cursor.executemany(sql, values)
        connection.commit()
        return inserted_count
    except Exception as exc:
        print(f"      [!] 数据库写入异常: {exc}")
        connection.rollback()
        return 0
    finally:
        connection.close()


def fetch_with_retry(api_url, max_retries=5):
    """Preserve the current GDELT retry and backoff behavior."""
    if requests is None:
        raise RuntimeError("GDELT dataset building requires requests to be installed.")

    for _ in range(max_retries):
        try:
            response = requests.get(api_url, headers=HEADERS, timeout=20)
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429:
                print("      [!] 触发严格限流 (HTTP 429)，IP 被暂封。进入 15 分钟长时休眠以恢复状态...")
                time.sleep(900)
            elif response.status_code >= 500:
                print(f"      [!] GDELT 服务器内部错误 ({response.status_code})，等待 30 秒后重试...")
                time.sleep(30)
            else:
                print(f"      [!] 异常状态码: {response.status_code}")
                break
        except requests.exceptions.RequestException as exc:
            print(f"      [!] 网络异常或超时 ({exc})，等待 30 秒后重试...")
            time.sleep(30)

    print("      [!] 达到最大长时重试次数，为保护程序运行，跳过当前时间段。")
    return None


def run_incremental_gdelt_dataset_build():
    """Run the current incremental GDELT collection job."""
    print("GDELT 增量抓取 (带长时休眠保护与 MySQL 批量写入)...")
    ensure_checkpoint_table()

    for source_name, domain in DOMAINS.items():
        current_start = get_checkpoint(source_name)
        print(f"\n[{source_name}] 从断点时间 {current_start.strftime('%Y-%m-%d %H:%M:%S UTC')} 开始抓取...")

        while current_start < END_DATE:
            current_end = current_start + timedelta(days=DAYS_PER_STEP)
            if current_end > END_DATE:
                current_end = END_DATE

            start_str = current_start.strftime("%Y%m%d%H%M%S")
            end_str = current_end.strftime("%Y%m%d%H%M%S")
            api_url = (
                "https://api.gdeltproject.org/api/v2/doc/doc"
                f"?query=domain:{domain}"
                f"&startdatetime={start_str}"
                f"&enddatetime={end_str}"
                "&mode=artlist&maxrecords=250&format=json"
            )

            data = fetch_with_retry(api_url)
            if data is None:
                print(
                    f"  -> {current_start.strftime('%Y-%m-%d')} 至 "
                    f"{current_end.strftime('%Y-%m-%d')} : 抓取严重失败，停止当前数据源，保留断点。"
                )
                break

            articles = data.get("articles", [])
            inserted = save_to_mysql(articles, source_name)
            if articles:
                print(
                    f"  -> {current_start.strftime('%Y-%m-%d')} 至 "
                    f"{current_end.strftime('%Y-%m-%d')} : 获取 {len(articles)} 条，成功入库 MySQL {inserted} 条。"
                )
            else:
                print(
                    f"  -> {current_start.strftime('%Y-%m-%d')} 至 "
                    f"{current_end.strftime('%Y-%m-%d')} : 无新数据。"
                )

            next_start = current_end + timedelta(seconds=1)
            update_checkpoint(source_name, next_start)
            time.sleep(180)
            current_start = next_start
