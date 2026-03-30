import pymysql
import requests
import time
import uuid
import os
from datetime import datetime, timedelta, timezone

# ================= 配置区 =================
# 数据库配置 
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '123456', 
    'database': 'news_db',
    'charset': 'utf8mb4'
}

DOMAINS = {
    "Al Jazeera": "aljazeera.com",
    "BBC": "bbc.com",
    "DW": "dw.com",
    "联合早报": "zaobao.com",
    "The New York Times": "nytimes.com",
    "新华网": "xinhuanet.com"
}

# 全局默认时间范围
DEFAULT_START_DATE = datetime(2025, 6, 1, tzinfo=timezone.utc)
END_DATE = datetime.now(timezone.utc)
DAYS_PER_STEP = 7

# 请求头伪装，降低被拦截概率
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
# ==========================================

def get_db_connection():
    """获取数据库连接"""
    return pymysql.connect(**DB_CONFIG)

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
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS gdelt_checkpoints (
                    source VARCHAR(255) PRIMARY KEY,
                    next_start_time VARCHAR(16) NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) CHARACTER SET utf8mb4
                """
            )
        conn.commit()
    finally:
        conn.close()


def get_checkpoint(source_name):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT next_start_time FROM gdelt_checkpoints WHERE source = %s", (source_name,))
            result = cursor.fetchone()
            if result and result[0]:
                checkpoint = parse_gdelt_time(result[0])
                if checkpoint:
                    return checkpoint

            sql = "SELECT MAX(raw_time) FROM raw_news_data WHERE source = %s AND id LIKE 'gdelt_%%' AND raw_time IS NOT NULL AND raw_time <> ''"
            cursor.execute(sql, (source_name,))
            result = cursor.fetchone()[0]
            checkpoint = parse_gdelt_time(result)
            if checkpoint:
                return checkpoint + timedelta(seconds=1)
            return DEFAULT_START_DATE
    finally:
        conn.close()


def update_checkpoint(source_name, next_start_time):
    checkpoint_str = next_start_time.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO gdelt_checkpoints (source, next_start_time)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE next_start_time = VALUES(next_start_time)
                """,
                (source_name, checkpoint_str)
            )
        conn.commit()
    finally:
        conn.close()

def save_to_mysql(articles, source_name):
    """将数据批量写入 MySQL，利用 INSERT IGNORE 自动忽略重复的 URL"""
    if not articles:
        return 0
        
    conn = get_db_connection()
    inserted_count = 0
    try:
        with conn.cursor() as cursor:
            sql = """
                INSERT IGNORE INTO raw_news_data 
                (id, title, raw_time, source, url) 
                VALUES (%s, %s, %s, %s, %s)
            """
            values = []
            for art in articles:
                uid = f"gdelt_{str(uuid.uuid4())[:8]}"
                title = art.get("title", "")
                raw_time = normalize_gdelt_time(art.get("seendate", ""))
                url = art.get("url", "")
                if title and url:
                    values.append((uid, title, raw_time, source_name, url))
            
            if values:
                # executemany 用于批量插入，提升性能
                inserted_count = cursor.executemany(sql, values)
        conn.commit()
        return inserted_count
    except Exception as e:
        print(f"      [!] 数据库写入异常: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()

def fetch_with_retry(api_url, max_retries=6):
    """带指数退避的请求函数，应对 429 和超时"""
    for attempt in range(max_retries):
        try:
            response = requests.get(api_url, headers=HEADERS, timeout=20)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                sleep_time = 2 ** attempt  # 1, 2, 4, 8, 16, 32 秒
                print(f"      [!] 触发 429 限流，进入指数退避，等待 {sleep_time} 秒后重试...")
                time.sleep(sleep_time)
            elif response.status_code >= 500:
                print(f"      [!] GDELT 服务器错误 ({response.status_code})，稍后重试...")
                time.sleep(5)
            else:
                print(f"      [!] 异常状态码: {response.status_code}")
                break
        except requests.exceptions.RequestException as e:
            sleep_time = 2 ** attempt
            print(f"      [!] 网络异常/超时，等待 {sleep_time} 秒后重试...")
            time.sleep(sleep_time)
            
    print("      [!] 达到最大重试次数，跳过当前时间段。")
    return None

def run_gdelt_scraper():
    print("GDELT 增量抓取 (写入MySQL)...")
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
            api_url = f"https://api.gdeltproject.org/api/v2/doc/doc?query=domain:{domain}&startdatetime={start_str}&enddatetime={end_str}&mode=artlist&maxrecords=250&format=json"
            data = fetch_with_retry(api_url)

            if data is None:
                print(f"  -> {current_start.strftime('%Y-%m-%d')} 至 {current_end.strftime('%Y-%m-%d')} : 抓取失败，停止当前数据源以避免错误推进断点。")
                break

            articles = data.get("articles", [])
            inserted = save_to_mysql(articles, source_name)
            if articles:
                print(f"  -> {current_start.strftime('%Y-%m-%d')} 至 {current_end.strftime('%Y-%m-%d')} : 获取 {len(articles)} 条，成功入库 {inserted} 条新数据。")
            else:
                print(f"  -> {current_start.strftime('%Y-%m-%d')} 至 {current_end.strftime('%Y-%m-%d')} : 无新数据。")

            next_start = current_end + timedelta(seconds=1)
            update_checkpoint(source_name, next_start)
            time.sleep(3)
            current_start = next_start

if __name__ == "__main__":
    run_gdelt_scraper()
    print("\n抓取任务执行完毕")