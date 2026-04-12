"""Current DCT normalization flow under the formal architecture."""

from __future__ import annotations

from datetime import datetime, timezone

from configs.db_config import get_db_config

try:
    import pymysql
except ImportError:  # pragma: no cover - optional until runtime.
    pymysql = None

try:
    import pytz
except ImportError:  # pragma: no cover - optional until runtime.
    pytz = None


def get_db_connection():
    """Return a DictCursor connection matching the current standardizer behavior."""
    if pymysql is None:
        raise RuntimeError("Time standardization requires PyMySQL to be installed.")
    config = get_db_config({"cursorclass": pymysql.cursors.DictCursor})
    return pymysql.connect(**config)


def run_time_standardization():
    """Normalize raw publication timestamps into standard_timestamp."""
    if pytz is None:
        raise RuntimeError("Time standardization requires pytz to be installed.")

    print("开始执行新闻基准时间 (DCT) 标准化与时区转换...")
    connection = get_db_connection()
    shanghai_tz = pytz.timezone("Asia/Shanghai")

    try:
        with connection.cursor() as cursor:
            select_sql = """
                SELECT id, raw_time 
                FROM raw_news_data 
                WHERE standard_timestamp IS NULL 
                  AND raw_time IS NOT NULL 
                  AND raw_time != ''
            """
            cursor.execute(select_sql)
            records = cursor.fetchall()

            if not records:
                print("没有需要标准化的新数据。")
                return

            print(f"获取到 {len(records)} 条待处理数据，开始解析...")
            update_values = []
            error_count = 0

            for row in records:
                raw_time_str = row["raw_time"].strip()
                record_id = row["id"]
                parsed_utc_dt = None

                for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
                    try:
                        parsed_utc_dt = datetime.strptime(raw_time_str, fmt).replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        continue

                if parsed_utc_dt:
                    local_dt = parsed_utc_dt.astimezone(shanghai_tz)
                    standard_time_str = local_dt.strftime("%Y-%m-%d %H:%M:%S")
                    update_values.append((standard_time_str, record_id))
                else:
                    error_count += 1

            if update_values:
                update_sql = """
                    UPDATE raw_news_data 
                    SET standard_timestamp = %s 
                    WHERE id = %s
                """
                cursor.executemany(update_sql, update_values)
                connection.commit()
                print(f"成功将 {len(update_values)} 条记录的时间转换为东八区并更新入库")

            if error_count > 0:
                print(f"警告：有 {error_count} 条数据的 raw_time 格式无法解析，已跳过。")
    except Exception as exc:
        print(f"数据库操作异常: {exc}")
        connection.rollback()
    finally:
        connection.close()
        print("数据库连接已关闭。")
