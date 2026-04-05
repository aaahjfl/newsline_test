import pymysql
import pytz
from datetime import datetime, timezone

# 数据库配置（复用你的配置）
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '123456', 
    'database': 'news_db',
    'charset': 'utf8mb4',
    # 推荐使用 DictCursor，便于通过字段名读取数据
    'cursorclass': pymysql.cursors.DictCursor 
}

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

def normalize_and_convert_timezone():
    print("开始执行新闻基准时间 (DCT) 标准化与时区转换...")
    conn = get_db_connection()
    
    # 目标时区：东八区
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    
    try:
        with conn.cursor() as cursor:
            # 1. 幂等性查询：只抽取尚未标准化的数据，避免重复处理
            # 过滤掉 raw_time 为空或格式无效的脏数据
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

            # 2. 时区解析与转换
            for row in records:
                raw_time_str = row['raw_time'].strip()
                record_id = row['id']
                parsed_utc_dt = None

                # 兼容 GDELT 的两种常见时间格式
                for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
                    try:
                        # 解析为 UTC 绝对时间
                        parsed_utc_dt = datetime.strptime(raw_time_str, fmt).replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        continue
                
                if parsed_utc_dt:
                    # 强制转换为 Asia/Shanghai (北京时间)
                    local_dt = parsed_utc_dt.astimezone(shanghai_tz)
                    # 格式化为 MySQL datetime 接受的格式
                    standard_time_str = local_dt.strftime("%Y-%m-%d %H:%M:%S")
                    update_values.append((standard_time_str, record_id))
                else:
                    error_count += 1

            # 3. MySQL 事务级批量更新
            if update_values:
                update_sql = """
                    UPDATE raw_news_data 
                    SET standard_timestamp = %s 
                    WHERE id = %s
                """
                # 使用 executemany 进行批量操作，极大减少网络 I/O 开销
                cursor.executemany(update_sql, update_values)
                conn.commit()
                
                print(f"成功将 {len(update_values)} 条记录的时间转换为东八区并更新入库")
            
            if error_count > 0:
                print(f"警告：有 {error_count} 条数据的 raw_time 格式无法解析，已跳过。")

    except Exception as e:
        print(f"数据库操作异常: {e}")
        conn.rollback()
    finally:
        conn.close()
        print("数据库连接已关闭。")

if __name__ == "__main__":
    normalize_and_convert_timezone()