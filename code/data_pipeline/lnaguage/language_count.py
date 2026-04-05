import pymysql
from langdetect import detect, DetectorFactory
from collections import Counter
import traceback

# 保证每次语言检测结果的一致性
DetectorFactory.seed = 0

# 数据库配置（复用你的配置）
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '123456', 
    'database': 'news_db',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor 
}

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

def analyze_language_distribution():
    print("开始扫描数据库，探查语言分布...")
    conn = get_db_connection()
    lang_counter = Counter()
    error_count = 0
    total_records = 0

    try:
        with conn.cursor() as cursor:
            # 只查询需要处理的数据，可以根据情况去掉 WHERE 条件以扫描全表
            select_sql = "SELECT id, title FROM raw_news_data WHERE standard_timestamp IS NOT NULL"
            cursor.execute(select_sql)
            
            # 使用 fetchmany 分批读取，防止数据量过大撑爆内存
            batch_size = 5000
            while True:
                records = cursor.fetchmany(batch_size)
                if not records:
                    break
                    
                total_records += len(records)
                
                for row in records:
                    title = row['title']
                    if not title or not title.strip():
                        lang_counter['empty'] += 1
                        continue
                        
                    try:
                        detected_lang = detect(title)
                        lang_counter[detected_lang] += 1
                    except Exception:
                        # 捕获无法识别的乱码或纯符号
                        lang_counter['unknown'] += 1
                        error_count += 1
                        
            print(f"\n扫描完成,共处理 {total_records} 条新闻标题。")
            print("=" * 40)
            print("语言分布统计报告 (ISO 639-1 简码)：")
            print("=" * 40)
            
            # 按出现频次从高到低排序打印
            for lang, count in lang_counter.most_common():
                # 计算占比
                percentage = (count / total_records) * 100 if total_records > 0 else 0
                print(f"语种: {lang:<8} | 频次: {count:<8} | 占比: {percentage:.2f}%")
                
            print("-" * 40)
            print(f"解析失败 (纯符号/乱码): {error_count} 条")

    except Exception as e:
        print(f"数据库操作异常: {e}")
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    analyze_language_distribution()