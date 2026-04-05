import pymysql
import dateparser
import re
from datetime import datetime
from py_heideltime import heideltime
from langdetect import detect, DetectorFactory

# 保证每次语言检测结果的一致性
DetectorFactory.seed = 0

# 数据库配置
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

# 强力文本清洗：防止 Java 进程 I/O 流崩溃及 UIMA 报错
def clean_text_for_java(text):
    if not text:
        return "Empty."
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if text and not text.endswith('.'):
        text += '.'
    return text

def build_english_ht_input(title, dct_str):
    safe_title = clean_text_for_java(title)
    if len(safe_title) < 20:
        return f"{safe_title} This news was published on {dct_str}."
    return f"{safe_title} Published on {dct_str}."

def extract_ht_value(ht_result):
    if isinstance(ht_result, dict):
        value = ht_result.get('value')
        return value if value else None

    if isinstance(ht_result, list):
        for item in ht_result:
            if isinstance(item, dict) and item.get('type') == 'DATE' and item.get('value'):
                return item.get('value')
        for item in ht_result:
            if isinstance(item, dict) and item.get('value'):
                return item.get('value')

    return None

def parse_ht_value(value, base_time):
    if not value or not isinstance(value, str):
        return None

    v = value.strip().upper()

    if v in {'PRESENT_REF', 'PAST_REF', 'FUTURE_REF'}:
        return base_time

    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', v):
        return datetime.strptime(v, "%Y-%m-%d")

    if re.fullmatch(r'\d{4}-\d{2}', v):
        return datetime.strptime(f"{v}-01", "%Y-%m-%d")

    if re.fullmatch(r'\d{4}', v):
        return datetime.strptime(f"{v}-01-01", "%Y-%m-%d")

    if re.fullmatch(r'\d{4}-Q[1-4]', v):
        year = int(v[:4])
        quarter = int(v[-1])
        month = (quarter - 1) * 3 + 1
        return datetime(year, month, 1)

    if re.fullmatch(r'\d{4}-W\d{2}', v):
        year = int(v[:4])
        week = int(v[-2:])
        return datetime.fromisocalendar(year, week, 1)

    return dateparser.parse(
        value,
        settings={
            'RELATIVE_BASE': base_time,
            'PREFER_DAY_OF_MONTH': 'first',
            'PREFER_DATES_FROM': 'past'
        }
    )

def extract_and_calculate_event_time():
    print("开始执行 HeidelTime 英语专版语义提取测试...")
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cursor:
            select_sql = """
                SELECT id, title, raw_time, standard_timestamp, source, url, true_order, is_noise 
                FROM raw_news_data 
                WHERE standard_timestamp IS NOT NULL
            """
            cursor.execute(select_sql)
            records = cursor.fetchall()
            
            if not records:
                print("未找到数据。")
                return

            insert_values = []
            
            # 精准计数器
            stats = {
                'english_ht_success': 0,    # 英语且 HeidelTime 成功提取
                'english_ht_error': 0,      # 英语但 HeidelTime 抛出异常
                'english_ht_empty': 0,      # 英语但 HeidelTime 没找到时间词
                'non_english_skipped': 0,   # 非英语直接跳过
                'total_fallback': 0         # 最终使用默认时间的总条数
            }

            for row in records:
                original_title = row['title']
                base_time = row['standard_timestamp'] 
                
                # 默认兜底事件时间
                event_time = base_time 
                used_heideltime = False

                # 1. 语言检测
                try:
                    detected_lang = detect(original_title)
                except Exception:
                    detected_lang = 'unknown'

                if detected_lang == 'en':
                    dct_str = base_time.strftime("%Y-%m-%d")
                    safe_title = build_english_ht_input(original_title, dct_str)

                    ht_result = None
                    last_error = None

                    for doc_type in ('news', 'colloquial', 'narratives'):
                        try:
                            ht_result = heideltime(
                                safe_title,
                                language='English',
                                document_type=doc_type,
                                dct=dct_str
                            )
                            if ht_result:
                                break
                        except Exception as parse_err:
                            last_error = parse_err

                    time_value_str = extract_ht_value(ht_result)
                    parsed_date = parse_ht_value(time_value_str, base_time)

                    if parsed_date:
                        event_time = parsed_date
                        used_heideltime = True
                        stats['english_ht_success'] += 1
                    elif last_error is not None:
                        stats['english_ht_error'] += 1
                    else:
                        stats['english_ht_empty'] += 1
                else:
                    stats['non_english_skipped'] += 1

                # 3. 兜底统计
                if not used_heideltime:
                    stats['total_fallback'] += 1

                # 4. 准备入库数据
                insert_values.append((
                    row['id'], row['title'], row['raw_time'], row['standard_timestamp'],
                    event_time, row['source'], row['url'], row['true_order'], row['is_noise']
                ))

            # 执行批量入库
            if insert_values:
                insert_sql = """
                    INSERT INTO parser_newsdata 
                    (id, title, raw_time, standard_timestamp, event_timestamp, source, url, true_order, is_noise)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE 
                        event_timestamp = VALUES(event_timestamp),
                        standard_timestamp = VALUES(standard_timestamp)
                """
                cursor.executemany(insert_sql, insert_values)
                conn.commit()
                
                print(f"\n成功扫描处理并入库 {len(insert_values)} 条记录。")
                print(f"================ 诊断报告 ================")
                print(f"英语且 HeidelTime 成功提取时间 : {stats['english_ht_success']} 条")
                print(f"英语但未包含明显时间词汇       : {stats['english_ht_empty']} 条")
                print(f"英语但 HeidelTime 解析崩溃报错 : {stats['english_ht_error']} 条")
                print(f"非英语数据 (直接跳过不调HT)    : {stats['non_english_skipped']} 条")
                print(f"最终使用发布时间兜底的总条数   : {stats['total_fallback']} 条")
                print(f"==========================================")

    except Exception as e:
        print(f"数据库异常: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    extract_and_calculate_event_time()