import pymysql
import spacy
import dateparser
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
nlp_models = {
    'en': spacy.load("en_core_web_sm"),
    'zh-cn': spacy.load("zh_core_web_sm"),
    'zh-tw': spacy.load("zh_core_web_sm"), # 繁体中文复用简体模型进行实体识别
    'es': spacy.load("es_core_news_sm"),
    'fr': spacy.load("fr_core_news_sm"),
    'ru': spacy.load("ru_core_news_sm"),
    'ko': spacy.load("ko_core_news_sm"),
    'uk': spacy.load("uk_core_news_sm")
}
print("模型加载完毕，小语种将自动执行安全降级。")

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

def extract_event_time(title, base_time):
    """
    三级降维分流提取逻辑
    """
    try:
        lang = detect(title)
    except Exception:
        return None, "fallback" # 乱码或纯符号直接走兜底

    # ---------------------------------------------------------
    # 第一梯队：spaCy 精准圈词 + dateparser 计算
    # ---------------------------------------------------------
    nlp_model = nlp_models.get(lang)
    if nlp_model:
        doc = nlp_model(title)
        # 筛选出被标记为日期(DATE)或时间(TIME)的实体
        time_entities = [ent.text for ent in doc.ents if ent.label_ in ['DATE', 'TIME']]
        
        if time_entities:
            target_time_str = time_entities[0]
            parsed_date = dateparser.parse(
                target_time_str,
                settings={'RELATIVE_BASE': base_time, 'TIMEZONE': 'Asia/Shanghai', 'RETURN_AS_TIMEZONE_AWARE': False}
            )
            if parsed_date:
                return parsed_date, "tier1"
        return None, "fallback" # 语言在第一梯队，但没提取出时间词

    # ---------------------------------------------------------
    # 第二梯队：斯瓦希里语 (sw) 专属逻辑，全文盲搜
    # ---------------------------------------------------------
    if lang == 'sw':
        try:
            extracted_dates = search_dates(
                title, 
                languages=['sw'],
                settings={'RELATIVE_BASE': base_time, 'TIMEZONE': 'Asia/Shanghai', 'RETURN_AS_TIMEZONE_AWARE': False}
            )
            if extracted_dates:
                # search_dates 返回格式: [('kesho', datetime.datetime(...))]
                _, parsed_date = extracted_dates[0] 
                if parsed_date:
                    return parsed_date, "tier2"
        except Exception:
            pass
        return None, "fallback"

    # ---------------------------------------------------------
    # 第三梯队：其余长尾小语种，直接返回 None 触发兜底
    # ---------------------------------------------------------
    return None, "fallback"

def process_news_pipeline():
    print("开始执行三级混合 OSINT 时间清洗流水线...")
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cursor:
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
                
                # 安全检查：如果连基准时间都缺失，强制走兜底
                if base_time is None:
                    event_time = None
                    stats['fallback_used'] += 1
                else:
                    # 核心分流处理
                    event_time, strategy = extract_event_time(title, base_time)
                    
                    if strategy == "tier1":
                        stats['tier1_success'] += 1
                    elif strategy == "tier2":
                        stats['tier2_success'] += 1
                    else:
                        # 触发兜底逻辑
                        event_time = base_time 
                        stats['fallback_used'] += 1

                # 准备待入库的数据 (确保字段一一对应)
                insert_values.append((
                    row['id'], row['title'], row['raw_time'], row['standard_timestamp'],
                    event_time, row['source'], row['url'], row['true_order'], row['is_noise']
                ))

            # 执行大批量事务写入 (完美 UPSERT 镜像写入)
            if insert_values:
                insert_sql = """
                    INSERT INTO parser_newsdata 
                    (id, title, raw_time, standard_timestamp, event_timestamp, source, url, true_order, is_noise)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE 
                        title = VALUES(title),
                        raw_time = VALUES(raw_time),
                        standard_timestamp = VALUES(standard_timestamp),
                        event_timestamp = VALUES(event_timestamp),
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
                print(f"sw救场: {stats['tier2_success']} 条")
                print(f"无特征/小语种/无基准: {stats['fallback_used']} 条")
                print(f"==============================================")

    except Exception as e:
        print(f"MySQL 事务执行异常，已自动回滚: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    process_news_pipeline()