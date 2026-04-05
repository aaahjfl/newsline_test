import requests
import zipfile
import io
import pandas as pd
import pymysql
import uuid
from datetime import datetime
import re

# 数据库配置
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '123456', 
    'database': 'news_db',
    'charset': 'utf8mb4'
}

DOMAINS = ["aljazeera.com", "bbc.com", "dw.com", "zaobao.com", "nytimes.com", "xinhuanet.com"]

# GDELT v2 Master List URL
MASTER_LIST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

def get_target_zip_urls(start_date_str, end_date_str):
    """
    获取 Master List 并过滤出指定日期范围内的 CSV 压缩包链接
    """
    print("正在拉取 GDELT...")
    response = requests.get(MASTER_LIST_URL)
    response.raise_for_status()
    
    # 每一行的格式: size hash url
    lines = response.text.strip().split('\n')
    
    # 构建正则表达式，匹配 YYYYMMDD 的 URL
    # 例如：http://data.gdeltproject.org/gdeltv2/20250601000000.export.CSV.zip
    target_urls = []
    for line in lines:
        parts = line.split()
        if len(parts) == 3:
            url = parts[2]
            # 我们只需要 export 的主数据表
            if "export.CSV.zip" in url:
                # 从 URL 中提取时间字符串
                match = re.search(r'/(\d{14})\.export', url)
                if match:
                    file_time_str = match.group(1)[:8] # 只取 YYYYMMDD 比较
                    if start_date_str <= file_time_str <= end_date_str:
                        target_urls.append(url)
    
    print(f"筛选完毕，共找到 {len(target_urls)} 个数据包。")
    return target_urls

def process_and_save(url):
    """
    下载、内存解压、Pandas 过滤并写入 MySQL
    """
    try:
        # 1. 下载 ZIP 包进内存
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        
        # 2. 内存解压并用 Pandas 读取
        with zipfile.ZipFile(io.BytesIO(resp.content)) as thezip:
            csv_filename = thezip.namelist()[0]
            with thezip.open(csv_filename) as thefile:
                # GDELT export 表没有表头，且为 Tab 分隔
                df = pd.read_csv(thefile, sep='\t', header=None, dtype=str, on_bad_lines='skip')
                
                # 提取需要的列 (根据 GDELT 2.0 规范，时间通常在第 1/2 列，URL 在第 60 列)
                # 列索引: 1=SQLDATE(YYYYMMDD), 60=SOURCEURL
                # 注意：这里需要根据你实际关注的列做微调，这里演示基础版
                if df.shape[1] > 60:
                    df = df.iloc[:, [1, 60]] 
                    df.columns = ['raw_time', 'url']
                else:
                    return 0

        # 3. 过滤目标域名 (只要 URL 包含我们的目标域名就留下)
        pattern = '|'.join(DOMAINS)
        df_filtered = df[df['url'].str.contains(pattern, na=False, case=False)].copy()
        
        if df_filtered.empty:
            return 0
            
        # 补充其他字段
        df_filtered['id'] = [f"gdelt_{str(uuid.uuid4())[:8]}" for _ in range(len(df_filtered))]
        df_filtered['title'] = "从URL解析或留空" # 静态 CSV 通常不带标题，可能需要从 URL 截取或后续补全
        df_filtered['source'] = df_filtered['url'].apply(
            lambda x: next((d for d in DOMAINS if d in x.lower()), "Unknown")
        )

        # 4. 批量写入 MySQL
        conn = get_db_connection()
        inserted_count = 0
        try:
            with conn.cursor() as cursor:
                sql = """
                    INSERT IGNORE INTO raw_news_data 
                    (id, title, raw_time, source, url) 
                    VALUES (%s, %s, %s, %s, %s)
                """
                # 转化为元组列表
                values = df_filtered[['id', 'title', 'raw_time', 'source', 'url']].values.tolist()
                inserted_count = cursor.executemany(sql, values)
            conn.commit()
            return inserted_count
        finally:
            conn.close()
            
    except Exception as e:
        print(f"处理 {url} 时出错: {e}")
        return 0

if __name__ == "__main__":
    # 设定你要抓取的日期范围 (字符串 YYYYMMDD)
    START_DATE = "20250601"
    END_DATE = "20260326"
    
    urls = target_urls = get_target_zip_urls(START_DATE, END_DATE)
    
    total_inserted = 0
    for idx, url in enumerate(urls, 1):
        print(f"[{idx}/{len(urls)}] 正在处理: {url.split('/')[-1]}...")
        inserted = process_and_save(url)
        if inserted > 0:
            print(f"  -> 成功写入 {inserted} 条数据。")
        total_inserted += inserted
        
    print(f"\n全部执行完毕，共成功入库 {total_inserted} 条数据")