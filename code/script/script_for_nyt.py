import requests
import time
import json
import uuid
from datetime import datetime

# ================= 配置区 =================
API_KEY = "替换为你的_NYT_API_KEY"
QUERY = "Ali Larijani Iran"  # 你要搜索的事件关键词
BEGIN_DATE = "20260301"      # 搜索起始日期 YYYYMMDD
END_DATE = "20260328"        # 搜索结束日期 YYYYMMDD
MAX_PAGES = 5                # NYT 每页返回 10 条数据
OUTPUT_FILE = "nyt_news_data.json"
# ==========================================

def fetch_nyt_articles(query, begin_date, end_date, max_pages):
    base_url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
    results = []

    print(f"开始抓取 NYT 数据: 关键词='{query}'...")

    for page in range(max_pages):
        params = {
            "q": query,
            "begin_date": begin_date,
            "end_date": end_date,
            "page": page,
            "api-key": API_KEY,
            # fl 参数限制只返回我们需要的数据，减少带宽消耗
            "fl": "headline,pub_date,web_url,_id" 
        }

        try:
            response = requests.get(base_url, params=params)
            
            # 应对 NYT 严格的速率限制 (免费版通常是 5次/分钟, 500次/天)
            if response.status_code == 429:
                print("触发 API 频率限制，休眠 20 秒...")
                time.sleep(20)
                continue
                
            response.raise_for_status()
            data = response.json()
            
            docs = data.get("response", {}).get("docs", [])
            if not docs:
                print(f"第 {page} 页没有更多数据，抓取结束。")
                break

            for doc in docs:
                # 按照你的 MVP 格式组装数据
                news_item = {
                    "id": f"nyt_{doc.get('_id', str(uuid.uuid4())[:8])}",
                    "title": doc.get("headline", {}).get("main", ""),
                    "raw_time": doc.get("pub_date", ""),
                    "standard_timestamp": None, # 留给后续清洗脚本
                    "source": "The New York Times",
                    "url": doc.get("web_url", ""),
                    "true_order": None,
                    "is_noise": None
                }
                results.append(news_item)
                
            print(f"已获取第 {page} 页，共解析 {len(docs)} 条数据。")
            
            # 必须休眠，防止被封 IP
            time.sleep(12) 

        except Exception as e:
            print(f"抓取第 {page} 页时发生错误: {e}")
            break

    return results

if __name__ == "__main__":
    if API_KEY == "替换为你的_NYT_API_KEY":
        print("请先填写你的 NYT API KEY！")
    else:
        articles = fetch_nyt_articles(QUERY, BEGIN_DATE, END_DATE, MAX_PAGES)
        
        # 保存为 JSON 文件
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
            
        print(f"\n抓取完成！共抓取 {len(articles)} 条数据，已保存至 {OUTPUT_FILE}")