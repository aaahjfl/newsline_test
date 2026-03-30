import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "newsdata")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rss_news_dataset.json")
import feedparser
import json
import uuid
import time
import os
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime # 专门对付 RSS 奇葩时间格式的神器

# ================= 配置区 =================
RSS_SOURCES = {
    "The New York Times": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "BBC": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "联合早报": "https://www.zaobao.com/realtime/world/rss",
    "DW": "https://rss.dw.com/xml/rss-en-all",
    "新华网": "http://www.xinhuanet.com/english/rss/world.xml",
    "新华网-国际新闻": "http://www.xinhuanet.com/world/news_world.xml",
    "联合早报": "https://rsshub.app/zaobao/realtime/world"
}

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rss_news_dataset.json")

# 控制参数
MAX_ITEMS_PER_SOURCE = 20  # 每个源单次最多保留多少条
DAYS_LOOKBACK = 3          # 时间范围：只保留最近 3 天内的新闻（用来过滤陈年老贴）
# ==========================================

def load_existing_data(filepath):
    """读取已有的数据集，用于去重"""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def is_within_time_window(raw_time_str, days):
    """判断新闻是否在时间窗口内"""
    if not raw_time_str:
        return True # 如果没有时间字段，默认放行交给后续 NLP 处理
    
    try:
        # 尝试将 RSS 的标准时间转化为 datetime 对象
        pub_date = parsedate_to_datetime(raw_time_str)
        now = datetime.now(timezone.utc)
        time_diff = now - pub_date
        return time_diff <= timedelta(days=days)
    except Exception:
        # 解析失败的新闻也先放行，后续用 HeidelTime 处理
        return True

def fetch_rss_news_pro(sources):
    existing_data = load_existing_data(OUTPUT_FILE)
    # 用集合存储现有的 URL，利用哈希实现 O(1) 极速去重
    existing_urls = {item["url"] for item in existing_data if "url" in item}
    
    new_data_count = 0
    
    for source_name, rss_url in sources.items():
        print(f"正在抓取: {source_name} ...")
        try:
            # 1. 戴上面具：伪装成 macOS 下的 Chrome 浏览器
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            # 2. 用 requests 加载数据，加上 10 秒超时防止卡死
            import requests # 记得在文件开头 import requests
            response = requests.get(rss_url, headers=headers, timeout=10)
            
            # 如果网站报错（比如 403 被拦截），直接打印出来并跳过
            if response.status_code != 200:
                print(f"  -> 抓取失败: HTTP 状态码 {response.status_code}")
                continue
                
            # 3. 将拿到的安全文本喂给 feedparser 解析
            feed = feedparser.parse(response.content)
            
            # 如果解析成功但没内容，也给个明确的提示
            if not feed.entries:
                print("  -> 警告：抓取成功但未发现文章，可能 RSS 源为空或页面结构改变。")
                continue
                
            source_count = 0
            for entry in feed.entries:
                if source_count >= MAX_ITEMS_PER_SOURCE:
                    break # 达到该源的条数上限
                    
                url = entry.get("link", "")
                # 去重判定：如果 URL 已经在我们库里了，直接跳过
                if url in existing_urls:
                    continue
                    
                title = entry.get("title", "").strip()
                raw_time = entry.get("published", entry.get("updated", ""))
                
                if not title:
                    continue
                    
                # 时间范围过滤
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
                    "is_noise": None
                }
                
                existing_data.append(news_item)
                existing_urls.add(url) # 把新 URL 加进防重集合
                source_count += 1
                new_data_count += 1
                
            print(f"  -> 抓取并保留了 {source_count} 条新数据。")
            
        except Exception as e:
            print(f"  -> 抓取失败: {e}")
            
        time.sleep(2)
        
    return existing_data, new_data_count

if __name__ == "__main__":
    print(f"开始执行 RSS 增量抓取任务 (时间窗: 近 {DAYS_LOOKBACK} 天, 限额: {MAX_ITEMS_PER_SOURCE}条/源)...\n")
    
    updated_dataset, added_count = fetch_rss_news_pro(RSS_SOURCES)
    
    if added_count > 0:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(updated_dataset, f, ensure_ascii=False, indent=2)
        print(f"\n抓取完成！本次新增 {added_count} 条数据，数据集总规模达到 {len(updated_dataset)} 条。")
    else:
        print("\n抓取完成！没有发现新的新闻，数据集规模未变。")