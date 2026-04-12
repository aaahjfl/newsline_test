"""Dataset-building configuration for currently active MVP capabilities."""

from datetime import datetime, timezone


DATASET_CONFIG = {
    "rss_output_filename": "rss_news_dataset.json",
    "rss_sources": {
        "The New York Times": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "BBC": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
        "联合早报": "https://rsshub.app/zaobao/realtime/world",
        "DW": "https://rss.dw.com/xml/rss-en-all",
        "新华网": "http://www.xinhuanet.com/english/rss/world.xml",
        "新华网-国际新闻": "http://www.xinhuanet.com/world/news_world.xml",
    },
    "rss_max_items_per_source": 20,
    "rss_days_lookback": 3,
    "gdelt_domains": {
        "Al Jazeera": "aljazeera.com",
        "BBC": "bbc.com",
        "DW": "dw.com",
        "The New York Times": "nytimes.com",
        "新华网": "xinhuanet.com",
        "亚洲新闻台": "channelnewsasia.com",
    },
    "gdelt_default_start_date": datetime(2025, 6, 1, tzinfo=timezone.utc),
    "gdelt_end_date": datetime(2026, 4, 1, tzinfo=timezone.utc),
    "gdelt_days_per_step": 7,
    "http_headers": {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    },
}
