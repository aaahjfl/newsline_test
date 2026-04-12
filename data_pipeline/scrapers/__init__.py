"""Formal scraper entry points backed by legacy ingestion scripts."""

from .gdelt import run_legacy_gdelt_api_scraper, run_legacy_gdelt_csv_scraper
from .nyt import fetch_legacy_nyt_articles
from .rss import fetch_legacy_rss_news

__all__ = [
    "fetch_legacy_nyt_articles",
    "fetch_legacy_rss_news",
    "run_legacy_gdelt_api_scraper",
    "run_legacy_gdelt_csv_scraper",
]
