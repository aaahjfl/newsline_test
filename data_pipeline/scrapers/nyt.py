"""Compatibility wrapper for the legacy New York Times scraper."""

from __future__ import annotations

from pathlib import Path

from data_pipeline._legacy import load_legacy_module


def fetch_legacy_nyt_articles(query: str, begin_date: str, end_date: str, max_pages: int):
    """Call the original NYT fetch function unchanged."""
    project_root = Path(__file__).resolve().parents[2]
    module_path = project_root / "code" / "script" / "script_for_nyt.py"
    module = load_legacy_module("legacy_nyt_scraper", module_path)
    return module.fetch_nyt_articles(query, begin_date, end_date, max_pages)
