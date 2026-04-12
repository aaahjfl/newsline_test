"""Compatibility wrapper for the active RSS dataset builder."""

from data_pipeline.datasets.rss_dataset import build_rss_dataset, RSS_SOURCES


def fetch_legacy_rss_news(sources=None):
    """Compatibility alias that now routes to the migrated RSS dataset builder."""
    if sources is None:
        sources = RSS_SOURCES
    records, added_count, _ = build_rss_dataset(sources)
    return records, added_count
