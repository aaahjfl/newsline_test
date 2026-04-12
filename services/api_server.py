"""Service-layer facade for the formal project structure."""

from dataclasses import dataclass
from typing import Any

from core.event_discovery.pipeline import run_event_discovery
from core.event_discovery.legacy_adapter import get_legacy_event_discovery_paths
from core.timeline_reasoning.pipeline import run_timeline_reasoning
from core.timeline_reasoning.legacy_adapter import get_legacy_timeline_reasoning_paths
from data_pipeline.processors.time_parser import (
    list_legacy_time_processors,
    run_time_parser,
    run_time_standardization,
)
from data_pipeline.scrapers import (
    fetch_legacy_nyt_articles,
    fetch_legacy_rss_news,
    run_legacy_gdelt_api_scraper,
    run_legacy_gdelt_csv_scraper,
)


def healthcheck() -> dict[str, str]:
    """Return a lightweight readiness payload."""
    return {"status": "ok", "stage": "skeleton"}


@dataclass
class NewsTimelineService:
    """High-level orchestration facade for future API or UI usage."""

    service_name: str = "newsline"

    def run_pipeline(self, raw_news_items: list[Any]):
        """Execute the formal end-to-end pipeline once implementations land."""
        parsed_news_items = run_time_parser(raw_news_items)
        event_nodes = run_event_discovery(parsed_news_items)
        return run_timeline_reasoning(event_nodes)

    def run_legacy_processing_job(self, processor: str = "spacy_v2"):
        """Route a processing task through the original DB-driven implementation."""
        return run_time_parser(parser=processor)

    def run_legacy_time_standardization(self):
        """Execute the original发布时间标准化任务."""
        return run_time_standardization()

    def list_legacy_processors(self) -> dict[str, str]:
        """Expose available legacy processing jobs."""
        return list_legacy_time_processors()

    def list_legacy_core_modules(self) -> dict[str, dict[str, str]]:
        """Expose archived core-layer module locations for staged migration."""
        return {
            "event_discovery": {
                name: str(path) for name, path in get_legacy_event_discovery_paths().items()
            },
            "timeline_reasoning": {
                name: str(path) for name, path in get_legacy_timeline_reasoning_paths().items()
            },
        }

    def run_legacy_scraper(self, scraper: str, **kwargs: Any):
        """Route ingestion requests to the original scraper implementations."""
        if scraper == "rss":
            return fetch_legacy_rss_news(kwargs.get("sources"))
        if scraper == "gdelt_api":
            return run_legacy_gdelt_api_scraper()
        if scraper == "gdelt_csv":
            return run_legacy_gdelt_csv_scraper(kwargs["start_date"], kwargs["end_date"])
        if scraper == "nyt":
            return fetch_legacy_nyt_articles(
                kwargs["query"],
                kwargs["begin_date"],
                kwargs["end_date"],
                kwargs.get("max_pages", 5),
            )
        raise ValueError(f"Unknown scraper: {scraper}")
