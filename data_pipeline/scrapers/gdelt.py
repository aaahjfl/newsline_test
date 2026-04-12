"""Compatibility wrappers for current and historical GDELT ingestion flows."""

from __future__ import annotations

from pathlib import Path

from data_pipeline.datasets.gdelt_dataset import run_incremental_gdelt_dataset_build
from data_pipeline._legacy import load_legacy_module


def run_legacy_gdelt_api_scraper():
    """Compatibility alias that now routes to the migrated active GDELT builder."""
    return run_incremental_gdelt_dataset_build()


def run_legacy_gdelt_csv_scraper(start_date: str, end_date: str):
    """Execute the older CSV-based GDELT backfill flow."""
    project_root = Path(__file__).resolve().parents[2]
    module_path = project_root / "code" / "script" / "script_forcsv.py"
    module = load_legacy_module("legacy_gdelt_csv_scraper", module_path)
    urls = module.get_target_zip_urls(start_date, end_date)
    inserted_total = 0
    for url in urls:
        inserted_total += module.process_and_save(url)
    return inserted_total
