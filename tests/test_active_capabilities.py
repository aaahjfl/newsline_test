"""Minimal verification for the currently migrated active capabilities."""

from __future__ import annotations

import unittest

from data_pipeline.datasets.gdelt_dataset import normalize_gdelt_time
from data_pipeline.datasets.rss_dataset import get_rss_output_path, is_within_time_window
from data_pipeline.processors.spacy_pipeline import get_active_spacy_legacy_path


class ActiveCapabilitySmokeTest(unittest.TestCase):
    def test_rss_dataset_path_targets_newsdata(self) -> None:
        self.assertTrue(str(get_rss_output_path()).endswith("newsdata/rss_news_dataset.json"))

    def test_rss_time_window_helper(self) -> None:
        self.assertTrue(is_within_time_window(None, days=3))

    def test_gdelt_time_normalization(self) -> None:
        self.assertEqual(normalize_gdelt_time("20260401123045"), "20260401T123045Z")

    def test_spacy_pipeline_path_exists(self) -> None:
        self.assertTrue(get_active_spacy_legacy_path().exists())


if __name__ == "__main__":
    unittest.main()
