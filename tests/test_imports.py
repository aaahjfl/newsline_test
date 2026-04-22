"""Minimal import tests for the formal project skeleton."""

import importlib
import unittest


MODULES = [
    "configs",
    "configs.db_config",
    "configs.model_config",
    "configs.pipeline_config",
    "database.db_utils",
    "database.crud",
    "database.db_config",
    "data_pipeline._legacy",
    "data_pipeline.processors.cleaner",
    "data_pipeline.processors.language_stats",
    "data_pipeline.processors.normalizer",
    "data_pipeline.processors.time_parser",
    "data_pipeline.scrapers.gdelt",
    "data_pipeline.scrapers.nyt",
    "data_pipeline.scrapers.rss",
    "core.schemas",
    "core.llm.ollama_client",
    "core.event_discovery.legacy_adapter",
    "core.event_discovery.pipeline",
    "core.timeline_reasoning.legacy_adapter",
    "core.timeline_reasoning.pipeline",
    "core.timeline_builder",
    "services.api_server",
    "frontend.app",
]


class ImportSmokeTest(unittest.TestCase):
    def test_modules_import(self) -> None:
        for module_name in MODULES:
            with self.subTest(module=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))

    def test_legacy_paths_exist(self) -> None:
        from data_pipeline.processors.time_parser import get_legacy_time_parser_paths
        from core.event_discovery.legacy_adapter import get_legacy_event_discovery_paths
        from core.timeline_reasoning.legacy_adapter import get_legacy_timeline_reasoning_paths

        all_paths = {}
        all_paths.update(get_legacy_time_parser_paths())
        all_paths.update(get_legacy_event_discovery_paths())
        all_paths.update(get_legacy_timeline_reasoning_paths())

        for path in all_paths.values():
            with self.subTest(path=path):
                self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
