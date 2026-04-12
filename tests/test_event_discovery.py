"""Tests for the formal SBERT event discovery layer."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import numpy as np

from core.schemas import NewsItem


def _normalized(array: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


class EventDiscoveryPipelineTest(unittest.TestCase):
    def test_oversized_component_is_refined(self) -> None:
        from core.event_discovery.clustering import cluster_embeddings

        news_items = [
            NewsItem(news_id=f"id_{index}", title=f"Trump event {index}", event_time_anchor="2026-04-01 00:00:00")
            for index in range(6)
        ]
        embeddings = np.zeros((6, 2), dtype=np.float32)
        similarity_matrix = np.asarray(
            [
                [1.0, 0.93, 0.92, 0.0, 0.0, 0.0],
                [0.93, 1.0, 0.91, 0.81, 0.0, 0.0],
                [0.92, 0.91, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.81, 0.0, 1.0, 0.92, 0.93],
                [0.0, 0.0, 0.0, 0.92, 1.0, 0.91],
                [0.0, 0.0, 0.0, 0.93, 0.91, 1.0],
            ],
            dtype=np.float32,
        )

        with patch("core.event_discovery.clustering.OVERSIZED_COMPONENT_LIMIT", 4):
            with patch("core.event_discovery.clustering.np.matmul", return_value=similarity_matrix):
                clusters, edges, _ = cluster_embeddings(news_items, embeddings, topic="Trump")

        self.assertEqual(sorted(cluster.cluster_size for cluster in clusters), [3, 3])
        edge_pairs = {(edge.left_index, edge.right_index) for edge in edges}
        self.assertNotIn((1, 3), edge_pairs)

    def test_low_cohesion_medium_component_is_refined(self) -> None:
        from core.event_discovery.clustering import cluster_embeddings

        news_items = [
            NewsItem(news_id=f"id_{index}", title=f"Apple event {index}", event_time_anchor="2026-04-01 00:00:00")
            for index in range(8)
        ]
        embeddings = np.zeros((8, 2), dtype=np.float32)
        similarity_matrix = np.asarray(
            [
                [1.0, 0.94, 0.93, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.94, 1.0, 0.92, 0.0, 0.0, 0.81, 0.0, 0.0],
                [0.93, 0.92, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.93, 0.92, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.93, 1.0, 0.91, 0.0, 0.0],
                [0.0, 0.81, 0.0, 0.92, 0.91, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.94],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.94, 1.0],
            ],
            dtype=np.float32,
        )

        with patch("core.event_discovery.clustering.np.matmul", return_value=similarity_matrix):
            clusters, _, _ = cluster_embeddings(news_items, embeddings, topic="Apple")

        self.assertEqual(sorted(cluster.cluster_size for cluster in clusters), [2, 3, 3])

    def test_topic_matcher_avoids_grapples_false_positive(self) -> None:
        from core.event_discovery.pipeline import _title_matches_topic

        self.assertTrue(_title_matches_topic("Apple sued over AI training", "Apple"))
        self.assertFalse(_title_matches_topic("World grapples with economic fallout", "Apple"))
        self.assertFalse(_title_matches_topic("Is an apple a day really good for your health ?", "Apple"))
        self.assertTrue(_title_matches_topic("苹果公司发布新产品", "苹果"))

    def test_topic_alias_expansion_is_extensible(self) -> None:
        from core.event_discovery.topic_expansion import expand_topic_aliases

        def fake_translate(topic: str, src_lang: str, tgt_lang: str) -> str | None:
            del src_lang
            table = {
                ("苹果", "en"): "Apple",
                ("苹果", "es"): "Apple",
                ("苹果", "fr"): "Apple",
            }
            return table.get((topic, tgt_lang))

        with patch("core.event_discovery.topic_expansion.detect_topic_language", return_value="zh-cn"):
            with patch("core.event_discovery.topic_expansion._translate_topic_once", side_effect=fake_translate):
                aliases = expand_topic_aliases("苹果", target_langs=["en", "es", "fr"])

        self.assertEqual(aliases[0], "苹果")
        self.assertIn("Apple", aliases)

    def test_named_entity_topic_expansion_stays_conservative(self) -> None:
        from core.event_discovery.topic_expansion import expand_topic_aliases

        with patch("core.event_discovery.topic_expansion.detect_topic_language", return_value="en"):
            with patch("core.event_discovery.topic_expansion._translate_topic_once") as translate_mock:
                aliases = expand_topic_aliases("Apple", target_langs=["en", "es", "fr", "zh-cn", "ru"])

        self.assertEqual(translate_mock.call_count, 0)
        self.assertEqual(aliases, ["Apple"])
        self.assertEqual(aliases[0], "Apple")

    def test_encoder_uses_fixed_prompt(self) -> None:
        from core.event_discovery.encoder import EMBEDDING_PROMPT, encode_titles

        class DummyModel:
            def __init__(self) -> None:
                self.kwargs = None

            def encode(self, titles, **kwargs):
                self.kwargs = kwargs
                return np.asarray([[1.0, 0.0]], dtype=np.float32)

        dummy_model = DummyModel()
        with patch("core.event_discovery.encoder.load_embedding_model", return_value=dummy_model):
            vectors = encode_titles(["Example title"])

        self.assertEqual(vectors.shape, (1, 2))
        self.assertEqual(dummy_model.kwargs["prompt"], EMBEDDING_PROMPT)

    def test_topic_run_produces_events_and_exports(self) -> None:
        from core.event_discovery.pipeline import run_event_discovery

        sample_news = [
            NewsItem(
                news_id=1,
                title="Fed holds rates steady after policy meeting",
                source="Reuters",
                url="https://example.com/1",
                event_time_anchor="2026-04-01 00:00:00",
                event_time_start="2026-04-01 00:00:00",
                event_time_end="2026-04-01 00:00:00",
            ),
            NewsItem(
                news_id=2,
                title="Fed keeps interest rates unchanged",
                source="AP",
                url="https://example.com/2",
                event_time_anchor="2026-04-02 00:00:00",
                event_time_start="2026-04-02 00:00:00",
                event_time_end="2026-04-02 00:00:00",
            ),
            NewsItem(
                news_id=3,
                title="Markets rally on tech earnings optimism",
                source="Bloomberg",
                url="https://example.com/3",
                event_time_anchor="2026-04-03 00:00:00",
                event_time_start="2026-04-03 00:00:00",
                event_time_end="2026-04-03 00:00:00",
            ),
        ]
        embeddings = _normalized([[1.0, 0.0], [0.98, 0.08]])

        with TemporaryDirectory() as tempdir:
            with patch("core.event_discovery.pipeline.OUTPUTS_DIR", Path(tempdir)):
                with patch("core.event_discovery.pipeline.fetch_candidate_news", return_value=sample_news):
                    with patch("core.event_discovery.pipeline.encode_titles", return_value=embeddings):
                        with patch("core.event_discovery.pipeline.persist_result_to_db"):
                            result = run_event_discovery("Fed", limit=10)

            export_paths = list(result.output_paths.values())
            for path in export_paths:
                self.assertTrue(Path(path).exists())

            events_payload = json.loads(Path(result.output_paths["events"]).read_text(encoding="utf-8"))
            assignments_payload = json.loads(Path(result.output_paths["assignments"]).read_text(encoding="utf-8"))

        self.assertEqual(result.topic, "Fed")
        self.assertTrue(result.run_id.startswith("Fed_"))
        self.assertEqual(result.candidate_count, 3)
        self.assertEqual(result.filtered_count, 2)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(len(result.assignments), 2)
        self.assertGreaterEqual(len(result.graph_edges), 1)
        self.assertEqual(events_payload["run_id"], result.run_id)
        self.assertEqual(assignments_payload["run_id"], result.run_id)
        self.assertEqual(assignments_payload["assignments"][0]["run_id"], result.run_id)
        self.assertIn("url", assignments_payload["assignments"][0])
        self.assertIn("system_is_noise", assignments_payload["assignments"][0])

    def test_empty_result_does_not_crash(self) -> None:
        from core.event_discovery.pipeline import run_event_discovery

        with TemporaryDirectory() as tempdir:
            with patch("core.event_discovery.pipeline.OUTPUTS_DIR", Path(tempdir)):
                with patch("core.event_discovery.pipeline.fetch_candidate_news", return_value=[]):
                    with patch("core.event_discovery.pipeline.persist_result_to_db"):
                        result = run_event_discovery("NoHits")

            export_paths = list(result.output_paths.values())
            for path in export_paths:
                self.assertTrue(Path(path).exists())

        self.assertEqual(result.candidate_count, 0)
        self.assertEqual(result.filtered_count, 0)
        self.assertEqual(result.events, [])
        self.assertEqual(result.assignments, [])
        self.assertEqual(result.graph_edges, [])

    def test_single_news_still_forms_event(self) -> None:
        from core.event_discovery.pipeline import run_event_discovery

        sample_news = [
            NewsItem(
                news_id=99,
                title="Single article about one isolated event",
                source="BBC",
                url="https://example.com/single",
                event_time_anchor="2026-04-08 00:00:00",
                event_time_start="2026-04-08 00:00:00",
                event_time_end="2026-04-08 00:00:00",
            )
        ]

        with TemporaryDirectory() as tempdir:
            with patch("core.event_discovery.pipeline.OUTPUTS_DIR", Path(tempdir)):
                with patch("core.event_discovery.pipeline.fetch_candidate_news", return_value=sample_news):
                    with patch(
                        "core.event_discovery.pipeline.encode_titles",
                        return_value=np.asarray([[1.0, 0.0]], dtype=np.float32),
                    ):
                        with patch("core.event_discovery.pipeline.persist_result_to_db"):
                            result = run_event_discovery("Single")

        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].cluster_size, 1)
        self.assertEqual(result.events[0].member_news_ids, [99])
        self.assertTrue(result.events[0].event_id.startswith(f"{result.run_id}:"))

    def test_event_fields_are_complete(self) -> None:
        from core.event_discovery.pipeline import run_event_discovery

        sample_news = [
            NewsItem(
                news_id=7,
                title="Topic event title",
                source="Xinhua",
                url="https://example.com/topic",
                event_time_anchor="2026-04-05 00:00:00",
                event_time_start="2026-04-05 00:00:00",
                event_time_end="2026-04-05 00:00:00",
                is_noise=False,
            ),
            NewsItem(
                news_id=8,
                title="Another Topic event title",
                source="Xinhua",
                url="https://example.com/noise",
                event_time_anchor="2026-04-05 00:00:00",
                event_time_start="2026-04-05 00:00:00",
                event_time_end="2026-04-05 00:00:00",
                is_noise=True,
            ),
        ]

        with TemporaryDirectory() as tempdir:
            with patch("core.event_discovery.pipeline.OUTPUTS_DIR", Path(tempdir)):
                with patch("core.event_discovery.pipeline.fetch_candidate_news", return_value=sample_news):
                    with patch(
                        "core.event_discovery.pipeline.encode_titles",
                        return_value=np.asarray([[1.0, 0.0], [0.99, 0.05]], dtype=np.float32),
                    ):
                        with patch("core.event_discovery.pipeline.persist_result_to_db"):
                            result = run_event_discovery("Topic")

        self.assertEqual(result.candidate_count, 2)
        self.assertEqual(result.filtered_count, 2)
        event_payload = result.events[0].to_dict()
        self.assertEqual(result.assignments[0]["url"], "https://example.com/topic")
        self.assertEqual(result.assignments[0]["run_id"], result.run_id)
        self.assertEqual(
            set(event_payload),
            {
                "event_id",
                "topic",
                "member_news_ids",
                "cluster_size",
                "canonical_title",
                "representative_news_id",
                "event_time_start",
                "event_time_end",
                "event_time_anchor",
                "source_count",
                "confidence",
                "system_is_noise",
                "noise_reason",
            },
        )

    def test_parser_is_noise_is_not_used_as_filter(self) -> None:
        from core.event_discovery.pipeline import run_event_discovery

        sample_news = [
            NewsItem(
                news_id=1,
                title="Topic singleton title",
                source="BBC",
                url="https://example.com/a",
                event_time_anchor="2026-04-08 00:00:00",
                event_time_start="2026-04-08 00:00:00",
                event_time_end="2026-04-08 00:00:00",
                is_noise=True,
            )
        ]

        with TemporaryDirectory() as tempdir:
            with patch("core.event_discovery.pipeline.OUTPUTS_DIR", Path(tempdir)):
                with patch("core.event_discovery.pipeline.fetch_candidate_news", return_value=sample_news):
                    with patch(
                        "core.event_discovery.pipeline.encode_titles",
                        return_value=np.asarray([[1.0, 0.0]], dtype=np.float32),
                    ):
                        with patch("core.event_discovery.pipeline.persist_result_to_db"):
                            result = run_event_discovery("Topic")

        self.assertEqual(result.filtered_count, 1)
        self.assertEqual(len(result.events), 1)
        self.assertTrue(result.events[0].system_is_noise)
        self.assertEqual(result.events[0].noise_reason, "singleton_component")


if __name__ == "__main__":
    unittest.main()
