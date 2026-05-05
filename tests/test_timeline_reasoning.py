"""Tests for the formal timeline reasoning loading layer."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core.schemas import EventNode


class _FakeCursor:
    def __init__(self, responses):
        self._responses = list(responses)
        self._current = None

    def execute(self, sql, params=None):
        del sql, params
        self._current = self._responses.pop(0) if self._responses else []

    def fetchone(self):
        if isinstance(self._current, list):
            return self._current[0] if self._current else None
        return self._current

    def fetchall(self):
        if isinstance(self._current, list):
            return list(self._current)
        if self._current is None:
            return []
        return [self._current]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def __init__(self, responses):
        self._cursor = _FakeCursor(responses)

    def cursor(self):
        return self._cursor

    def close(self):
        return None


class TimelineReasoningPipelineTest(unittest.TestCase):
    def test_build_initial_timeline_orders_by_event_time(self) -> None:
        from core.timeline_reasoning.pipeline import build_initial_timeline

        events = [
            EventNode(
                event_id="event_b",
                topic="Fed",
                canonical_title="Later event",
                event_time_anchor="2026-04-03 00:00:00",
            ),
            EventNode(
                event_id="event_a",
                topic="Fed",
                canonical_title="Earlier event",
                event_time_anchor="2026-04-01 00:00:00",
            ),
        ]

        timeline = build_initial_timeline(events)

        self.assertEqual([node.event_id for node in timeline], ["event_a", "event_b"])
        self.assertEqual([node.order_index for node in timeline], [1, 2])

    def test_load_event_nodes_for_timeline_uses_latest_run_by_default(self) -> None:
        from core.timeline_reasoning.pipeline import load_event_nodes_for_timeline

        responses = [
            [{"run_id": "fed_20260412_000001_abcd1234"}],
            [
                {
                    "event_id": "fed_20260412_000001_abcd1234:Fed_event_001",
                    "topic": "Fed",
                    "cluster_size": 2,
                    "canonical_title": "Fed keeps rates unchanged",
                    "representative_news_id": "1",
                    "member_news_ids": "[1, 2]",
                    "event_time_start": "2026-04-01 00:00:00",
                    "event_time_end": "2026-04-02 00:00:00",
                    "event_time_anchor": "2026-04-01 00:00:00",
                    "source_count": 2,
                    "confidence": 0.9234,
                    "system_is_noise": 0,
                    "noise_reason": None,
                }
            ],
        ]
        fake_connections = [_FakeConnection([responses[0]]), _FakeConnection([[{"Field": "event_id"}], responses[1]])]

        with patch("core.timeline_reasoning.pipeline.get_db_connection", side_effect=fake_connections):
            run_id, events = load_event_nodes_for_timeline("Fed")

        self.assertEqual(run_id, "fed_20260412_000001_abcd1234")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].member_news_ids, [1, 2])
        self.assertEqual(events[0].canonical_title, "Fed keeps rates unchanged")
        self.assertFalse(events[0].system_is_noise)

    def test_load_event_assignments_for_timeline_preserves_url(self) -> None:
        from core.timeline_reasoning.pipeline import load_event_assignments_for_timeline

        responses = [
            [
                {
                    "run_id": "run_1",
                    "topic": "Fed",
                    "event_id": "run_1:Fed_event_001",
                    "news_id": "1",
                    "title": "Fed keeps rates unchanged",
                    "source": "Reuters",
                    "url": "https://example.com/1",
                    "event_time_anchor": "2026-04-01 00:00:00",
                    "cluster_size": 2,
                    "canonical_title": "Fed keeps rates unchanged",
                    "system_is_noise": 0,
                    "noise_reason": None,
                }
            ]
        ]

        with patch("core.timeline_reasoning.pipeline.get_db_connection", return_value=_FakeConnection(responses)):
            assignments = load_event_assignments_for_timeline("run_1")

        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0]["url"], "https://example.com/1")
        self.assertEqual(assignments[0]["run_id"], "run_1")
        self.assertFalse(assignments[0]["system_is_noise"])

    def test_run_timeline_reasoning_accepts_topic_or_nodes(self) -> None:
        from core.timeline_reasoning.pipeline import run_timeline_reasoning

        with patch(
            "core.timeline_reasoning.pipeline.load_event_nodes_for_timeline",
            return_value=(
                "run_1",
                [
                    EventNode(
                        event_id="event_1",
                        topic="Fed",
                        canonical_title="Event 1",
                        event_time_anchor="2026-04-01 00:00:00",
                    )
                ],
            ),
        ):
            topic_timeline = run_timeline_reasoning("Fed")

        node_timeline = run_timeline_reasoning(
            [
                EventNode(
                    event_id="event_2",
                    topic="Fed",
                    canonical_title="Event 2",
                    event_time_anchor="2026-04-02 00:00:00",
                )
            ]
        )

        self.assertEqual(topic_timeline[0].event_id, "event_1")
        self.assertEqual(node_timeline[0].event_id, "event_2")

    def test_event_card_keeps_all_articles_but_llm_payload_is_compact(self) -> None:
        from core.timeline_reasoning.event_cards import build_event_cards

        event = EventNode(
            event_id="run_1:Fed_event_001",
            topic="Fed",
            member_news_ids=["1", "2"],
            cluster_size=2,
            canonical_title="Fed keeps rates unchanged",
            event_time_anchor="2026-04-01 00:00:00",
            source_count=2,
            confidence=0.91,
        )
        assignments = [
            {
                "event_id": "run_1:Fed_event_001",
                "news_id": "1",
                "title": "Fed keeps rates unchanged",
                "source": "Reuters",
                "url": "https://example.com/1",
                "event_time_anchor": "2026-04-01 00:00:00",
                "cluster_size": 2,
                "canonical_title": "Fed keeps rates unchanged",
                "system_is_noise": False,
                "noise_reason": None,
            },
            {
                "event_id": "run_1:Fed_event_001",
                "news_id": "2",
                "title": "Federal Reserve leaves interest rates steady",
                "source": "AP",
                "url": "https://example.com/2",
                "event_time_anchor": "2026-04-01 00:00:00",
                "cluster_size": 2,
                "canonical_title": "Fed keeps rates unchanged",
                "system_is_noise": False,
                "noise_reason": None,
            },
        ]

        cards = build_event_cards(discovery_run_id="run_1", events=[event], assignments=assignments)

        self.assertEqual(len(cards), 1)
        self.assertEqual(len(cards[0].articles), 2)
        self.assertEqual(cards[0].articles[1]["url"], "https://example.com/2")
        self.assertIn("evidence", cards[0].to_llm_dict())
        self.assertIn("topic_profile", cards[0].to_llm_dict())
        self.assertNotIn("articles", cards[0].to_llm_dict())
        self.assertNotIn("member_titles_sample", cards[0].to_llm_dict())
        self.assertNotIn("member_title_evidence", cards[0].to_llm_dict())

    def test_event_card_includes_quality_summary_and_diverse_title_evidence(self) -> None:
        from core.timeline_reasoning.event_cards import build_event_cards

        event = EventNode(
            event_id="run_1:Fed_event_001",
            topic="Fed",
            member_news_ids=["1", "2", "3"],
            cluster_size=3,
            canonical_title="Fed keeps rates unchanged",
            event_time_anchor="2026-04-02 00:00:00",
            source_count=3,
            confidence=0.91,
            quality_metrics={
                "semantic_cohesion": 0.88,
                "temporal_coherence": 0.93,
                "duplicate_ratio": 0.0,
                "article_count": 3,
                "ignored_detail": "not sent to llm",
            },
        )
        assignments = [
            {
                "event_id": "run_1:Fed_event_001",
                "news_id": "1",
                "title": "Fed keeps rates unchanged",
                "source": "Reuters",
                "event_time_anchor": "2026-04-02 00:00:00",
            },
            {
                "event_id": "run_1:Fed_event_001",
                "news_id": "2",
                "title": "Federal Reserve leaves interest rates steady",
                "source": "AP",
                "event_time_anchor": "2026-04-01 00:00:00",
            },
            {
                "event_id": "run_1:Fed_event_001",
                "news_id": "3",
                "title": "Fed rate decision live updates",
                "source": "Example",
                "event_time_anchor": "2026-04-03 00:00:00",
            },
        ]

        card = build_event_cards(discovery_run_id="run_1", events=[event], assignments=assignments)[0]
        payload = card.to_llm_dict()

        self.assertNotIn("quality_summary", payload)
        self.assertIn("quality_hints", payload)
        self.assertGreaterEqual(len(payload["evidence"]), 3)
        self.assertLessEqual(len(payload["evidence"]), 4)
        self.assertIn("container_title", {item["role"] for item in payload["evidence"]})
        self.assertNotIn("news_id", payload["evidence"][0])
        self.assertEqual(payload["time"]["anchor"], "2026-04-02")

    def test_rule_routing_sends_system_noise_to_llm_review(self) -> None:
        from core.timeline_reasoning.filters import route_event_card
        from core.timeline_reasoning.models import EventCard

        card = EventCard(
            discovery_run_id="run_1",
            topic="Fed",
            event_id="run_1:Fed_event_001",
            canonical_title="Fed keeps rates unchanged",
            cluster_size=1,
            confidence=0.52,
            system_is_noise=True,
            event_time_anchor="2026-04-01 00:00:00",
        )

        self.assertEqual(route_event_card(card, mode="standard"), "llm_review")
        self.assertIn("system_noise", card.risk_flags)

    def test_fast_routing_does_not_review_low_confidence_singleton_without_structural_risk(self) -> None:
        from core.timeline_reasoning.filters import route_event_card
        from core.timeline_reasoning.models import EventCard

        card = EventCard(
            discovery_run_id="run_1",
            topic="Fed",
            event_id="run_1:Fed_event_001",
            canonical_title="Fed keeps rates unchanged",
            cluster_size=1,
            source_count=1,
            confidence=0.62,
            event_time_anchor="2026-04-01 00:00:00",
        )

        self.assertEqual(route_event_card(card, mode="fast"), "auto_accept")
        self.assertIn("low_confidence", card.risk_flags)

    def test_standard_routing_does_not_force_review_for_translated_alias_risk_alone(self) -> None:
        from core.timeline_reasoning.filters import route_event_card
        from core.timeline_reasoning.models import EventCard
        from core.timeline_reasoning.topic_profile import build_topic_profile

        card = EventCard(
            discovery_run_id="run_1",
            topic="Apple",
            topic_profile=build_topic_profile("Apple"),
            event_id="run_1:Apple_event_001",
            canonical_title="静宁苹果产业品牌价值提升",
            cluster_size=1,
            source_count=1,
            confidence=0.62,
            event_time_anchor="2026-04-01 00:00:00",
            member_titles_sample=["静宁苹果产业品牌价值提升"],
        )

        self.assertEqual(route_event_card(card, mode="standard"), "auto_accept")
        self.assertIn("translated_topic_alias_risk", card.risk_flags)

    def test_standard_routing_reviews_translated_alias_when_structural_risk_exists(self) -> None:
        from core.timeline_reasoning.filters import route_event_card
        from core.timeline_reasoning.models import EventCard
        from core.timeline_reasoning.topic_profile import build_topic_profile

        card = EventCard(
            discovery_run_id="run_1",
            topic="Apple",
            topic_profile=build_topic_profile("Apple"),
            event_id="run_1:Apple_event_001",
            canonical_title="静宁苹果产业品牌价值提升",
            cluster_size=2,
            source_count=1,
            confidence=0.62,
            event_time_start="2026-01-01 00:00:00",
            event_time_end="2026-04-01 00:00:00",
            event_time_anchor="2026-04-01 00:00:00",
            member_titles_sample=["静宁苹果产业品牌价值提升"],
        )

        self.assertEqual(route_event_card(card, mode="standard"), "llm_review")
        self.assertIn("translated_topic_alias_risk", card.risk_flags)
        self.assertIn("long_time_span", card.risk_flags)

    def test_standard_mode_samples_uncertain_events_without_full_review(self) -> None:
        from core.timeline_reasoning.models import EventCard, EventDecision
        from core.timeline_reasoning.pipeline import _route_and_decide

        cards = [
            EventCard(
                discovery_run_id="run_1",
                topic="Fed",
                event_id=f"event_{index}",
                canonical_title=f"Fed event {index}",
                cluster_size=1,
                source_count=1,
                confidence=0.62,
                event_time_anchor="2026-04-01 00:00:00",
            )
            for index in range(12)
        ]

        def fake_judge(review_cards, **kwargs):
            del kwargs
            return [
                EventDecision(
                    event_id=card.event_id,
                    decision_source="llm",
                    keep_event=True,
                    is_topic_relevant=True,
                    final_is_noise=False,
                    display_title=card.canonical_title,
                    resolved_time_anchor=card.event_time_anchor,
                )
                for card in review_cards
            ]

        with patch("core.timeline_reasoning.pipeline.judge_event_cards_with_llm", side_effect=fake_judge):
            decisions, review_count = _route_and_decide(
                cards,
                mode="standard",
                model_name="dummy",
                llm_batch_size=1,
                llm_timeout_seconds=1,
            )

        self.assertEqual(review_count, 4)
        self.assertEqual(len(decisions), 12)
        self.assertEqual(sum(1 for decision in decisions if decision.decision_source == "llm"), 4)

    def test_llm_json_parser_ignores_visible_think_blocks(self) -> None:
        from core.timeline_reasoning.llm_judge import _extract_json_object

        parsed = _extract_json_object(
            '<think>reasoning text</think>{"decisions": [{"event_id": "event_1"}]}'
        )

        self.assertEqual(parsed["decisions"][0]["event_id"], "event_1")

    def test_llm_decision_preserves_explicit_null_time(self) -> None:
        from core.timeline_reasoning.llm_judge import _decision_from_payload
        from core.timeline_reasoning.models import EventCard

        card = EventCard(
            discovery_run_id="run_1",
            topic="Fed",
            event_id="event_1",
            canonical_title="Fed keeps rates unchanged",
            event_time_anchor="2026-04-01 00:00:00",
        )
        decision = _decision_from_payload(
            {
                "event_id": "event_1",
                "keep_event": True,
                "is_topic_relevant": True,
                "final_is_noise": False,
                "resolved_time_anchor": None,
            },
            card,
            {"decisions": []},
        )

        self.assertIsNone(decision.resolved_time_anchor)

    def test_llm_decision_uses_display_title_but_ignores_other_locally_derived_fields(self) -> None:
        from core.timeline_reasoning.llm_judge import _decision_from_payload
        from core.timeline_reasoning.models import EventCard

        card = EventCard(
            discovery_run_id="run_1",
            topic="Fed",
            event_id="event_1",
            canonical_title="Fed keeps rates unchanged",
            confidence=0.82,
            event_time_start="2026-04-01 00:00:00",
            event_time_end="2026-04-01 00:00:00",
            event_time_anchor="2026-04-01 00:00:00",
        )
        decision = _decision_from_payload(
            {
                "event_id": "event_1",
                "keep_event": True,
                "is_topic_relevant": True,
                "final_is_noise": False,
                "display_title": "LLM rewritten title",
                "resolved_time_start": "2026-05-01 00:00:00",
                "resolved_time_end": "2026-05-02 00:00:00",
                "resolved_time_anchor": "2026-04-02 00:00:00",
                "decision_confidence": 0.1,
                "time_confidence": 0.1,
                "decision_reason": "Relevant concrete event.",
            },
            card,
            {"decisions": []},
        )

        self.assertEqual(decision.display_title, "LLM rewritten title")
        self.assertEqual(decision.resolved_time_start, card.event_time_start)
        self.assertEqual(decision.resolved_time_end, card.event_time_end)
        self.assertEqual(decision.resolved_time_anchor, "2026-04-02 00:00:00")
        self.assertEqual(decision.decision_confidence, 0.82)
        self.assertEqual(decision.time_confidence, 0.8)

    def test_prompt_excludes_locally_derived_decision_fields(self) -> None:
        from core.timeline_reasoning.models import EventCard
        from core.timeline_reasoning.prompts import build_event_decision_prompt

        prompt = build_event_decision_prompt(
            [
                EventCard(
                    discovery_run_id="run_1",
                    topic="Fed",
                    event_id="event_1",
                    canonical_title="Fed keeps rates unchanged",
                    event_time_anchor="2026-04-01 00:00:00",
                )
            ]
        )

        self.assertIn('"display_title"', prompt)
        self.assertNotIn('"resolved_time_start"', prompt)
        self.assertNotIn('"resolved_time_end"', prompt)
        self.assertNotIn('"decision_confidence"', prompt)
        self.assertNotIn('"time_confidence"', prompt)
        self.assertIn('"resolved_time_anchor"', prompt)

    def test_run_timeline_reasoning_pipeline_exports_json_and_preserves_articles(self) -> None:
        from core.timeline_reasoning.models import EventDecision
        from core.timeline_reasoning.pipeline import run_timeline_reasoning_pipeline

        events = [
            EventNode(
                event_id="run_1:Fed_event_001",
                topic="Fed",
                member_news_ids=["1", "2"],
                cluster_size=2,
                canonical_title="Fed keeps rates unchanged",
                event_time_anchor="2026-04-01 00:00:00",
                source_count=2,
                confidence=0.91,
            ),
            EventNode(
                event_id="run_1:Fed_event_002",
                topic="Fed",
                member_news_ids=["3"],
                cluster_size=1,
                canonical_title="Live updates: Fed decision",
                event_time_anchor="2026-04-02 00:00:00",
                source_count=1,
                confidence=0.61,
                system_is_noise=True,
                noise_reason="low_cluster_confidence",
            ),
        ]
        assignments = [
            {
                "event_id": "run_1:Fed_event_001",
                "news_id": "1",
                "title": "Fed keeps rates unchanged",
                "source": "Reuters",
                "url": "https://example.com/1",
                "event_time_anchor": "2026-04-01 00:00:00",
                "cluster_size": 2,
                "canonical_title": "Fed keeps rates unchanged",
                "system_is_noise": False,
                "noise_reason": None,
            },
            {
                "event_id": "run_1:Fed_event_001",
                "news_id": "2",
                "title": "Federal Reserve leaves interest rates steady",
                "source": "AP",
                "url": "https://example.com/2",
                "event_time_anchor": "2026-04-01 00:00:00",
                "cluster_size": 2,
                "canonical_title": "Fed keeps rates unchanged",
                "system_is_noise": False,
                "noise_reason": None,
            },
            {
                "event_id": "run_1:Fed_event_002",
                "news_id": "3",
                "title": "Live updates: Fed decision",
                "source": "Example",
                "url": "https://example.com/3",
                "event_time_anchor": "2026-04-02 00:00:00",
                "cluster_size": 1,
                "canonical_title": "Live updates: Fed decision",
                "system_is_noise": True,
                "noise_reason": "low_cluster_confidence",
            },
        ]

        def fake_judge(cards, **kwargs):
            del kwargs
            self.assertEqual([card.event_id for card in cards], ["run_1:Fed_event_002"])
            return [
                EventDecision(
                    event_id="run_1:Fed_event_002",
                    decision_source="llm",
                    keep_event=False,
                    is_topic_relevant=True,
                    final_is_noise=True,
                    display_title="Live updates: Fed decision",
                    resolved_time_anchor="2026-04-02 00:00:00",
                    decision_confidence=0.8,
                    time_confidence=0.7,
                    decision_reason="Rolling coverage is not a concrete event.",
                )
            ]

        with tempfile.TemporaryDirectory() as tempdir:
            with patch(
                "core.timeline_reasoning.pipeline.PIPELINE_CONFIG",
                {"output_root": tempdir},
            ), patch(
                "core.timeline_reasoning.pipeline.load_event_nodes_for_timeline",
                return_value=("run_1", events),
            ), patch(
                "core.timeline_reasoning.pipeline.load_event_assignments_for_timeline",
                return_value=assignments,
            ), patch(
                "core.timeline_reasoning.pipeline.load_event_graph_summary_for_timeline",
                return_value={},
            ), patch(
                "core.timeline_reasoning.pipeline.judge_event_cards_with_llm",
                side_effect=fake_judge,
            ), patch(
                "core.timeline_reasoning.pipeline.persist_timeline_reasoning_result"
            ):
                result = run_timeline_reasoning_pipeline("Fed", dry_run=True)
                self.assertTrue(Path(result.output_paths["timeline_json"]).exists())

        self.assertEqual(result.input_event_count, 2)
        self.assertEqual(result.review_event_count, 1)
        self.assertEqual(result.accepted_event_count, 1)
        self.assertEqual(result.rejected_event_count, 1)
        self.assertEqual(len(result.timeline), 1)
        self.assertEqual(len(result.timeline[0].articles), 2)
        self.assertTrue(result.output_paths["timeline_json"].endswith(".json"))


if __name__ == "__main__":
    unittest.main()
