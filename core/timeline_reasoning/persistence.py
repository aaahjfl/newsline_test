"""MySQL persistence for the timeline reasoning layer."""

from __future__ import annotations

from datetime import date, datetime
import json

from database.db_utils import get_db_connection

from .models import TimelineReasoningResult


RUN_TABLE = "timeline_reasoning_runs"
DECISION_TABLE = "timeline_event_decisions"
NODE_TABLE = "timeline_nodes"
ARTICLE_TABLE = "timeline_node_articles"


def _to_db_datetime(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S")
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("T", " ").replace("Z", "")
    if len(normalized) == 10:
        normalized = f"{normalized} 00:00:00"
    return normalized[:19]


def ensure_timeline_reasoning_schema(cursor) -> None:
    """Create timeline reasoning tables if they do not exist."""
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {RUN_TABLE} (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            reasoning_run_id VARCHAR(191) NOT NULL,
            discovery_run_id VARCHAR(191) NOT NULL,
            topic VARCHAR(255) NOT NULL,
            model_name VARCHAR(191) NULL,
            mode VARCHAR(32) NOT NULL,
            prompt_version VARCHAR(64) NOT NULL,
            input_event_count INT NOT NULL DEFAULT 0,
            review_event_count INT NOT NULL DEFAULT 0,
            accepted_event_count INT NOT NULL DEFAULT 0,
            rejected_event_count INT NOT NULL DEFAULT 0,
            status VARCHAR(32) NOT NULL DEFAULT 'completed',
            config_json LONGTEXT NULL,
            generated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uniq_timeline_reasoning_run (reasoning_run_id),
            KEY idx_timeline_reasoning_runs_topic (topic),
            KEY idx_timeline_reasoning_runs_discovery (discovery_run_id)
        )
        """
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {DECISION_TABLE} (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            reasoning_run_id VARCHAR(191) NOT NULL,
            discovery_run_id VARCHAR(191) NOT NULL,
            topic VARCHAR(255) NOT NULL,
            event_id VARCHAR(191) NOT NULL,
            canonical_title TEXT NULL,
            event_time_start DATETIME NULL,
            event_time_end DATETIME NULL,
            event_time_anchor DATETIME NULL,
            cluster_size INT NOT NULL DEFAULT 0,
            source_count INT NOT NULL DEFAULT 0,
            confidence DECIMAL(6,4) NOT NULL DEFAULT 0.0000,
            system_is_noise BOOLEAN NOT NULL DEFAULT FALSE,
            noise_reason VARCHAR(64) NULL,
            risk_flags LONGTEXT NULL,
            decision_source VARCHAR(32) NULL,
            keep_event BOOLEAN NOT NULL DEFAULT TRUE,
            is_topic_relevant BOOLEAN NOT NULL DEFAULT TRUE,
            final_is_noise BOOLEAN NOT NULL DEFAULT FALSE,
            needs_split BOOLEAN NOT NULL DEFAULT FALSE,
            needs_merge BOOLEAN NOT NULL DEFAULT FALSE,
            display_title TEXT NULL,
            resolved_time_start DATETIME NULL,
            resolved_time_end DATETIME NULL,
            resolved_time_anchor DATETIME NULL,
            decision_confidence DECIMAL(6,4) NOT NULL DEFAULT 0.0000,
            time_confidence DECIMAL(6,4) NOT NULL DEFAULT 0.0000,
            decision_reason TEXT NULL,
            raw_response_json LONGTEXT NULL,
            generated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uniq_timeline_decisions_run_event (reasoning_run_id, event_id),
            KEY idx_timeline_decisions_run (reasoning_run_id),
            KEY idx_timeline_decisions_topic (topic),
            KEY idx_timeline_decisions_keep (keep_event)
        )
        """
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {NODE_TABLE} (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            reasoning_run_id VARCHAR(191) NOT NULL,
            discovery_run_id VARCHAR(191) NOT NULL,
            topic VARCHAR(255) NOT NULL,
            event_id VARCHAR(191) NOT NULL,
            order_index INT NOT NULL,
            canonical_title TEXT NULL,
            display_title TEXT NULL,
            event_time_start DATETIME NULL,
            event_time_end DATETIME NULL,
            event_time_anchor DATETIME NULL,
            resolved_time_start DATETIME NULL,
            resolved_time_end DATETIME NULL,
            resolved_time_anchor DATETIME NULL,
            display_date VARCHAR(32) NULL,
            cluster_size INT NOT NULL DEFAULT 0,
            source_count INT NOT NULL DEFAULT 0,
            member_news_ids LONGTEXT NULL,
            confidence DECIMAL(6,4) NOT NULL DEFAULT 0.0000,
            system_is_noise BOOLEAN NOT NULL DEFAULT FALSE,
            noise_reason VARCHAR(64) NULL,
            decision_source VARCHAR(32) NULL,
            keep_event BOOLEAN NOT NULL DEFAULT TRUE,
            is_topic_relevant BOOLEAN NOT NULL DEFAULT TRUE,
            final_is_noise BOOLEAN NOT NULL DEFAULT FALSE,
            needs_split BOOLEAN NOT NULL DEFAULT FALSE,
            needs_merge BOOLEAN NOT NULL DEFAULT FALSE,
            decision_confidence DECIMAL(6,4) NOT NULL DEFAULT 0.0000,
            time_confidence DECIMAL(6,4) NOT NULL DEFAULT 0.0000,
            decision_reason TEXT NULL,
            risk_flags LONGTEXT NULL,
            generated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uniq_timeline_nodes_run_event (reasoning_run_id, event_id),
            KEY idx_timeline_nodes_run (reasoning_run_id),
            KEY idx_timeline_nodes_topic (topic),
            KEY idx_timeline_nodes_order (reasoning_run_id, order_index),
            KEY idx_timeline_nodes_anchor (resolved_time_anchor)
        )
        """
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ARTICLE_TABLE} (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            reasoning_run_id VARCHAR(191) NOT NULL,
            discovery_run_id VARCHAR(191) NOT NULL,
            topic VARCHAR(255) NOT NULL,
            event_id VARCHAR(191) NOT NULL,
            news_id VARCHAR(128) NOT NULL,
            title TEXT NULL,
            source VARCHAR(255) NULL,
            url TEXT NULL,
            event_time_anchor DATETIME NULL,
            cluster_size INT NOT NULL DEFAULT 0,
            canonical_title TEXT NULL,
            system_is_noise BOOLEAN NOT NULL DEFAULT FALSE,
            noise_reason VARCHAR(64) NULL,
            sort_index INT NOT NULL DEFAULT 0,
            generated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            KEY idx_timeline_articles_run (reasoning_run_id),
            KEY idx_timeline_articles_event (reasoning_run_id, event_id),
            KEY idx_timeline_articles_news (news_id)
        )
        """
    )


def persist_timeline_reasoning_result(
    result: TimelineReasoningResult,
    *,
    config: dict | None = None,
) -> None:
    """Persist a timeline reasoning run and its materialized display records."""
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            ensure_timeline_reasoning_schema(cursor)
            cursor.execute(
                f"""
                INSERT INTO {RUN_TABLE} (
                    reasoning_run_id,
                    discovery_run_id,
                    topic,
                    model_name,
                    mode,
                    prompt_version,
                    input_event_count,
                    review_event_count,
                    accepted_event_count,
                    rejected_event_count,
                    status,
                    config_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    result.reasoning_run_id,
                    result.discovery_run_id,
                    result.topic,
                    result.model_name,
                    result.mode,
                    result.prompt_version,
                    result.input_event_count,
                    result.review_event_count,
                    result.accepted_event_count,
                    result.rejected_event_count,
                    result.status,
                    json.dumps(config or {}, ensure_ascii=False),
                ),
            )

            if result.decisions:
                cards_by_id = dict(result.decision_contexts)
                for record in result.timeline:
                    cards_by_id.setdefault(record.event_id, {
                        "canonical_title": record.canonical_title,
                        "event_time_start": record.event_time_start,
                        "event_time_end": record.event_time_end,
                        "event_time_anchor": record.event_time_anchor,
                        "cluster_size": record.cluster_size,
                        "source_count": record.source_count,
                        "confidence": record.confidence,
                        "system_is_noise": record.system_is_noise,
                        "noise_reason": record.noise_reason,
                        "risk_flags": record.risk_flags,
                    })
                decision_rows = []
                for decision in result.decisions:
                    card_info = cards_by_id.get(decision.event_id, {})
                    decision_rows.append(
                        (
                            result.reasoning_run_id,
                            result.discovery_run_id,
                            result.topic,
                            decision.event_id,
                            card_info.get("canonical_title"),
                            _to_db_datetime(card_info.get("event_time_start")),
                            _to_db_datetime(card_info.get("event_time_end")),
                            _to_db_datetime(card_info.get("event_time_anchor")),
                            int(card_info.get("cluster_size") or 0),
                            int(card_info.get("source_count") or 0),
                            float(card_info.get("confidence") or 0.0),
                            bool(card_info.get("system_is_noise")),
                            card_info.get("noise_reason"),
                            json.dumps(card_info.get("risk_flags") or [], ensure_ascii=False),
                            decision.decision_source,
                            decision.keep_event,
                            decision.is_topic_relevant,
                            decision.final_is_noise,
                            decision.needs_split,
                            decision.needs_merge,
                            decision.display_title,
                            _to_db_datetime(decision.resolved_time_start),
                            _to_db_datetime(decision.resolved_time_end),
                            _to_db_datetime(decision.resolved_time_anchor),
                            decision.decision_confidence,
                            decision.time_confidence,
                            decision.decision_reason,
                            json.dumps(decision.raw_response_json or {}, ensure_ascii=False),
                        )
                    )
                cursor.executemany(
                    f"""
                    INSERT INTO {DECISION_TABLE} (
                        reasoning_run_id,
                        discovery_run_id,
                        topic,
                        event_id,
                        canonical_title,
                        event_time_start,
                        event_time_end,
                        event_time_anchor,
                        cluster_size,
                        source_count,
                        confidence,
                        system_is_noise,
                        noise_reason,
                        risk_flags,
                        decision_source,
                        keep_event,
                        is_topic_relevant,
                        final_is_noise,
                        needs_split,
                        needs_merge,
                        display_title,
                        resolved_time_start,
                        resolved_time_end,
                        resolved_time_anchor,
                        decision_confidence,
                        time_confidence,
                        decision_reason,
                        raw_response_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    decision_rows,
                )

            if result.timeline:
                cursor.executemany(
                    f"""
                    INSERT INTO {NODE_TABLE} (
                        reasoning_run_id,
                        discovery_run_id,
                        topic,
                        event_id,
                        order_index,
                        canonical_title,
                        display_title,
                        event_time_start,
                        event_time_end,
                        event_time_anchor,
                        resolved_time_start,
                        resolved_time_end,
                        resolved_time_anchor,
                        display_date,
                        cluster_size,
                        source_count,
                        member_news_ids,
                        confidence,
                        system_is_noise,
                        noise_reason,
                        decision_source,
                        keep_event,
                        is_topic_relevant,
                        final_is_noise,
                        needs_split,
                        needs_merge,
                        decision_confidence,
                        time_confidence,
                        decision_reason,
                        risk_flags
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            record.reasoning_run_id,
                            record.discovery_run_id,
                            record.topic,
                            record.event_id,
                            record.order_index,
                            record.canonical_title,
                            record.display_title,
                            _to_db_datetime(record.event_time_start),
                            _to_db_datetime(record.event_time_end),
                            _to_db_datetime(record.event_time_anchor),
                            _to_db_datetime(record.resolved_time_start),
                            _to_db_datetime(record.resolved_time_end),
                            _to_db_datetime(record.resolved_time_anchor),
                            record.display_date,
                            record.cluster_size,
                            record.source_count,
                            json.dumps(record.member_news_ids, ensure_ascii=False),
                            record.confidence,
                            record.system_is_noise,
                            record.noise_reason,
                            record.decision_source,
                            record.keep_event,
                            record.is_topic_relevant,
                            record.final_is_noise,
                            record.needs_split,
                            record.needs_merge,
                            record.decision_confidence,
                            record.time_confidence,
                            record.decision_reason,
                            json.dumps(record.risk_flags, ensure_ascii=False),
                        )
                        for record in result.timeline
                    ],
                )

                article_rows = []
                for record in result.timeline:
                    for sort_index, article in enumerate(record.articles, start=1):
                        article_rows.append(
                            (
                                record.reasoning_run_id,
                                record.discovery_run_id,
                                record.topic,
                                record.event_id,
                                str(article.get("news_id") or ""),
                                article.get("title"),
                                article.get("source"),
                                article.get("url"),
                                _to_db_datetime(article.get("event_time_anchor")),
                                int(article.get("cluster_size") or record.cluster_size or 0),
                                article.get("canonical_title") or record.canonical_title,
                                bool(article.get("system_is_noise")),
                                article.get("noise_reason"),
                                sort_index,
                            )
                        )
                if article_rows:
                    cursor.executemany(
                        f"""
                        INSERT INTO {ARTICLE_TABLE} (
                            reasoning_run_id,
                            discovery_run_id,
                            topic,
                            event_id,
                            news_id,
                            title,
                            source,
                            url,
                            event_time_anchor,
                            cluster_size,
                            canonical_title,
                            system_is_noise,
                            noise_reason,
                            sort_index
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        article_rows,
                    )

        connection.commit()
    finally:
        connection.close()
