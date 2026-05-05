"""Formal event discovery pipeline entry point."""

from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import re
from typing import Any
import unicodedata
import warnings
from uuid import uuid4

from configs.pipeline_config import PIPELINE_CONFIG
from configs.path_config import OUTPUTS_DIR
from database.db_utils import get_db_connection

from .clustering import cluster_embeddings
from .encoder import encode_titles
from .event_builder import build_event_nodes
from .title_features import detect_title_risk_flags, normalize_title_for_matching
from .topic_expansion import TopicAlias, expand_topic_alias_candidates
from core.schemas import EventDiscoveryResult, NewsItem

EVENT_TABLE = "event_discovery_events"
ASSIGNMENT_TABLE = "event_discovery_assignments"
GRAPH_TABLE = "event_discovery_graph"


def _sanitize_topic_token(topic: str) -> str:
    normalized = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", topic.strip(), flags=re.UNICODE)
    collapsed = re.sub(r"_+", "_", normalized).strip("_")
    return collapsed or "topic"


def _generate_run_id(topic: str) -> str:
    topic_token = _sanitize_topic_token(topic)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{topic_token}_{timestamp}_{uuid4().hex[:8]}"


def _normalize_match_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _ascii_word_match(text: str, token: str, *, case_sensitive: bool) -> bool:
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(token)}(?:['’]s)?(?![A-Za-z0-9])",
        flags=flags,
    )
    return bool(pattern.search(text))


def _title_matches_topic(title: str, topic: str) -> bool:
    raw_topic = _normalize_match_text(topic)
    raw_title = _normalize_match_text(title)
    if not raw_topic or not raw_title:
        return False

    if _contains_cjk(raw_topic):
        return raw_topic.casefold() in raw_title.casefold()

    ascii_tokens = re.findall(r"[A-Za-z0-9]+", raw_topic)
    if not ascii_tokens:
        return raw_topic.casefold() in raw_title.casefold()

    prefer_case_sensitive = any(char.isupper() for char in raw_topic) and not raw_topic.isupper()
    if prefer_case_sensitive:
        return all(
            _ascii_word_match(raw_title, token, case_sensitive=True)
            or _ascii_word_match(raw_title, token.upper(), case_sensitive=True)
            for token in ascii_tokens
        )

    return all(_ascii_word_match(raw_title, token, case_sensitive=False) for token in ascii_tokens)


def _serialize_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time()).isoformat(sep=" ", timespec="seconds")
    text = str(value).strip()
    return text or None


def _is_noise(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _normalize_date_bound(value: str | date | datetime | None, *, end_of_day: bool = False) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        base = datetime.combine(value, datetime.max.time() if end_of_day else datetime.min.time())
        return base.strftime("%Y-%m-%d %H:%M:%S")

    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt)
            if fmt == "%Y-%m-%d" and end_of_day:
                parsed = datetime.combine(parsed.date(), datetime.max.time())
            return parsed.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    raise ValueError("date bounds must use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format.")


def fetch_candidate_news(
    topic: str,
    limit: int | None = None,
    *,
    aliases: list[str] | None = None,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> list[NewsItem]:
    """Fetch candidate news rows from `parser_newsdata` by topic."""
    if not topic or not topic.strip():
        raise ValueError("topic must be a non-empty string.")

    start_bound = _normalize_date_bound(start_date)
    end_bound = _normalize_date_bound(end_date, end_of_day=True)
    if start_bound and end_bound and start_bound > end_bound:
        raise ValueError("start_date must be earlier than or equal to end_date.")

    topic_aliases = aliases or [topic.strip()]
    topic_aliases = [alias for alias in topic_aliases if str(alias).strip()]
    if not topic_aliases:
        topic_aliases = [topic.strip()]
    like_clauses = " OR ".join(["title LIKE %s" for _ in topic_aliases])
    time_expr = "COALESCE(event_timestamp, event_time_start, event_time_end, standard_timestamp)"

    sql = """
        SELECT
            id,
            title,
            source,
            url,
            standard_timestamp,
            event_timestamp,
            event_time_start,
            event_time_end,
            time_granularity,
            is_noise
        FROM parser_newsdata
        WHERE title IS NOT NULL
          AND TRIM(title) <> ''
          AND (
    """ + like_clauses + """
          )
    """
    params: list[Any] = [f"%{alias}%" for alias in topic_aliases]
    if start_bound:
        sql += f" AND {time_expr} >= %s\n"
        params.append(start_bound)
    if end_bound:
        sql += f" AND {time_expr} <= %s\n"
        params.append(end_bound)
    sql += """
        ORDER BY COALESCE(
            event_timestamp,
            event_time_start,
            event_time_end,
            standard_timestamp,
            CAST('9999-12-31 23:59:59' AS DATETIME)
        ) ASC, id ASC
    """
    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)

    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            rows = list(cursor.fetchall())
    finally:
        connection.close()

    return [
        NewsItem(
            news_id=row["id"],
            title=str(row.get("title", "")).strip(),
            source=row.get("source"),
            url=row.get("url"),
            publish_time=_serialize_datetime(row.get("standard_timestamp")),
            event_time_anchor=_serialize_datetime(row.get("event_timestamp")),
            event_time_start=_serialize_datetime(row.get("event_time_start")),
            event_time_end=_serialize_datetime(row.get("event_time_end")),
            time_granularity=row.get("time_granularity"),
            is_noise=_is_noise(row.get("is_noise")),
        )
        for row in rows
    ]


def _filter_candidates(news_items: list[NewsItem], topic_aliases: list[str]) -> list[NewsItem]:
    return [
        item
        for item in news_items
        if item.title.strip() and any(_title_matches_topic(item.title, alias) for alias in topic_aliases)
    ]


def _prepare_news_for_clustering(news_items: list[NewsItem]) -> list[NewsItem]:
    """Annotate generic title features and collapse exact normalized duplicates."""
    groups: dict[str, list[NewsItem]] = {}
    ordered_keys: list[str] = []

    for item in news_items:
        normalized_title = normalize_title_for_matching(item.title)
        risk_flags = detect_title_risk_flags(item.title)
        item.metadata["normalized_title"] = normalized_title
        item.metadata["title_risk_flags"] = risk_flags

        key = normalized_title or item.title.strip().casefold() or str(item.news_id)
        if key not in groups:
            groups[key] = []
            ordered_keys.append(key)
        groups[key].append(item)

    representatives: list[NewsItem] = []
    for key in ordered_keys:
        group = groups[key]
        representative = group[0]
        representative.metadata["duplicate_members"] = group
        representative.metadata["duplicate_count"] = len(group)
        representatives.append(representative)

    return representatives


def _dedupe_alias_texts(aliases: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        normalized = _normalize_match_text(str(alias))
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def _serialize_topic_alias_details(alias_candidates: list[TopicAlias]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for alias in alias_candidates:
        payload: dict[str, Any] = {
            "text": alias.text,
            "lang": alias.lang,
            "priority": alias.priority,
        }
        if alias.notes:
            payload["notes"] = list(alias.notes)
        details.append(payload)
    return details


def _fetch_candidates_with_alias_strategy(
    topic: str,
    limit: int | None,
    *,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> tuple[list[str], list[dict[str, Any]], list[NewsItem], list[NewsItem]]:
    alias_candidates = expand_topic_alias_candidates(topic)
    used_aliases = _dedupe_alias_texts([alias.text for alias in alias_candidates] or [topic.strip()])
    alias_details = _serialize_topic_alias_details(alias_candidates)
    candidate_news = fetch_candidate_news(
        topic,
        limit=limit,
        aliases=used_aliases,
        start_date=start_date,
        end_date=end_date,
    )
    filtered_news = _filter_candidates(candidate_news, used_aliases)

    warning_count = int(PIPELINE_CONFIG.get("event_discovery_candidate_warning_count", 8000))
    if len(filtered_news) > warning_count:
        warnings.warn(
            f"Topic '{topic}' produced {len(filtered_news)} filtered candidates; "
            "graph-link clustering builds an N x N similarity matrix.",
            RuntimeWarning,
            stacklevel=2,
        )

    return used_aliases, alias_details, candidate_news, filtered_news


def _resolve_output_dir() -> Path:
    output_dir = OUTPUTS_DIR / "clustered"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _export_outputs(result: EventDiscoveryResult) -> dict[str, str]:
    output_dir = _resolve_output_dir()
    topic_token = _sanitize_topic_token(result.topic)

    events_path = output_dir / f"{topic_token}_events.json"
    assignments_path = output_dir / f"{topic_token}_assignments.json"
    graph_path = output_dir / f"{topic_token}_graph.json"

    _write_json(
        events_path,
        {
            "topic": result.topic,
            "run_id": result.run_id,
            "topic_aliases": result.topic_aliases,
            "topic_alias_details": result.topic_alias_details,
            "candidate_count": result.candidate_count,
            "filtered_count": result.filtered_count,
            "events": [event.to_dict() for event in result.events],
        },
    )
    _write_json(
        assignments_path,
        {
            "topic": result.topic,
            "run_id": result.run_id,
            "topic_aliases": result.topic_aliases,
            "topic_alias_details": result.topic_alias_details,
            "candidate_count": result.candidate_count,
            "filtered_count": result.filtered_count,
            "assignments": result.assignments,
        },
    )
    _write_json(
        graph_path,
        {
            "topic": result.topic,
            "run_id": result.run_id,
            "topic_aliases": result.topic_aliases,
            "topic_alias_details": result.topic_alias_details,
            "candidate_count": result.candidate_count,
            "filtered_count": result.filtered_count,
            "graph_edges": result.graph_edges,
        },
    )

    return {
        "events": str(events_path),
        "assignments": str(assignments_path),
        "graph": str(graph_path),
    }


def _enrich_graph_edges(
    graph_edges: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    *,
    run_id: str,
) -> list[dict[str, Any]]:
    news_to_event = {assignment["news_id"]: assignment["event_id"] for assignment in assignments}

    enriched_edges = []
    for edge in graph_edges:
        edge_payload = dict(edge)
        edge_payload["run_id"] = run_id
        edge_payload["left_event_id"] = news_to_event.get(edge["left_news_id"])
        edge_payload["right_event_id"] = news_to_event.get(edge["right_news_id"])
        enriched_edges.append(edge_payload)
    return enriched_edges


def _attach_run_context(
    run_id: str,
    events,
    assignments: list[dict[str, Any]],
) -> tuple[list[Any], list[dict[str, Any]]]:
    raw_to_run_event_id: dict[str, str] = {}
    for event in events:
        raw_event_id = event.event_id
        scoped_event_id = f"{run_id}:{raw_event_id}"
        event.event_id = scoped_event_id
        raw_to_run_event_id[raw_event_id] = scoped_event_id

    enriched_assignments: list[dict[str, Any]] = []
    for assignment in assignments:
        enriched_assignment = dict(assignment)
        enriched_assignment["run_id"] = run_id
        enriched_assignment["event_id"] = raw_to_run_event_id[str(assignment["event_id"])]
        enriched_assignments.append(enriched_assignment)
    return events, enriched_assignments


def _to_db_datetime(value: Any) -> str | None:
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


def ensure_event_discovery_schema(cursor) -> None:
    """Create formal event discovery tables if they do not exist."""
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {EVENT_TABLE} (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            run_id VARCHAR(191) NOT NULL,
            event_id VARCHAR(191) NOT NULL,
            topic VARCHAR(255) NOT NULL,
            cluster_size INT NOT NULL,
            canonical_title TEXT NULL,
            representative_news_id VARCHAR(128) NULL,
            member_news_ids LONGTEXT NOT NULL,
            event_time_start DATETIME NULL,
            event_time_end DATETIME NULL,
            event_time_anchor DATETIME NULL,
            source_count INT NOT NULL DEFAULT 0,
            confidence DECIMAL(6,4) NOT NULL DEFAULT 0.0000,
            system_is_noise BOOLEAN NOT NULL DEFAULT FALSE,
            noise_reason VARCHAR(64) NULL,
            risk_flags LONGTEXT NULL,
            quality_metrics LONGTEXT NULL,
            generated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uniq_event_discovery_events_run_event (run_id, event_id),
            KEY idx_event_discovery_events_run (run_id),
            KEY idx_event_discovery_events_topic (topic),
            KEY idx_event_discovery_events_anchor (event_time_anchor)
        )
        """
    )
    cursor.execute(f"SHOW COLUMNS FROM {EVENT_TABLE}")
    event_columns = {row["Field"] for row in cursor.fetchall()}
    if "run_id" not in event_columns:
        cursor.execute(f"ALTER TABLE {EVENT_TABLE} ADD COLUMN run_id VARCHAR(191) NULL AFTER id")
    if "system_is_noise" not in event_columns:
        cursor.execute(f"ALTER TABLE {EVENT_TABLE} ADD COLUMN system_is_noise BOOLEAN NOT NULL DEFAULT FALSE AFTER confidence")
    if "noise_reason" not in event_columns:
        cursor.execute(f"ALTER TABLE {EVENT_TABLE} ADD COLUMN noise_reason VARCHAR(64) NULL AFTER system_is_noise")
    if "risk_flags" not in event_columns:
        cursor.execute(f"ALTER TABLE {EVENT_TABLE} ADD COLUMN risk_flags LONGTEXT NULL AFTER noise_reason")
    if "quality_metrics" not in event_columns:
        cursor.execute(f"ALTER TABLE {EVENT_TABLE} ADD COLUMN quality_metrics LONGTEXT NULL AFTER risk_flags")
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ASSIGNMENT_TABLE} (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            run_id VARCHAR(191) NOT NULL,
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
            generated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            KEY idx_event_discovery_assignments_run (run_id),
            KEY idx_event_discovery_assignments_topic (topic),
            KEY idx_event_discovery_assignments_event (event_id),
            KEY idx_event_discovery_assignments_news (news_id)
        )
        """
    )
    cursor.execute(f"SHOW COLUMNS FROM {ASSIGNMENT_TABLE}")
    assignment_columns = {row["Field"] for row in cursor.fetchall()}
    if "run_id" not in assignment_columns:
        cursor.execute(f"ALTER TABLE {ASSIGNMENT_TABLE} ADD COLUMN run_id VARCHAR(191) NULL AFTER id")
    if "url" not in assignment_columns:
        cursor.execute(f"ALTER TABLE {ASSIGNMENT_TABLE} ADD COLUMN url TEXT NULL AFTER source")
    if "system_is_noise" not in assignment_columns:
        cursor.execute(f"ALTER TABLE {ASSIGNMENT_TABLE} ADD COLUMN system_is_noise BOOLEAN NOT NULL DEFAULT FALSE AFTER canonical_title")
    if "noise_reason" not in assignment_columns:
        cursor.execute(f"ALTER TABLE {ASSIGNMENT_TABLE} ADD COLUMN noise_reason VARCHAR(64) NULL AFTER system_is_noise")
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {GRAPH_TABLE} (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            run_id VARCHAR(191) NOT NULL,
            topic VARCHAR(255) NOT NULL,
            left_news_id VARCHAR(128) NOT NULL,
            right_news_id VARCHAR(128) NOT NULL,
            left_event_id VARCHAR(191) NULL,
            right_event_id VARCHAR(191) NULL,
            similarity DECIMAL(8,6) NOT NULL,
            time_gap_days DECIMAL(10,3) NULL,
            edge_reason VARCHAR(32) NULL,
            generated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            KEY idx_event_discovery_graph_run (run_id),
            KEY idx_event_discovery_graph_topic (topic),
            KEY idx_event_discovery_graph_left_event (left_event_id),
            KEY idx_event_discovery_graph_right_event (right_event_id)
        )
        """
    )
    cursor.execute(f"SHOW COLUMNS FROM {GRAPH_TABLE}")
    graph_columns = {row["Field"] for row in cursor.fetchall()}
    if "run_id" not in graph_columns:
        cursor.execute(f"ALTER TABLE {GRAPH_TABLE} ADD COLUMN run_id VARCHAR(191) NULL AFTER id")


def persist_result_to_db(result: EventDiscoveryResult) -> None:
    """Persist one run-scoped event discovery output into MySQL."""
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            ensure_event_discovery_schema(cursor)

            if result.events:
                cursor.executemany(
                    f"""
                    INSERT INTO {EVENT_TABLE} (
                        run_id,
                        event_id,
                        topic,
                        cluster_size,
                        canonical_title,
                        representative_news_id,
                        member_news_ids,
                        event_time_start,
                        event_time_end,
                        event_time_anchor,
                        source_count,
                        confidence,
                        system_is_noise,
                        noise_reason,
                        risk_flags,
                        quality_metrics
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            result.run_id,
                            event.event_id,
                            event.topic,
                            event.cluster_size,
                            event.canonical_title,
                            None if event.representative_news_id is None else str(event.representative_news_id),
                            json.dumps(event.member_news_ids, ensure_ascii=False),
                            _to_db_datetime(event.event_time_start),
                            _to_db_datetime(event.event_time_end),
                            _to_db_datetime(event.event_time_anchor),
                            event.source_count,
                            event.confidence,
                            event.system_is_noise,
                            event.noise_reason,
                            json.dumps(event.risk_flags, ensure_ascii=False),
                            json.dumps(event.quality_metrics, ensure_ascii=False),
                        )
                        for event in result.events
                    ],
                )

            if result.assignments:
                cursor.executemany(
                    f"""
                    INSERT INTO {ASSIGNMENT_TABLE} (
                        run_id,
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
                        noise_reason
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            result.run_id,
                            result.topic,
                            str(assignment["event_id"]),
                            str(assignment["news_id"]),
                            assignment.get("title"),
                            assignment.get("source"),
                            assignment.get("url"),
                            _to_db_datetime(assignment.get("event_time_anchor")),
                            int(assignment.get("cluster_size") or 0),
                            assignment.get("canonical_title"),
                            bool(assignment.get("system_is_noise")),
                            assignment.get("noise_reason"),
                        )
                        for assignment in result.assignments
                    ],
                )

            if result.graph_edges:
                cursor.executemany(
                    f"""
                    INSERT INTO {GRAPH_TABLE} (
                        run_id,
                        topic,
                        left_news_id,
                        right_news_id,
                        left_event_id,
                        right_event_id,
                        similarity,
                        time_gap_days,
                        edge_reason
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            result.run_id,
                            result.topic,
                            str(edge["left_news_id"]),
                            str(edge["right_news_id"]),
                            edge.get("left_event_id"),
                            edge.get("right_event_id"),
                            float(edge["similarity"]),
                            edge.get("time_gap_days"),
                            edge.get("edge_reason"),
                        )
                        for edge in result.graph_edges
                    ],
                )

        connection.commit()
    finally:
        connection.close()


def run_event_discovery(
    topic: str,
    limit: int | None = None,
    *,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> EventDiscoveryResult:
    """Run the formal SBERT event discovery pipeline for a single topic."""
    if not isinstance(topic, str):
        raise TypeError("run_event_discovery expects a topic string and reads candidates from MySQL.")

    topic_aliases, topic_alias_details, candidate_news, filtered_news = _fetch_candidates_with_alias_strategy(
        topic,
        limit,
        start_date=start_date,
        end_date=end_date,
    )
    run_id = _generate_run_id(topic)

    result = EventDiscoveryResult(
        topic=topic,
        run_id=run_id,
        topic_aliases=topic_aliases,
        topic_alias_details=topic_alias_details,
        candidate_count=len(candidate_news),
        filtered_count=len(filtered_news),
    )

    if not filtered_news:
        persist_result_to_db(result)
        result.output_paths = _export_outputs(result)
        return result

    clustering_news = _prepare_news_for_clustering(filtered_news)
    embeddings = encode_titles([item.title for item in clustering_news])
    clusters, edges, similarity_matrix = cluster_embeddings(clustering_news, embeddings, topic=topic)
    events, assignments = build_event_nodes(topic, clusters, clustering_news, similarity_matrix)
    events, assignments = _attach_run_context(run_id, events, assignments)

    result.events = events
    result.assignments = assignments
    result.graph_edges = _enrich_graph_edges([edge.to_dict() for edge in edges], assignments, run_id=run_id)
    persist_result_to_db(result)
    result.output_paths = _export_outputs(result)
    return result
