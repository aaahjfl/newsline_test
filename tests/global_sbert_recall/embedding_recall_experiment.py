"""Archived experiment: embedding-based global candidate recall.

This module is intentionally kept under tests/ as an experiment record. It is
not part of the production event-discovery path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from configs.path_config import OUTPUTS_DIR
from database.db_utils import get_db_connection

from core.event_discovery.encoder import get_embedding_model_name, load_embedding_model
from core.schemas import NewsItem


DEFAULT_TOP_K = 500
DEFAULT_MIN_SCORE = 0.30
DEFAULT_INDEX_DIR = OUTPUTS_DIR / "embeddings"
DEFAULT_INDEX_PATH = DEFAULT_INDEX_DIR / "title_embedding_index.npz"
DEFAULT_META_PATH = DEFAULT_INDEX_DIR / "title_embedding_index_meta.json"
RECALL_EMBEDDING_PROMPT = (
    "请为跨语言新闻主题检索生成语义向量。"
    "向量应能召回不同语言中与同一主题相关的新闻标题，"
    "包括同一实体、机构、人物、地点、组织或事件主题的不同表达。"
)


@dataclass(slots=True)
class TitleEmbeddingIndex:
    news_ids: np.ndarray
    embeddings: np.ndarray
    titles: np.ndarray
    meta: dict[str, Any]


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return embeddings / safe_norms


def encode_recall_texts(texts: list[str], *, batch_size: int = 32) -> np.ndarray:
    """Encode topics or titles for the archived global recall experiment."""
    cleaned_texts = [str(text).strip() for text in texts if str(text).strip()]
    if not cleaned_texts:
        return np.empty((0, 0), dtype=np.float32)

    model = load_embedding_model()
    embeddings = model.encode(
        cleaned_texts,
        prompt=RECALL_EMBEDDING_PROMPT,
        batch_size=batch_size,
        show_progress_bar=len(cleaned_texts) >= 32,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return _normalize_embeddings(array)


def _serialize_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
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


def fetch_title_rows_for_index(limit: int | None = None) -> list[dict[str, Any]]:
    """Fetch title rows used to build the persistent recall index."""
    sql = """
        SELECT id, title
        FROM parser_newsdata
        WHERE title IS NOT NULL
          AND TRIM(title) <> ''
        ORDER BY id ASC
    """
    params: list[Any] = []
    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)

    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            return list(cursor.fetchall())
    finally:
        connection.close()


def build_title_embedding_index(
    *,
    limit: int | None = None,
    batch_size: int = 64,
    index_path: Path = DEFAULT_INDEX_PATH,
    meta_path: Path = DEFAULT_META_PATH,
) -> dict[str, Any]:
    """Build and persist title embeddings for broad topic recall."""
    rows = fetch_title_rows_for_index(limit=limit)
    titles = [str(row.get("title", "")).strip() for row in rows]
    news_ids = [str(row["id"]) for row in rows]
    embeddings = encode_recall_texts(titles, batch_size=batch_size)

    if len(news_ids) != embeddings.shape[0]:
        raise RuntimeError("title count does not match embedding count.")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        index_path,
        news_ids=np.asarray(news_ids, dtype=str),
        titles=np.asarray(titles, dtype=str),
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )

    meta = {
        "built_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "model": get_embedding_model_name(),
        "prompt": RECALL_EMBEDDING_PROMPT,
        "count": len(news_ids),
        "dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[0] else 0,
        "limit": limit,
        "index_path": str(index_path),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_title_embedding_index(
    *,
    index_path: Path = DEFAULT_INDEX_PATH,
    meta_path: Path = DEFAULT_META_PATH,
) -> TitleEmbeddingIndex:
    """Load the local title embedding recall index."""
    if not index_path.exists():
        raise FileNotFoundError(
            f"Title embedding index not found: {index_path}. "
            "Run tests/global_sbert_recall/build_title_embedding_index_experiment.py first."
        )

    payload = np.load(index_path, allow_pickle=False)
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return TitleEmbeddingIndex(
        news_ids=payload["news_ids"],
        titles=payload["titles"],
        embeddings=np.asarray(payload["embeddings"], dtype=np.float32),
        meta=meta,
    )


def recall_news_ids_by_embedding(
    topic: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    index_path: Path = DEFAULT_INDEX_PATH,
    meta_path: Path = DEFAULT_META_PATH,
) -> list[dict[str, Any]]:
    """Return news ids ranked by topic/title embedding similarity."""
    normalized_topic = str(topic).strip()
    if not normalized_topic:
        raise ValueError("topic must be a non-empty string.")

    index = load_title_embedding_index(index_path=index_path, meta_path=meta_path)
    if index.embeddings.size == 0:
        return []

    query_embedding = encode_recall_texts([normalized_topic], batch_size=1)
    if query_embedding.size == 0:
        return []

    scores = np.clip(index.embeddings @ query_embedding[0], -1.0, 1.0)
    candidate_count = int(scores.shape[0])
    capped_top_k = min(max(int(top_k), 1), candidate_count)
    if capped_top_k == candidate_count:
        top_indices = np.argsort(scores)[::-1]
    else:
        top_indices = np.argpartition(scores, -capped_top_k)[-capped_top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    results: list[dict[str, Any]] = []
    for index_position in top_indices:
        score = float(scores[index_position])
        if score < min_score:
            continue
        results.append(
            {
                "news_id": str(index.news_ids[index_position]),
                "title": str(index.titles[index_position]),
                "score": score,
                "rank": len(results) + 1,
            }
        )
    return results


def fetch_news_by_ids(news_ids: list[str]) -> list[NewsItem]:
    """Fetch NewsItem rows by id while preserving the requested order."""
    ordered_ids = [str(news_id) for news_id in news_ids if str(news_id).strip()]
    if not ordered_ids:
        return []

    placeholders = ", ".join(["%s" for _ in ordered_ids])
    sql = f"""
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
        WHERE id IN ({placeholders})
    """

    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, tuple(ordered_ids))
            rows = list(cursor.fetchall())
    finally:
        connection.close()

    row_by_id = {str(row["id"]): row for row in rows}
    news_items: list[NewsItem] = []
    for news_id in ordered_ids:
        row = row_by_id.get(news_id)
        if row is None:
            continue
        news_items.append(
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
        )
    return news_items


def fetch_embedding_recall_candidates(
    topic: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    index_path: Path = DEFAULT_INDEX_PATH,
    meta_path: Path = DEFAULT_META_PATH,
) -> list[NewsItem]:
    """Recall candidate news using the persistent title embedding index."""
    recalled = recall_news_ids_by_embedding(
        topic,
        top_k=top_k,
        min_score=min_score,
        index_path=index_path,
        meta_path=meta_path,
    )
    news_items = fetch_news_by_ids([item["news_id"] for item in recalled])
    recall_by_id = {str(item["news_id"]): item for item in recalled}
    for item in news_items:
        recall_info = recall_by_id.get(str(item.news_id))
        if recall_info:
            item.metadata["embedding_recall_score"] = recall_info["score"]
            item.metadata["embedding_recall_rank"] = recall_info["rank"]
    return news_items
