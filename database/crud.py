"""Minimal CRUD helpers used by the formal project skeleton."""

from typing import Any

from .db_utils import get_db_connection


def fetch_raw_news(limit: int | None = None, *, table: str = "raw_news_data") -> list[dict[str, Any]]:
    """Fetch raw news rows for downstream processing."""
    sql = f"SELECT * FROM {table}"
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT %s"
        params = (limit,)

    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return list(cursor.fetchall())
    finally:
        connection.close()
