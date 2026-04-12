"""Shared database utilities."""

from typing import Any, Mapping

from configs.db_config import get_db_config

try:
    import pymysql
except ImportError:  # pragma: no cover - handled at runtime when DB access is needed.
    pymysql = None


def build_connection_kwargs(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build connection kwargs without forcing DB access during imports."""
    config = get_db_config(overrides)
    if pymysql is not None:
        config.setdefault("cursorclass", pymysql.cursors.DictCursor)
    return config


def get_db_connection(overrides: Mapping[str, Any] | None = None):
    """Return a PyMySQL connection using centralized config."""
    if pymysql is None:
        raise RuntimeError("PyMySQL is required before opening a database connection.")
    return pymysql.connect(**build_connection_kwargs(overrides))
