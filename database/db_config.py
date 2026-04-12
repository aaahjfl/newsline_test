"""Backward-compatible database config module.

The formal source of truth now lives in `configs.db_config`.
"""

from configs.db_config import DB_CONFIG, get_db_config

__all__ = ["DB_CONFIG", "get_db_config"]
