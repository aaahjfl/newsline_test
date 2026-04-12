"""Database configuration shared by the formal project structure.

This is the primary database config source for new top-level modules.
Historical scripts may still embed their own DB constants until a later
migration phase.
"""

from copy import deepcopy
from typing import Any, Mapping


# Keep defaults aligned with the current legacy scripts to avoid breakage.
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "database": "news_db",
    "charset": "utf8mb4",
}


def get_db_config(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a copy of the current database config with optional overrides."""
    config = deepcopy(DB_CONFIG)
    if overrides:
        config.update(overrides)
    return config
