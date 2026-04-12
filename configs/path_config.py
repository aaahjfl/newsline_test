"""Project path helpers for the formal architecture layer.

These helpers are only for the new top-level package structure. Historical
scripts under `code/` and `archive_mvp/` keep their own path behavior until a
later migration phase.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NEWSDATA_DIR = PROJECT_ROOT / "newsdata"
ARCHIVE_TEST_DATA_DIR = PROJECT_ROOT / "archive_mvp" / "newsdata_for_test"
LEGACY_CODE_DIR = PROJECT_ROOT / "code"
LEGACY_ARCHIVE_DIR = PROJECT_ROOT / "archive_mvp"
LEGACY_EXTERNAL_CONFIG = PROJECT_ROOT / "config.props"


def resolve_project_path(path_str: str) -> Path:
    """Resolve a project-relative path string from the repository root."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
