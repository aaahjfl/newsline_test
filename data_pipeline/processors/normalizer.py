"""Normalization helpers for formal pipeline inputs and outputs."""

from pathlib import Path

from configs.path_config import resolve_project_path
from configs.pipeline_config import PIPELINE_CONFIG


def normalize_title(title: str | None) -> str:
    """Apply a conservative title normalization for indexing and display."""
    return " ".join((title or "").split())


def resolve_output_root() -> Path:
    """Resolve the configured output directory from the project root."""
    return resolve_project_path(PIPELINE_CONFIG["output_root"])
