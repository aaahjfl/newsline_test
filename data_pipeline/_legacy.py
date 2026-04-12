"""Helpers for lazily loading legacy modules without changing their logic."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def load_legacy_module(module_name: str, file_path: str | Path):
    """Load a legacy Python module from a concrete path exactly once."""
    path = Path(file_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load legacy module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
