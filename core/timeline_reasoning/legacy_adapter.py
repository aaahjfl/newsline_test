"""Compatibility helpers for locating legacy timeline reasoning implementations."""

from __future__ import annotations

from pathlib import Path

from data_pipeline._legacy import load_legacy_module


def get_legacy_timeline_reasoning_paths() -> dict[str, Path]:
    """Return current legacy files related to timeline reasoning."""
    project_root = Path(__file__).resolve().parents[2]
    return {
        "archive_timeline_reconstruction": project_root / "archive_mvp" / "time_handling_test" / "timeline_reconstruction.py",
        "script_trans_to_json": project_root / "code" / "script" / "trans_to_json.py",
    }


def load_legacy_timeline_reasoning_module(module_name: str = "script_trans_to_json"):
    """Load one legacy timeline reasoning module lazily."""
    try:
        module_path = get_legacy_timeline_reasoning_paths()[module_name]
    except KeyError as exc:  # pragma: no cover - defensive path.
        raise ValueError(f"Unknown legacy timeline reasoning module: {module_name}") from exc
    return load_legacy_module(f"legacy_timeline_reasoning_{module_name}", module_path)
