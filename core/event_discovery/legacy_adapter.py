"""Compatibility helpers for locating legacy event discovery implementations."""

from __future__ import annotations

from pathlib import Path

from data_pipeline._legacy import load_legacy_module


def get_legacy_event_discovery_paths() -> dict[str, Path]:
    """Return current legacy files related to event discovery."""
    project_root = Path(__file__).resolve().parents[2]
    return {
        "archive_sbert_clustering": project_root / "archive_mvp" / "time_handling_test" / "time_sberting.py",
    }


def run_legacy_event_discovery_experiment():
    """Execute the archived SBERT clustering experiment script as-is."""
    module_path = get_legacy_event_discovery_paths()["archive_sbert_clustering"]
    return load_legacy_module("legacy_event_discovery_archive", module_path)
