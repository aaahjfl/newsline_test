"""Compatibility wrapper for the legacy language distribution script."""

from __future__ import annotations

from pathlib import Path

from data_pipeline._legacy import load_legacy_module


def run_language_distribution_scan():
    """Execute the legacy language analysis script through the formal package."""
    project_root = Path(__file__).resolve().parents[2]
    module_path = project_root / "code" / "data_pipeline" / "lnaguage" / "language_count.py"
    module = load_legacy_module("legacy_language_count", module_path)
    return module.analyze_language_distribution()
