"""Formal event discovery package."""

from .legacy_adapter import get_legacy_event_discovery_paths, run_legacy_event_discovery_experiment
from .pipeline import run_event_discovery

__all__ = [
    "get_legacy_event_discovery_paths",
    "run_event_discovery",
    "run_legacy_event_discovery_experiment",
]
