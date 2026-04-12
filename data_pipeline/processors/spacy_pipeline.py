"""Current active spaCy processing flow under the formal architecture.

The large parsing body still lives in the existing MVP-era implementation so we
can keep behavior stable during migration. This module is the new architecture
entry point for the current spaCy capability.
"""

from __future__ import annotations

from pathlib import Path

from data_pipeline._legacy import load_legacy_module


def get_active_spacy_legacy_path() -> Path:
    """Return the current MVP spaCy parser file used by the active pipeline."""
    return Path(__file__).resolve().parents[2] / "code" / "data_pipeline" / "processors" / "spacy_parser.py"


def load_active_spacy_module():
    """Load the current active spaCy parser implementation."""
    return load_legacy_module("active_spacy_parser", get_active_spacy_legacy_path())


def available_model_names() -> dict[str, str]:
    """Expose the current model map from the active parser implementation."""
    module = load_active_spacy_module()
    return dict(module.MODEL_NAME_MAP)


def normalize_base_time(base_time):
    """Delegate base-time normalization to the current parser implementation."""
    module = load_active_spacy_module()
    return module.normalize_base_time(base_time)


def extract_event_time(title, base_time):
    """Delegate event-time extraction to the current parser implementation."""
    module = load_active_spacy_module()
    return module.extract_event_time(title, base_time)


def process_news_pipeline():
    """Run the current DB-backed spaCy processing pipeline unchanged."""
    module = load_active_spacy_module()
    return module.process_news_pipeline()
