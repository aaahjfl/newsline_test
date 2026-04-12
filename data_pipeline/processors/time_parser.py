"""Formal time parsing entry points backed by legacy implementations for now."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from core.schemas import NewsItem, ParsedNews
from data_pipeline._legacy import load_legacy_module
from data_pipeline.processors.spacy_pipeline import process_news_pipeline as run_current_spacy_pipeline
from data_pipeline.processors.time_standardizer import run_time_standardization as run_current_time_standardization


LEGACY_PROCESSOR_SPECS = {
    "spacy_v1": {
        "path": Path("code/data_pipeline/processors/spacy_parser_v1.py"),
        "function": "process_news_pipeline",
    },
    "heideltime": {
        "path": Path("code/data_pipeline/processors/heideltime_parser.py"),
        "function": "extract_and_calculate_event_time",
    },
}


def get_legacy_time_parser_paths() -> dict[str, Path]:
    """Expose current legacy processor locations for staged migration."""
    project_root = Path(__file__).resolve().parents[2]
    legacy_paths = {
        name: project_root / spec["path"]
        for name, spec in LEGACY_PROCESSOR_SPECS.items()
    }
    legacy_paths["spacy_v2"] = project_root / "code" / "data_pipeline" / "processors" / "spacy_parser.py"
    legacy_paths["time_standardizer"] = project_root / "code" / "data_pipeline" / "processors" / "trans_standard.py"
    return legacy_paths


def list_legacy_time_processors() -> dict[str, str]:
    """Return a lightweight processor registry for docs and service routing."""
    processors = {
        name: spec["function"]
        for name, spec in LEGACY_PROCESSOR_SPECS.items()
    }
    processors["spacy_v2"] = "process_news_pipeline"
    processors["time_standardizer"] = "run_time_standardization"
    return processors


def _load_legacy_processor(processor_name: str):
    try:
        spec = LEGACY_PROCESSOR_SPECS[processor_name]
    except KeyError as exc:  # pragma: no cover - defensive path.
        raise ValueError(f"Unknown legacy processor: {processor_name}") from exc

    project_root = Path(__file__).resolve().parents[2]
    module_path = project_root / spec["path"]
    module = load_legacy_module(f"legacy_{processor_name}", module_path)
    return getattr(module, spec["function"])


def run_legacy_processor(processor_name: str):
    """Execute one legacy processor by name without changing its logic."""
    legacy_function = _load_legacy_processor(processor_name)
    return legacy_function()


def run_time_standardization():
    """Current architecture entry point for DCT normalization."""
    return run_current_time_standardization()


def run_spacy_parser_v2():
    """Current architecture entry point for the active spaCy parser."""
    return run_current_spacy_pipeline()


def run_spacy_parser_v1():
    """Compatibility entry point for the older comparison parser."""
    return run_legacy_processor("spacy_v1")


def run_heideltime_parser():
    """Compatibility entry point for the HeidelTime-based parser."""
    return run_legacy_processor("heideltime")


def run_time_parser(news_items: Iterable[NewsItem] | None = None, *, parser: str = "spacy_v2") -> list[ParsedNews] | None:
    """Formal parsing entry point.

    Current stage:
    - if `news_items` is omitted, route to the original DB-driven legacy job;
    - if `news_items` are provided, keep the interface reserved for the future
      in-memory formal pipeline to avoid changing algorithm behavior prematurely.
    """
    if news_items is not None:
        raise NotImplementedError(
            "TODO: migrate parser logic to support in-memory formal pipeline inputs safely."
        )
    return run_legacy_processor(parser)
