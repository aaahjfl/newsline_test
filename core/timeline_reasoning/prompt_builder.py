"""Prompt construction helpers for future timeline reasoning."""

from collections.abc import Sequence

from core.schemas import EventNode


def build_timeline_prompt(event_nodes: Sequence[EventNode]) -> str:
    """Build a compact prompt draft from event nodes."""
    titles = [node.canonical_title or node.event_id for node in event_nodes]
    return "Timeline ordering request:\n" + "\n".join(f"- {title}" for title in titles)
