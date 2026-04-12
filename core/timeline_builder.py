"""Utilities for materializing final timeline outputs."""

from collections.abc import Sequence

from core.schemas import TimelineNode


def build_timeline_output(timeline_nodes: Sequence[TimelineNode]) -> list[dict[str, object]]:
    """Serialize timeline nodes into a stable output structure."""
    return [
        {
            "event_id": node.event_id,
            "order_index": node.order_index,
            "reasoning_note": node.reasoning_note,
        }
        for node in sorted(timeline_nodes, key=lambda item: item.order_index)
    ]
