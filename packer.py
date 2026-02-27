"""
packer.py – First-Fit Decreasing bin packing (2D footprint only).

Sorts items by footprint area (desc), then by ID for determinism.
Height is NOT used for packing (Schubladen-Logik).

Items that don't fit go to UNPLACED.
No automatic repacking or container creation.
"""

import logging
from typing import Optional

from state import (
    read_state,
    update_state,
    place_item,
    log_event,
    StateBundle,
)

log = logging.getLogger("organizer.packer")


class PackResult:
    def __init__(self):
        self.placed: list = []       # [(item_id, slot_id)]
        self.unplaced: list = []     # [item_id]

    def to_dict(self) -> dict:
        return {
            "placed": [{"item_id": i, "slot_id": s} for i, s in self.placed],
            "unplaced": self.unplaced,
            "placed_count": len(self.placed),
            "unplaced_count": len(self.unplaced),
        }


def pack_unplaced(dry_run: bool = False) -> PackResult:
    """
    Run First-Fit Decreasing on all UNPLACED items.

    Strategy:
    - Sort items by footprint (width*depth) DESC, then by item_id for determinism
    - For each item, find the first available slot that fits (by footprint)
    - Slots are checked in order: container_id ASC, then slot index ASC

    Args:
        dry_run: If True, compute packing but don't commit to state.

    Returns:
        PackResult with placed and unplaced items.
    """
    state = read_state()
    result = PackResult()

    # Collect unplaced items
    unplaced_items = [
        (item_id, item)
        for item_id, item in state.items["items"].items()
        if item.get("slot_id") == "UNPLACED"
    ]

    if not unplaced_items:
        log.info("No unplaced items to pack.")
        return result

    # Sort: largest footprint first, then by ID for determinism
    unplaced_items.sort(
        key=lambda x: (x[1].get("footprint_mm2", 0), x[0]),
        reverse=True,
    )

    # Collect available (empty) slots, sorted by container then slot index
    available_slots = [
        (slot_id, slot)
        for slot_id, slot in sorted(state.slots["slots"].items())
        if not slot["occupied"]
    ]

    # Get container info for slot size lookup
    containers = state.containers["containers"]

    # Track which slots we've "used" during this packing run
    used_slots = set()

    for item_id, item in unplaced_items:
        item_w = item.get("width_mm", 0)
        item_d = item.get("depth_mm", 0)

        placed = False
        for slot_id, slot in available_slots:
            if slot_id in used_slots:
                continue

            container_id = slot["container_id"]
            container = containers.get(container_id, {})
            slot_size = container.get("slot_size_mm", 50)

            # Check if item fits in a single slot
            # An item needs ceil(width/slot_size) x ceil(depth/slot_size) slots
            # For simplicity in MVP: item fits if footprint <= slot_size^2
            # TODO: Multi-slot spanning in future version
            if item_w <= slot_size and item_d <= slot_size:
                result.placed.append((item_id, slot_id))
                used_slots.add(slot_id)
                placed = True
                break
            # Try rotated
            elif item_d <= slot_size and item_w <= slot_size:
                result.placed.append((item_id, slot_id))
                used_slots.add(slot_id)
                placed = True
                break

        if not placed:
            result.unplaced.append(item_id)

    # Commit if not dry run
    if not dry_run:
        for item_id, slot_id in result.placed:
            place_item(item_id, slot_id)

    log.info(
        "Packing complete: %d placed, %d unplaced",
        len(result.placed), len(result.unplaced),
    )
    log_event(
        "PACK_RUN",
        f"placed={len(result.placed)} unplaced={len(result.unplaced)} dry_run={dry_run}",
    )

    return result


def find_slot_for_item(item_id: str) -> Optional[str]:
    """
    Find the best single slot for one specific item.
    Returns slot_id or None.
    """
    state = read_state()

    item = state.items["items"].get(item_id)
    if not item:
        return None

    item_w = item.get("width_mm", 0)
    item_d = item.get("depth_mm", 0)
    containers = state.containers["containers"]

    for slot_id, slot in sorted(state.slots["slots"].items()):
        if slot["occupied"]:
            continue

        container_id = slot["container_id"]
        container = containers.get(container_id, {})
        slot_size = container.get("slot_size_mm", 50)

        # Check fit (including rotation)
        if (item_w <= slot_size and item_d <= slot_size) or \
           (item_d <= slot_size and item_w <= slot_size):
            return slot_id

    return None


def get_packing_stats() -> dict:
    """Return overall packing statistics."""
    state = read_state()

    total_items = len(state.items["items"])
    placed_items = sum(
        1 for i in state.items["items"].values()
        if i.get("slot_id") not in ("UNPLACED", None)
    )
    unplaced_items = total_items - placed_items

    total_slots = len(state.slots["slots"])
    occupied_slots = sum(1 for s in state.slots["slots"].values() if s["occupied"])
    free_slots = total_slots - occupied_slots

    total_containers = len(state.containers["containers"])

    return {
        "total_items": total_items,
        "placed_items": placed_items,
        "unplaced_items": unplaced_items,
        "total_slots": total_slots,
        "occupied_slots": occupied_slots,
        "free_slots": free_slots,
        "total_containers": total_containers,
        "utilization_pct": round(occupied_slots / total_slots * 100, 1) if total_slots > 0 else 0,
    }
