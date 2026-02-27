"""
state.py – Persistent JSON state management with file-locking.

Three separate state files + config lock + pending writes + event log.
All writes are atomic (write-to-tmp, then os.replace).
All mutations go through update_state() with a global lock.
"""

import os
import json
import time
import hashlib
import logging
from datetime import datetime, timezone
from typing import Callable, Optional, Any
from filelock import FileLock

log = logging.getLogger("organizer.state")

# ---------------------------------------------------------------------------
# Paths (set via init())
# ---------------------------------------------------------------------------
DATA_DIR: str = "/data"

CONTAINERS_FILE = ""
SLOTS_FILE = ""
ITEM_META_FILE = ""
CONFIG_LOCK_FILE = ""
PENDING_WRITES_FILE = ""
EVENTS_LOG_FILE = ""

CURRENT_SCHEMA_VERSION = 1

# Global reentrant lock for all state mutations
_global_lock: Optional[FileLock] = None


def _path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def init(data_dir: str = "/data") -> None:
    """Initialise paths and ensure data directory exists."""
    global DATA_DIR, CONTAINERS_FILE, SLOTS_FILE, ITEM_META_FILE
    global CONFIG_LOCK_FILE, PENDING_WRITES_FILE, EVENTS_LOG_FILE
    global _global_lock

    DATA_DIR = data_dir
    os.makedirs(DATA_DIR, exist_ok=True)

    CONTAINERS_FILE = _path("containers.json")
    SLOTS_FILE = _path("slots.json")
    ITEM_META_FILE = _path("item_meta.json")
    CONFIG_LOCK_FILE = _path("config_lock.json")
    PENDING_WRITES_FILE = _path("pending_writes.json")
    EVENTS_LOG_FILE = _path("events.log")

    _global_lock = FileLock(_path(".state.lock"))

    # Bootstrap empty files if they don't exist
    _ensure_file(CONTAINERS_FILE, {"schema_version": CURRENT_SCHEMA_VERSION, "containers": {}})
    _ensure_file(SLOTS_FILE, {"schema_version": CURRENT_SCHEMA_VERSION, "slots": {}})
    _ensure_file(ITEM_META_FILE, {"schema_version": CURRENT_SCHEMA_VERSION, "items": {}})
    _ensure_file(PENDING_WRITES_FILE, {"schema_version": CURRENT_SCHEMA_VERSION, "queue": []})


# ---------------------------------------------------------------------------
# Atomic I/O
# ---------------------------------------------------------------------------

def _write_json_atomic(path: str, data: dict) -> None:
    """Write JSON atomically: write to .tmp, then os.replace."""
    lock = FileLock(f"{path}.lock")
    with lock:
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)


def _read_json(path: str) -> dict:
    lock = FileLock(f"{path}.lock")
    with lock:
        with open(path, "r") as f:
            return json.load(f)


def _ensure_file(path: str, default: dict) -> None:
    if not os.path.exists(path):
        _write_json_atomic(path, default)


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

def _check_schema(data: dict, file_label: str) -> dict:
    v = data.get("schema_version", 0)
    if v < CURRENT_SCHEMA_VERSION:
        log.warning("Migrating %s from schema v%d → v%d", file_label, v, CURRENT_SCHEMA_VERSION)
        data["schema_version"] = CURRENT_SCHEMA_VERSION
        # Future migrations go here
    elif v > CURRENT_SCHEMA_VERSION:
        raise RuntimeError(
            f"{file_label}: schema_version {v} is newer than supported {CURRENT_SCHEMA_VERSION}. "
            "Please update the organizer service."
        )
    return data


# ---------------------------------------------------------------------------
# Config lock (immutable after first init)
# ---------------------------------------------------------------------------

def init_config_lock(grid_square_mm: int, slot_size_mm: int) -> dict:
    """
    Set physical constants on first run. Fatal error if file exists
    with different values.
    """
    if os.path.exists(CONFIG_LOCK_FILE):
        existing = _read_json(CONFIG_LOCK_FILE)
        if existing["grid_square_mm"] != grid_square_mm:
            raise RuntimeError(
                f"FATAL: grid_square_mm mismatch! "
                f"Locked={existing['grid_square_mm']}, Requested={grid_square_mm}. "
                "Physical calibration cannot change after initialisation."
            )
        if existing["slot_size_mm"] != slot_size_mm:
            raise RuntimeError(
                f"FATAL: slot_size_mm mismatch! "
                f"Locked={existing['slot_size_mm']}, Requested={slot_size_mm}. "
                "Physical calibration cannot change after initialisation."
            )
        log.info("Config lock verified: grid=%dmm, slot=%dmm", grid_square_mm, slot_size_mm)
        return existing

    cfg = {
        "grid_square_mm": grid_square_mm,
        "slot_size_mm": slot_size_mm,
        "locked_at": _now_iso(),
    }
    _write_json_atomic(CONFIG_LOCK_FILE, cfg)
    log.info("Config lock created: grid=%dmm, slot=%dmm", grid_square_mm, slot_size_mm)
    return cfg


def get_config_lock() -> dict:
    if not os.path.exists(CONFIG_LOCK_FILE):
        raise RuntimeError("FATAL: config_lock.json missing. System not initialised.")
    return _read_json(CONFIG_LOCK_FILE)


# ---------------------------------------------------------------------------
# Centralised state access  (read-modify-write as transaction)
# ---------------------------------------------------------------------------

class StateBundle:
    """All three state files loaded together."""
    def __init__(self, containers: dict, slots: dict, items: dict):
        self.containers = containers
        self.slots = slots
        self.items = items


def _load_all() -> StateBundle:
    c = _check_schema(_read_json(CONTAINERS_FILE), "containers.json")
    s = _check_schema(_read_json(SLOTS_FILE), "slots.json")
    i = _check_schema(_read_json(ITEM_META_FILE), "item_meta.json")
    return StateBundle(c, s, i)


def _save_all(state: StateBundle) -> None:
    _write_json_atomic(CONTAINERS_FILE, state.containers)
    _write_json_atomic(SLOTS_FILE, state.slots)
    _write_json_atomic(ITEM_META_FILE, state.items)


def update_state(mutator: Callable[[StateBundle], Any]) -> Any:
    """
    Central transaction function. All state mutations MUST go through here.
    Acquires global lock → loads all state → calls mutator → saves.
    Returns whatever the mutator returns.
    """
    with _global_lock:
        state = _load_all()
        result = mutator(state)
        _save_all(state)
        return result


def read_state() -> StateBundle:
    """Read-only access (still acquires lock for consistency)."""
    with _global_lock:
        return _load_all()


# ---------------------------------------------------------------------------
# Event log (append-only)
# ---------------------------------------------------------------------------

def log_event(event_type: str, detail: str) -> None:
    ts = _now_iso()
    line = f"[{ts}] {event_type} {detail}\n"
    lock = FileLock(f"{EVENTS_LOG_FILE}.lock")
    with lock:
        with open(EVENTS_LOG_FILE, "a") as f:
            f.write(line)
    log.info("EVENT: %s %s", event_type, detail)


# ---------------------------------------------------------------------------
# Pending writes queue (HomeBox retry)
# ---------------------------------------------------------------------------

def enqueue_pending_write(item_id: str, homebox_id: str, note: str) -> None:
    lock = FileLock(f"{PENDING_WRITES_FILE}.lock")
    with lock:
        data = _read_json(PENDING_WRITES_FILE) if os.path.exists(PENDING_WRITES_FILE) else {"schema_version": 1, "queue": []}
        data["queue"].append({
            "item_id": item_id,
            "homebox_id": homebox_id,
            "note": note,
            "created_at": _now_iso(),
            "retries": 0,
        })
        _write_json_atomic(PENDING_WRITES_FILE, data)


def get_pending_writes() -> list:
    if not os.path.exists(PENDING_WRITES_FILE):
        return []
    data = _read_json(PENDING_WRITES_FILE)
    return data.get("queue", [])


def clear_pending_write(homebox_id: str) -> None:
    lock = FileLock(f"{PENDING_WRITES_FILE}.lock")
    with lock:
        data = _read_json(PENDING_WRITES_FILE)
        data["queue"] = [w for w in data["queue"] if w["homebox_id"] != homebox_id]
        _write_json_atomic(PENDING_WRITES_FILE, data)


def increment_pending_retry(homebox_id: str) -> None:
    lock = FileLock(f"{PENDING_WRITES_FILE}.lock")
    with lock:
        data = _read_json(PENDING_WRITES_FILE)
        for w in data["queue"]:
            if w["homebox_id"] == homebox_id:
                w["retries"] = w.get("retries", 0) + 1
                break
        _write_json_atomic(PENDING_WRITES_FILE, data)


# ---------------------------------------------------------------------------
# Container operations
# ---------------------------------------------------------------------------

def create_container(
    container_id: str,
    name: str,
    location: str,
    width_mm: int,
    depth_mm: int,
    height_mm: int,
    slot_size_mm: int,
) -> dict:
    """
    Create a container and generate its immutable slot grid.
    Slots have fixed coordinates in mm.
    """
    def _mutate(s: StateBundle):
        if container_id in s.containers["containers"]:
            raise ValueError(f"Container '{container_id}' already exists.")

        # Calculate slot grid
        cols = max(1, width_mm // slot_size_mm)
        rows = max(1, depth_mm // slot_size_mm)

        container = {
            "name": name,
            "location": location,
            "width_mm": width_mm,
            "depth_mm": depth_mm,
            "height_mm": height_mm,
            "slot_size_mm": slot_size_mm,
            "cols": cols,
            "rows": rows,
            "created_at": _now_iso(),
            "locked": False,
        }
        s.containers["containers"][container_id] = container

        # Generate fixed slot grid
        slot_idx = 1
        for row in range(rows):
            for col in range(cols):
                slot_key = f"{container_id}/S{slot_idx:02d}"
                s.slots["slots"][slot_key] = {
                    "container_id": container_id,
                    "col": col,
                    "row": row,
                    "x_mm": col * slot_size_mm,
                    "y_mm": row * slot_size_mm,
                    "occupied": False,
                    "item_id": None,
                    "footprint_mm2": None,
                }
                slot_idx += 1

        log_event("CONTAINER_CREATED", f"{container_id} ({name}) {cols}x{rows} slots @ {location}")
        return container

    return update_state(_mutate)


def get_containers() -> dict:
    s = read_state()
    return s.containers["containers"]


def get_container(container_id: str) -> Optional[dict]:
    s = read_state()
    return s.containers["containers"].get(container_id)


def get_container_slots(container_id: str) -> dict:
    """Return all slots belonging to a container."""
    s = read_state()
    return {
        k: v for k, v in s.slots["slots"].items()
        if v["container_id"] == container_id
    }


def delete_container(container_id: str, force: bool = False) -> list:
    """
    Delete a container. Returns list of orphaned item IDs.
    Refuses if container has items unless force=True.
    """
    def _mutate(s: StateBundle):
        if container_id not in s.containers["containers"]:
            raise ValueError(f"Container '{container_id}' does not exist.")

        # Find occupied slots
        occupied = [
            (k, v) for k, v in s.slots["slots"].items()
            if v["container_id"] == container_id and v["occupied"]
        ]
        orphaned_items = [v["item_id"] for _, v in occupied if v["item_id"]]

        if occupied and not force:
            raise ValueError(
                f"Container '{container_id}' has {len(occupied)} occupied slot(s). "
                f"Use force=true to delete. Orphaned items: {orphaned_items}"
            )

        # Move orphaned items to UNPLACED
        for item_id in orphaned_items:
            if item_id and item_id in s.items["items"]:
                s.items["items"][item_id]["slot_id"] = "UNPLACED"

        # Remove slots
        s.slots["slots"] = {
            k: v for k, v in s.slots["slots"].items()
            if v["container_id"] != container_id
        }

        # Remove container
        del s.containers["containers"][container_id]

        log_event("CONTAINER_DELETED", f"{container_id} force={force} orphaned={orphaned_items}")
        return orphaned_items

    return update_state(_mutate)


# ---------------------------------------------------------------------------
# Item operations
# ---------------------------------------------------------------------------

def add_item(
    item_id: str,
    homebox_id: str,
    name: str,
    width_mm: int,
    depth_mm: int,
    height_mm: Optional[int],
    scan_hash: str,
    tags: Optional[list] = None,
) -> dict:
    """Add item to meta store. Initially UNPLACED."""
    def _mutate(s: StateBundle):
        # Idempotency: check scan_hash
        for existing in s.items["items"].values():
            if existing.get("scan_hash") == scan_hash:
                raise ValueError(
                    f"Duplicate scan detected (hash={scan_hash[:16]}...). "
                    f"Item '{existing['name']}' already registered."
                )

        item = {
            "homebox_id": homebox_id,
            "name": name,
            "width_mm": width_mm,
            "depth_mm": depth_mm,
            "height_mm": height_mm,
            "footprint_mm2": width_mm * depth_mm,
            "scan_hash": scan_hash,
            "slot_id": "UNPLACED",
            "tags": tags or [],
            "created_at": _now_iso(),
        }
        s.items["items"][item_id] = item
        log_event("ITEM_ADDED", f"{item_id} ({name}) {width_mm}x{depth_mm}mm → UNPLACED")
        return item

    return update_state(_mutate)


def place_item(item_id: str, slot_id: str) -> None:
    """Assign an item to a specific slot."""
    def _mutate(s: StateBundle):
        if item_id not in s.items["items"]:
            raise ValueError(f"Item '{item_id}' not found.")
        if slot_id not in s.slots["slots"]:
            raise ValueError(f"Slot '{slot_id}' not found.")
        if s.slots["slots"][slot_id]["occupied"]:
            raise ValueError(f"Slot '{slot_id}' is already occupied.")

        item = s.items["items"][item_id]

        # Unoccupy old slot if item was placed somewhere
        old_slot = item.get("slot_id")
        if old_slot and old_slot != "UNPLACED" and old_slot in s.slots["slots"]:
            s.slots["slots"][old_slot]["occupied"] = False
            s.slots["slots"][old_slot]["item_id"] = None
            s.slots["slots"][old_slot]["footprint_mm2"] = None

        # Place in new slot
        s.slots["slots"][slot_id]["occupied"] = True
        s.slots["slots"][slot_id]["item_id"] = item_id
        s.slots["slots"][slot_id]["footprint_mm2"] = item["footprint_mm2"]

        item["slot_id"] = slot_id

        # Lock the container (dimensions immutable now)
        container_id = s.slots["slots"][slot_id]["container_id"]
        s.containers["containers"][container_id]["locked"] = True

        log_event("ITEM_PLACED", f"{item_id} → {slot_id}")

    update_state(_mutate)


def remove_item_from_slot(item_id: str) -> Optional[str]:
    """Remove item from its current slot. Returns old slot_id."""
    def _mutate(s: StateBundle):
        if item_id not in s.items["items"]:
            raise ValueError(f"Item '{item_id}' not found.")

        item = s.items["items"][item_id]
        old_slot = item.get("slot_id")
        if old_slot and old_slot != "UNPLACED" and old_slot in s.slots["slots"]:
            s.slots["slots"][old_slot]["occupied"] = False
            s.slots["slots"][old_slot]["item_id"] = None
            s.slots["slots"][old_slot]["footprint_mm2"] = None

        item["slot_id"] = "UNPLACED"
        log_event("ITEM_REMOVED", f"{item_id} from {old_slot} → UNPLACED")
        return old_slot

    return update_state(_mutate)


def get_items() -> dict:
    s = read_state()
    return s.items["items"]


def get_item(item_id: str) -> Optional[dict]:
    s = read_state()
    return s.items["items"].get(item_id)


def get_unplaced_items() -> dict:
    s = read_state()
    return {
        k: v for k, v in s.items["items"].items()
        if v.get("slot_id") == "UNPLACED"
    }


def get_all_slots() -> dict:
    s = read_state()
    return s.slots["slots"]


# ---------------------------------------------------------------------------
# Address formatting
# ---------------------------------------------------------------------------

def format_address(item_id: str) -> str:
    """Build LOCATION / CONTAINER_ID / SLOT_ID address string."""
    s = read_state()
    item = s.items["items"].get(item_id)
    if not item:
        return "UNKNOWN"
    slot_id = item.get("slot_id", "UNPLACED")
    if slot_id == "UNPLACED":
        return "UNPLACED"

    slot = s.slots["slots"].get(slot_id)
    if not slot:
        return "UNPLACED"

    container_id = slot["container_id"]
    container = s.containers["containers"].get(container_id, {})
    location = container.get("location", "?")

    # Extract just the slot number part (e.g. "S12" from "R-01/S12")
    slot_label = slot_id.split("/")[-1] if "/" in slot_id else slot_id

    return f"{location} / {container_id} / {slot_label}"


def compute_scan_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
