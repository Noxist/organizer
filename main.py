"""
main.py – FastAPI application for the Organizer service.

Deterministic geometry service alongside HomeBox.
Measures real objects, assigns stable physical addresses,
writes them back to HomeBox – without modifying HomeBox itself.
"""

import os
import io
import uuid
import asyncio
import base64
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

import state
import homebox_client as hb
import vision
import measure
import packer
from ui import dashboard_html, scan_html, containers_html, items_html

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("organizer")

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------

DATA_DIR = os.environ.get("ORGANIZER_DATA_DIR", "/data")
HOMEBOX_URL = os.environ.get("HOMEBOX_URL", "http://homebox:7745")
HOMEBOX_USER = os.environ.get("HOMEBOX_USER", "")
HOMEBOX_PASS = os.environ.get("HOMEBOX_PASS", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_ENABLED = os.environ.get("OPENAI_ENABLED", "false").lower() in ("true", "1", "yes")
GRID_SQUARE_MM = int(os.environ.get("GRID_SQUARE_MM", "20"))
SLOT_SIZE_MM = int(os.environ.get("SLOT_SIZE_MM", "50"))


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting Organizer service …")

    # 1. Init state
    state.init(DATA_DIR)
    log.info("State initialised at %s", DATA_DIR)

    # 2. Lock physical config (immutable after first run)
    state.init_config_lock(GRID_SQUARE_MM, SLOT_SIZE_MM)

    # 3. Init HomeBox client
    try:
        hb.init(HOMEBOX_URL, HOMEBOX_USER, HOMEBOX_PASS)
        log.info("HomeBox connected at %s", HOMEBOX_URL)
    except Exception as e:
        log.error("HomeBox connection failed: %s (service will retry)", e)

    # 4. Init Vision (optional)
    vision.init(api_key=OPENAI_API_KEY, enabled=OPENAI_ENABLED)

    # 5. Start pending-writes retry worker
    task = asyncio.create_task(_retry_worker())

    yield

    task.cancel()
    log.info("Organizer shutdown.")


app = FastAPI(title="Organizer", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Pending writes retry worker
# ---------------------------------------------------------------------------

async def _retry_worker():
    """Periodically retry failed HomeBox note writes."""
    while True:
        await asyncio.sleep(60)  # check every 60s
        try:
            pending = state.get_pending_writes()
            for pw in pending:
                try:
                    hb.update_item_notes(pw["homebox_id"], pw["note"])
                    state.clear_pending_write(pw["homebox_id"])
                    state.log_event("WRITE_RETRY_OK", f"homebox_id={pw['homebox_id']}")
                    log.info("Pending write succeeded: %s", pw["homebox_id"])
                except Exception as e:
                    state.increment_pending_retry(pw["homebox_id"])
                    if pw.get("retries", 0) < 50:
                        log.debug("Pending write retry failed: %s", e)
                    else:
                        log.warning("Pending write stuck after %d retries: %s", pw.get("retries", 0), pw["homebox_id"])
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error("Retry worker error: %s", e)


# ---------------------------------------------------------------------------
# UI Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def ui_dashboard():
    stats = packer.get_packing_stats()
    homebox_ok = hb.is_healthy()
    return dashboard_html(stats, homebox_ok)


@app.get("/scan", response_class=HTMLResponse)
async def ui_scan():
    return scan_html(openai_enabled=vision.is_enabled())


@app.get("/containers", response_class=HTMLResponse)
async def ui_containers():
    containers = state.get_containers()
    slots = state.get_all_slots()
    return containers_html(containers, slots)


@app.get("/items", response_class=HTMLResponse)
async def ui_items():
    items = state.get_items()
    slots = state.get_all_slots()
    containers = state.get_containers()
    return items_html(items, slots, containers)


# ---------------------------------------------------------------------------
# API: Scan
# ---------------------------------------------------------------------------

@app.post("/api/scan/measure")
async def api_scan_measure(image: UploadFile = File(...)):
    """Upload photo → OpenCV measurement + optional AI identification."""
    image_bytes = await image.read()
    scan_hash = state.compute_scan_hash(image_bytes)

    # Measure
    try:
        result = measure.measure_object(image_bytes, GRID_SQUARE_MM)
    except measure.MeasurementError as e:
        log.warning("Measurement failed: %s", str(e))
        return JSONResponse({"error": str(e)}, status_code=400)

    response = {
        "width_mm": result.width_mm,
        "depth_mm": result.depth_mm,
        "width_cm": result.width_cm,
        "depth_cm": result.depth_cm,
        "width_px": result.width_px,
        "depth_px": result.depth_px,
        "grid_detected": result.grid_detected,
        "calibration_ok": result.calibration_ok,
        "px_per_mm": round(result.px_per_mm, 2),
        "scan_hash": scan_hash,
        # Extended measurements
        "length_mm": result.length_mm,
        "length_cm": result.length_cm,
        "handle_width_mm": result.handle_width_mm,
        "shaft_length_mm": result.shaft_length_mm,
        "angle": result.angle,
    }

    # Debug image
    if result.debug_image is not None:
        import cv2
        _, buf = cv2.imencode(".jpg", result.debug_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        response["debug_image"] = base64.b64encode(buf).decode("utf-8")

    # Optional AI identification (non-blocking attempt)
    ai_result = vision.identify(image_bytes)
    if ai_result:
        response["ai_result"] = ai_result.to_dict()

    state.log_event("SCAN_MEASURE", f"hash={scan_hash[:16]} size={result.width_mm}x{result.depth_mm}mm")
    return response


# ---------------------------------------------------------------------------
# API: Items
# ---------------------------------------------------------------------------

class ItemCreate(BaseModel):
    name: str
    tags: list = []
    description: str = ""
    width_mm: int
    depth_mm: int
    height_mm: Optional[int] = None
    scan_hash: str = ""


@app.post("/api/items")
async def api_create_item(body: ItemCreate):
    """Confirm scanned item → create in HomeBox → assign slot → write address."""
    item_id = str(uuid.uuid4())[:8]

    # 1. Create item in HomeBox
    try:
        hb_item = hb.create_item(
            name=body.name,
            description=body.description,
        )
        homebox_id = hb_item.get("id", "")
    except Exception as e:
        return JSONResponse({"error": f"HomeBox item creation failed: {e}"}, status_code=502)

    # 2. Register in local state
    try:
        state.add_item(
            item_id=item_id,
            homebox_id=homebox_id,
            name=body.name,
            width_mm=body.width_mm,
            depth_mm=body.depth_mm,
            height_mm=body.height_mm,
            scan_hash=body.scan_hash or state.compute_scan_hash(body.name.encode()),
            tags=body.tags,
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=409)

    # 3. Auto-assign slot (First-Fit)
    slot_id = packer.find_slot_for_item(item_id)
    if slot_id:
        state.place_item(item_id, slot_id)
    else:
        state.log_event("ITEM_NO_FIT", f"{item_id} ({body.name}) — no slot available")

    # 4. Build address and write to HomeBox
    address = state.format_address(item_id)
    note = f"📦 {address}"
    if body.width_mm and body.depth_mm:
        note += f"\n📏 {body.width_mm}×{body.depth_mm}"
        if body.height_mm:
            note += f"×{body.height_mm}"
        note += "mm"

    try:
        hb.update_item_notes(homebox_id, note)
    except Exception as e:
        log.warning("HomeBox note write failed, queuing: %s", e)
        state.enqueue_pending_write(item_id, homebox_id, note)

    return {
        "item_id": item_id,
        "homebox_id": homebox_id,
        "name": body.name,
        "slot_id": slot_id or "UNPLACED",
        "address": address,
    }


@app.get("/api/items")
async def api_list_items():
    return state.get_items()


@app.post("/api/items/{item_id}/remove")
async def api_remove_item(item_id: str):
    """Remove item from its slot (stays in HomeBox)."""
    try:
        old_slot = state.remove_item_from_slot(item_id)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    return RedirectResponse("/items", status_code=303)


# ---------------------------------------------------------------------------
# API: Containers
# ---------------------------------------------------------------------------

@app.post("/api/containers")
async def api_create_container(
    id: str = Form(...),
    name: str = Form(...),
    location: str = Form(...),
    width_mm: int = Form(...),
    depth_mm: int = Form(...),
    height_mm: int = Form(...),
):
    """Create new container with fixed slot grid."""
    try:
        state.create_container(
            container_id=id,
            name=name,
            location=location,
            width_mm=width_mm,
            depth_mm=depth_mm,
            height_mm=height_mm,
            slot_size_mm=SLOT_SIZE_MM,
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=409)
    return RedirectResponse("/containers", status_code=303)


@app.get("/api/containers")
async def api_list_containers():
    return state.get_containers()


@app.get("/api/containers/{container_id}")
async def api_get_container(container_id: str):
    container = state.get_container(container_id)
    if not container:
        return JSONResponse({"error": "Container not found"}, status_code=404)
    slots = state.get_container_slots(container_id)
    return {"container": container, "slots": slots}


@app.post("/api/containers/{container_id}/delete")
async def api_delete_container(container_id: str, force: bool = False):
    """Delete container. Fails if occupied unless force=true."""
    try:
        orphans = state.delete_container(container_id, force=force)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=409)
    return RedirectResponse("/containers", status_code=303)


# ---------------------------------------------------------------------------
# API: Packing
# ---------------------------------------------------------------------------

@app.post("/api/pack")
async def api_pack():
    """Run First-Fit Decreasing on all unplaced items."""
    result = packer.pack_unplaced(dry_run=False)

    # Write addresses to HomeBox for newly placed items
    for item_id, slot_id in result.placed:
        item = state.get_item(item_id)
        if item and item.get("homebox_id"):
            address = state.format_address(item_id)
            note = f"📦 {address}"
            try:
                hb.update_item_notes(item["homebox_id"], note)
            except Exception as e:
                log.warning("HomeBox note write failed for %s: %s", item_id, e)
                state.enqueue_pending_write(item_id, item["homebox_id"], note)

    return RedirectResponse("/items", status_code=303)


@app.get("/api/pack/preview")
async def api_pack_preview():
    """Preview packing without committing."""
    result = packer.pack_unplaced(dry_run=True)
    return result.to_dict()


# ---------------------------------------------------------------------------
# API: Stats & Health
# ---------------------------------------------------------------------------

@app.get("/api/stats")
async def api_stats():
    return packer.get_packing_stats()


@app.get("/api/health")
async def api_health():
    return {
        "status": "ok",
        "homebox": hb.is_healthy(),
        "openai_enabled": vision.is_enabled(),
    }
