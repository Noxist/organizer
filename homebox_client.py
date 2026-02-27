"""
homebox_client.py – HomeBox REST API client.

Uses login + auto-refresh for authentication.
All communication with HomeBox goes through this module.
"""

import os
import time
import logging
from typing import Optional

import httpx

log = logging.getLogger("organizer.homebox")

# ---------------------------------------------------------------------------
# Client state
# ---------------------------------------------------------------------------

_base_url: str = ""
_username: str = ""
_password: str = ""
_token: str = ""
_token_expires: float = 0.0  # unix timestamp
_attachment_token: str = ""

TOKEN_REFRESH_MARGIN = 300  # refresh 5 min before expiry


def init(base_url: str, username: str, password: str) -> None:
    global _base_url, _username, _password
    _base_url = base_url.rstrip("/")
    _username = username
    _password = password
    _login()


def _login() -> None:
    global _token, _token_expires, _attachment_token
    log.info("Logging into HomeBox at %s …", _base_url)
    r = httpx.post(
        f"{_base_url}/api/v1/users/login",
        json={"username": _username, "password": _password, "stayLoggedIn": True},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    _token = data["token"]
    _attachment_token = data.get("attachmentToken", "")
    # Parse expiry (ISO format)
    from datetime import datetime, timezone
    try:
        exp = datetime.fromisoformat(data["expiresAt"].replace("Z", "+00:00"))
        _token_expires = exp.timestamp()
    except Exception:
        # Fallback: 24h from now
        _token_expires = time.time() + 86400
    log.info("HomeBox login OK, token expires in %.0f hours", (_token_expires - time.time()) / 3600)


def _ensure_token() -> str:
    """Return valid Bearer token, refreshing if necessary."""
    global _token, _token_expires
    if time.time() > _token_expires - TOKEN_REFRESH_MARGIN:
        try:
            auth_value = _token if _token.startswith("Bearer ") else f"Bearer {_token}"
            r = httpx.get(
                f"{_base_url}/api/v1/users/refresh",
                headers={"Authorization": auth_value},
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                _token = data["token"]
                from datetime import datetime
                try:
                    exp = datetime.fromisoformat(data["expiresAt"].replace("Z", "+00:00"))
                    _token_expires = exp.timestamp()
                except Exception:
                    _token_expires = time.time() + 86400
                log.info("HomeBox token refreshed")
            else:
                log.warning("Token refresh failed (%d), re-logging in", r.status_code)
                _login()
        except Exception as e:
            log.warning("Token refresh error: %s, re-logging in", e)
            _login()
    return _token


def _headers() -> dict:
    token = _ensure_token()
    # HomeBox returns token as "Bearer XXX", use as-is if prefixed
    auth_value = token if token.startswith("Bearer ") else f"Bearer {token}"
    return {
        "Authorization": auth_value,
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

def get_items(page: int = 1, page_size: int = 100, search: str = "") -> dict:
    """GET /api/v1/items – Query items with pagination."""
    params = {"page": page, "pageSize": page_size}
    if search:
        params["q"] = search
    r = httpx.get(
        f"{_base_url}/api/v1/items",
        headers=_headers(),
        params=params,
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def get_item(item_id: str) -> dict:
    """GET /api/v1/items/{id} – Get single item."""
    r = httpx.get(
        f"{_base_url}/api/v1/items/{item_id}",
        headers=_headers(),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def create_item(name: str, description: str = "", location_id: str = "",
                tag_ids: list = None) -> dict:
    """POST /api/v1/items – Create new item in HomeBox.
    Note: HomeBox requires a locationId, otherwise returns 500.
    If none provided, we fetch the first available location.
    """
    if not location_id:
        location_id = _get_default_location_id()

    payload: dict = {"name": name, "locationId": location_id}
    if description:
        payload["description"] = description
    if tag_ids:
        payload["tagIds"] = tag_ids
    r = httpx.post(
        f"{_base_url}/api/v1/items",
        headers=_headers(),
        json=payload,
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


_default_location_id: str = ""


def _get_default_location_id() -> str:
    """Get first available location from HomeBox (cached)."""
    global _default_location_id
    if _default_location_id:
        return _default_location_id
    locations = get_locations()
    if locations:
        _default_location_id = locations[0]["id"]
        log.info("Default location: %s (%s)", locations[0]["name"], _default_location_id)
    return _default_location_id


def update_item_notes(item_id: str, notes: str) -> dict:
    """
    Write notes to an existing HomeBox item.
    Uses PUT /api/v1/items/{id} (PATCH doesn't support notes).
    Strategy: GET current → merge notes → PUT back.
    """
    # Get current item state
    current = get_item(item_id)

    # Build update payload (all required fields)
    payload = {
        "id": item_id,
        "name": current.get("name", ""),
        "description": current.get("description", ""),
        "notes": notes,
        "quantity": current.get("quantity", 1),
        "insured": current.get("insured", False),
        "archived": current.get("archived", False),
        "lifetimeWarranty": current.get("lifetimeWarranty", False),
        "manufacturer": current.get("manufacturer", ""),
        "modelNumber": current.get("modelNumber", ""),
        "serialNumber": current.get("serialNumber", ""),
        "purchaseFrom": current.get("purchaseFrom", ""),
    }

    # Preserve location
    loc = current.get("location")
    if loc and loc.get("id"):
        payload["locationId"] = loc["id"]

    # Preserve fields
    if current.get("fields"):
        payload["fields"] = current["fields"]

    r = httpx.put(
        f"{_base_url}/api/v1/items/{item_id}",
        headers=_headers(),
        json=payload,
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def patch_item(item_id: str, location_id: str = None, tag_ids: list = None,
               quantity: int = None) -> dict:
    """PATCH /api/v1/items/{id} – Partial update (location, tags, quantity)."""
    payload: dict = {"id": item_id}
    if location_id is not None:
        payload["locationId"] = location_id
    if tag_ids is not None:
        payload["tagIds"] = tag_ids
    if quantity is not None:
        payload["quantity"] = quantity
    r = httpx.patch(
        f"{_base_url}/api/v1/items/{item_id}",
        headers=_headers(),
        json=payload,
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

def get_locations() -> list:
    """GET /api/v1/locations – List all locations."""
    r = httpx.get(
        f"{_base_url}/api/v1/locations",
        headers=_headers(),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

def get_tags() -> list:
    """GET /api/v1/tags – List all tags."""
    r = httpx.get(
        f"{_base_url}/api/v1/tags",
        headers=_headers(),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def is_healthy() -> bool:
    """Quick health check against HomeBox."""
    try:
        r = httpx.get(f"{_base_url}/api/v1/status", timeout=5)
        return r.status_code == 200 and r.json().get("health", False)
    except Exception:
        return False
