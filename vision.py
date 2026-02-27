"""
vision.py – Optional GPT-4o Vision integration.

Behind OPENAI_ENABLED flag. Non-blocking, async.
Returns name suggestion + tags from a photo.
System works 100% without this module.
"""

import os
import base64
import logging
from typing import Optional

log = logging.getLogger("organizer.vision")

_enabled: bool = False
_api_key: str = ""


def init(api_key: str = "", enabled: bool = False) -> None:
    global _enabled, _api_key
    _api_key = api_key
    _enabled = enabled and bool(api_key)
    if _enabled:
        log.info("Vision (GPT-4o) enabled")
    else:
        log.info("Vision (GPT-4o) disabled")


def is_enabled() -> bool:
    return _enabled


class VisionResult:
    def __init__(self, name: str, tags: list, description: str, confidence: str = "medium"):
        self.name = name
        self.tags = tags
        self.description = description
        self.confidence = confidence

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "tags": self.tags,
            "description": self.description,
            "confidence": self.confidence,
        }


def identify(image_bytes: bytes) -> Optional[VisionResult]:
    """
    Send image to GPT-4o Vision for identification.
    Returns VisionResult with name, tags, description.
    Returns None if disabled or on error (never blocks workflow).
    """
    if not _enabled:
        return None

    try:
        import httpx
        import json

        b64 = base64.b64encode(image_bytes).decode("utf-8")

        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an inventory identification assistant. "
                            "Identify the object in the photo and return ONLY valid JSON with these fields:\n"
                            '{"name": "short descriptive name", '
                            '"tags": ["tag1", "tag2", "tag3"], '
                            '"description": "one sentence description", '
                            '"confidence": "high/medium/low"}\n'
                            "Keep the name concise (2-4 words). Tags should be useful for inventory categorization. "
                            "Respond in the same language as common for the object. "
                            "Return ONLY the JSON, no markdown, no explanation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                    "detail": "low",
                                },
                            },
                            {
                                "type": "text",
                                "text": "Identify this object for my home inventory.",
                            },
                        ],
                    },
                ],
                "max_tokens": 200,
                "temperature": 0.2,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"].strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(content)

        return VisionResult(
            name=result.get("name", "Unknown Object"),
            tags=result.get("tags", []),
            description=result.get("description", ""),
            confidence=result.get("confidence", "medium"),
        )

    except Exception as e:
        log.warning("Vision identification failed (non-fatal): %s", e)
        return None
