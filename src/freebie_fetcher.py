"""Fetch free AI / cloud credits from the public freebie-radar API.

Source: https://github.com/ZiChuanLan/freebie-radar (demo instance at
radar.zichuanlan.top). Only a whitelisted, sanitised subset of fields is kept;
the page renders it as plain text.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

RADAR_API = "https://radar.zichuanlan.top/api/items"
RADAR_PARAMS = {
    "category": "ai,cloud,dev,opensource",
    "obtain": "free,conditional",
    "kind": "freebie,deal",
    "aiOnly": "1",
    "unexpired": "1",
    "maxAgeDays": "30",
    "sort": "score",
    "limit": "200",
}
HEADERS = {
    "User-Agent": "ai-deals/1.0",
    "Accept": "application/json",
}

OUTPUT_PATH = Path(__file__).parent.parent / "docs" / "data" / "freebies.json"

MAX_TITLE_LENGTH = 200
MAX_SUMMARY_LENGTH = 300
MAX_SHORT_FIELD_LENGTH = 40

SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{8,}|[A-Za-z0-9+/=_]{24,}")
ISO_TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z?$")
SHORT_FIELD_PATTERN = re.compile(r"^[a-z0-9_\-]+$")


def redact_secrets(text: str) -> str:
    return SECRET_PATTERN.sub("[redacted]", text)


def clean_text(value: Any, max_length: int) -> str:
    if not isinstance(value, str):
        return ""
    collapsed = " ".join(value.split())
    redacted = redact_secrets(collapsed)
    if len(redacted) > max_length:
        return redacted[: max_length - 1].rstrip() + "…"
    return redacted


def clean_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    url = value.strip()
    if not url.startswith(("https://", "http://")) or len(url) > 2000:
        return None
    if any(character.isspace() or character in "\"'<>" for character in url):
        return None
    return url


def clean_short_field(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if len(value) > MAX_SHORT_FIELD_LENGTH or not SHORT_FIELD_PATTERN.match(value):
        return None
    return value


def clean_timestamp(value: Any) -> str | None:
    if isinstance(value, str) and ISO_TIMESTAMP_PATTERN.match(value):
        return value
    return None


def clean_score(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0, min(100, int(value)))


def sanitize_item(raw: dict[str, Any]) -> dict[str, Any] | None:
    url = clean_url(raw.get("url"))
    title = clean_text(raw.get("title"), MAX_TITLE_LENGTH)
    if not url or not title:
        return None
    return {
        "title": title,
        "summary": clean_text(raw.get("summary"), MAX_SUMMARY_LENGTH),
        "url": url,
        "category": clean_short_field(raw.get("category")),
        "obtain": clean_short_field(raw.get("obtain")),
        "source": clean_short_field(raw.get("source_id")),
        "score": clean_score(raw.get("score")),
        "published_at": clean_timestamp(raw.get("published_at") or raw.get("first_seen_at")),
        "expires_at": clean_timestamp(raw.get("expires_at")),
    }


def fetch_freebies(timeout: int = 30) -> list[dict[str, Any]]:
    response = requests.get(RADAR_API, params=RADAR_PARAMS, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    raw_items = response.json().get("items")
    if not isinstance(raw_items, list):
        raise ValueError("freebie-radar response has no items list")
    sanitized = (sanitize_item(item) for item in raw_items if isinstance(item, dict))
    return [item for item in sanitized if item]


def write_freebies(items: list[dict[str, Any]], generated_at: datetime, path: Path = OUTPUT_PATH) -> None:
    payload = {
        "generated_at": generated_at.strftime("%Y-%m-%d %H:%M UTC"),
        "source": "https://github.com/ZiChuanLan/freebie-radar",
        "items": items,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def update_freebies(generated_at: datetime, path: Path = OUTPUT_PATH) -> int:
    try:
        items = fetch_freebies()
    except (requests.RequestException, ValueError) as exc:
        log.warning("freebie-radar fetch failed (%s); keeping previous freebies.json", exc)
        return 0
    if not items:
        log.warning("freebie-radar returned 0 usable items; keeping previous freebies.json")
        return 0
    write_freebies(items, generated_at, path)
    log.info("freebie-radar: %d items written to %s", len(items), path)
    return len(items)
