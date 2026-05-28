"""Fetch model rankings from Arena AI (LMSYS Chatbot Arena) via community API.

API: https://api.wulong.dev/arena-ai-leaderboards/v1/leaderboard?name={name}
No auth required. Returns Elo scores, confidence intervals, vote counts.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import requests

from src.models import RawModelRecord

log = logging.getLogger(__name__)

ARENA_BASE = "https://api.wulong.dev/arena-ai-leaderboards/v1"
LEADERBOARDS = ["text", "code"]

HEADERS = {
    "User-Agent": "ai-deals/1.0",
    "Accept": "application/json",
}


# ---------------------------------------------------------------------------
# HTTP fetch
# ---------------------------------------------------------------------------


def _fetch_json(url: str, retries: int = 3, backoff: float = 2.0) -> Optional[dict]:
    """Fetch JSON from a URL with exponential backoff. Returns None on failure."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else 0
            if status == 429:
                wait = backoff * (2**attempt)
                log.warning("Arena API rate-limited, retrying in %.0fs", wait)
                time.sleep(wait)
            else:
                log.warning("Arena API HTTP %d (attempt %d/%d)", status, attempt + 1, retries)
                if attempt == retries - 1:
                    return None
        except requests.RequestException as exc:
            log.warning("Arena API request error: %s (attempt %d/%d)", exc, attempt + 1, retries)
            if attempt == retries - 1:
                return None
        time.sleep(backoff)
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _derive_slug(name: str) -> str:
    """Derive a URL-slug-style identifier from a model name."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def parse_arena_models(raw_models: list[dict], leaderboard: str) -> list[RawModelRecord]:
    """Map Arena API model dicts to RawModelRecord instances."""
    records: list[RawModelRecord] = []
    source = f"arena-{leaderboard}"

    for m in raw_models:
        name = m.get("model", "").strip()
        if not name:
            continue

        # Strip harness annotations like "(codex-harness)" — these are
        # evaluation infrastructure labels, not part of the model identity
        name = re.sub(r"\s*\(codex-harness\)", "", name, flags=re.IGNORECASE)

        kwargs: dict = {
            "model_id": _derive_slug(name),
            "name": name,
            "source_pages": [source],
        }

        vendor = m.get("vendor")
        if vendor:
            kwargs["creator"] = vendor

        score = m.get("score")
        ci = m.get("ci")
        votes = m.get("votes")

        if leaderboard == "text":
            kwargs["arena_elo"] = score
            kwargs["arena_ci"] = ci
            kwargs["arena_votes"] = votes
        elif leaderboard == "code":
            kwargs["arena_coding_elo"] = score

        try:
            records.append(RawModelRecord(**kwargs))
        except Exception as exc:
            log.debug("Skipping Arena model %s: %s", name, exc)

    return records


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def fetch_all_arena() -> list[RawModelRecord]:
    """Fetch both text and code leaderboards from Arena AI."""
    all_records: list[RawModelRecord] = []
    for lb in LEADERBOARDS:
        url = f"{ARENA_BASE}/leaderboard?name={lb}"
        log.info("Fetching Arena %s leaderboard...", lb)
        data = _fetch_json(url)
        if data is None:
            log.warning("Failed to fetch Arena %s leaderboard, skipping", lb)
            continue
        models = data.get("models", [])
        records = parse_arena_models(models, lb)
        log.info("Arena %s: %d models", lb, len(records))
        all_records.extend(records)
    return all_records
