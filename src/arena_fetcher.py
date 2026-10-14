"""Fetch model rankings from Arena AI via its official Hugging Face dataset.

arena.ai sits behind a Cloudflare managed challenge, so scraping the page
(directly or through Jina Reader) only returns "Just a moment...". Arena
publishes every leaderboard snapshot to the lmarena-ai/leaderboard-dataset
dataset, which the Hugging Face datasets-server exposes as paginated JSON.
The /rows endpoint is used rather than /filter because /filter regularly
times out on the large text subset; "overall" rows are stored contiguously,
so paging stops once they have been read.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import requests

from src.models import RawModelRecord

log = logging.getLogger(__name__)

DATASETS_SERVER_ROWS_URL = "https://datasets-server.huggingface.co/rows"
ARENA_DATASET = "lmarena-ai/leaderboard-dataset"
PAGE_SIZE = 100
OVERALL_CATEGORY = "overall"

LEADERBOARD_CONFIGS = {
    "text": "text_style_control",
    "code": "webdev",
}

HEADERS = {
    "User-Agent": "ai-deals/1.0",
    "Accept": "application/json",
}

_HARNESS_SUFFIX_RE = re.compile(r"\s*\(codex-harness\)", re.IGNORECASE)


def _fetch_page(config: str, offset: int, retries: int = 3, backoff: float = 3.0) -> Optional[dict]:
    params = {
        "dataset": ARENA_DATASET,
        "config": config,
        "split": "latest",
        "offset": offset,
        "length": PAGE_SIZE,
    }
    for attempt in range(retries):
        try:
            response = requests.get(DATASETS_SERVER_ROWS_URL, params=params, headers=HEADERS, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            log.warning("Arena dataset HTTP %d for %s (attempt %d/%d)", status, config, attempt + 1, retries)
        except Exception as exc:
            log.warning("Arena dataset error for %s: %s (attempt %d/%d)", config, exc, attempt + 1, retries)
        if attempt < retries - 1:
            time.sleep(backoff * (2 ** attempt))
    return None


def _fetch_overall_rows(config: str) -> Optional[list[dict]]:
    overall_rows: list[dict] = []
    offset = 0
    total = None
    while total is None or offset < total:
        page = _fetch_page(config, offset)
        if page is None:
            return overall_rows or None
        page_rows = [entry["row"] for entry in page.get("rows", [])]
        if not page_rows:
            break
        total = page.get("num_rows_total", 0)
        page_overall = [row for row in page_rows if row.get("category") == OVERALL_CATEGORY]
        if overall_rows and not page_overall:
            break
        overall_rows.extend(page_overall)
        offset += len(page_rows)
    return overall_rows


def _strip_harness(text: str) -> str:
    return _HARNESS_SUFFIX_RE.sub("", text).strip()


def _confidence_interval(row: dict) -> Optional[float]:
    lower = row.get("rating_lower")
    upper = row.get("rating_upper")
    if lower is None or upper is None:
        return None
    return round((upper - lower) / 2)


def _optional_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _row_to_record(row: dict, leaderboard: str) -> Optional[RawModelRecord]:
    model_name = row.get("model_name")
    rating = row.get("rating")
    if not model_name or rating is None:
        return None

    name = _strip_harness(model_name)
    fields: dict = {
        "model_id": name,
        "name": name,
        "source_pages": [f"arena-{leaderboard}"],
        "arena_votes": _optional_int(row.get("vote_count")),
    }

    organization = (row.get("organization") or "").strip()
    if organization and organization.lower() not in ("n/a", "unknown"):
        fields["creator"] = organization

    if leaderboard == "text":
        fields["arena_elo"] = round(rating)
        fields["arena_ci"] = _confidence_interval(row)
    else:
        fields["arena_coding_elo"] = round(rating)

    try:
        return RawModelRecord(**fields)
    except Exception as exc:
        log.debug("Skipping Arena model %s: %s", name, exc)
        return None


def parse_rows(rows: list[dict], leaderboard: str) -> list[RawModelRecord]:
    records: list[RawModelRecord] = []
    for row in rows:
        if row.get("category", OVERALL_CATEGORY) != OVERALL_CATEGORY:
            continue
        record = _row_to_record(row, leaderboard)
        if record is not None:
            records.append(record)
    return records


def fetch_all_arena() -> list[RawModelRecord]:
    """Fetch the text and code (WebDev) overall leaderboards from Arena AI."""
    all_records: list[RawModelRecord] = []

    for leaderboard, config in LEADERBOARD_CONFIGS.items():
        log.info("Fetching Arena %s leaderboard (%s) from Hugging Face...", leaderboard, config)
        rows = _fetch_overall_rows(config)
        if rows is None:
            log.warning("Failed to fetch Arena %s, skipping", leaderboard)
            continue
        records = parse_rows(rows, leaderboard)
        log.info("Arena %s: %d models", leaderboard, len(records))
        all_records.extend(records)

    return all_records
