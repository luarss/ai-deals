"""Fetch model rankings from Arena AI via Jina Reader (markdown conversion).

The Arena AI leaderboard is a Next.js SPA. Jina Reader (r.jina.ai) renders it
as markdown, exposing the full 361-model table that we parse directly.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import requests

from src.models import RawModelRecord

log = logging.getLogger(__name__)

JINA_READER = "https://r.jina.ai/"
ARENA_TEXT_URL = "https://arena.ai/leaderboard/text"
ARENA_CODE_URL = "https://arena.ai/leaderboard/code"

HEADERS = {
    "User-Agent": "ai-deals/1.0",
    "Accept": "application/json",
    "X-Return-Format": "markdown",
}

# Matches vendor and license from "Vendor · License" text
_VENDOR_RE = re.compile(r"([^·]+?)\s*·\s*(.+)")

# Extract model name and slug from a markdown link: [name](url "slug")
_LINK_RE = re.compile(r'\[([^\]]+)\]\([^"]*(?:"([^"]+)")?\)')


def _derive_slug(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def _parse_context(text: str) -> Optional[int]:
    """Parse context string like '1M', '262K', '2000' into K-tokens."""
    text = text.strip().upper().replace(",", "")
    if not text or text in ("N/A", "—", "-", ""):
        return None
    try:
        if text.endswith("M"):
            return int(float(text[:-1]) * 1000)
        if text.endswith("K"):
            return int(float(text[:-1]))
        if text.endswith("B"):
            return int(float(text[:-1]) * 1_000_000)
        return int(float(text))
    except ValueError:
        return None


def _parse_int(text: str) -> Optional[int]:
    if text is None:
        return None
    try:
        return int(text.strip().replace(",", ""))
    except (ValueError, TypeError):
        return None


def _parse_price(text: str) -> Optional[float]:
    if text is None:
        return None
    try:
        return float(text.strip().replace("$", "").replace(",", ""))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# HTTP fetch via Jina Reader
# ---------------------------------------------------------------------------


def _fetch_markdown(url: str, retries: int = 3, backoff: float = 3.0) -> Optional[str]:
    """Fetch a page via Jina Reader as markdown. Returns markdown text or None."""
    reader_url = f"{JINA_READER}{url}"
    for attempt in range(retries):
        try:
            resp = requests.get(reader_url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("data", {}).get("content", "")
            if not content:
                log.warning("Jina Reader returned empty content for %s", url)
                return None
            return content
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else 0
            if status == 429:
                wait = backoff * (2 ** attempt)
                log.warning("Jina Reader rate-limited, retrying in %.0fs", wait)
                time.sleep(wait)
            else:
                log.warning("Jina Reader HTTP %d (attempt %d/%d)", status, attempt + 1, retries)
                if attempt == retries - 1:
                    return None
        except Exception as exc:
            log.warning("Jina Reader error: %s (attempt %d/%d)", exc, attempt + 1, retries)
            if attempt == retries - 1:
                return None
        time.sleep(backoff)
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_table_cell(cell: str, leaderboard: str) -> dict:
    """Parse a single model's table row cells into a field dict."""
    fields: dict = {"source_pages": [f"arena-{leaderboard}"]}

    # --- Cell 0: Model link + vendor info ---
    # Format: [model-name](url "slug") Vendor · License
    cell0 = cell.strip()
    link_match = _LINK_RE.search(cell0)
    if link_match:
        fields["name"] = link_match.group(1).strip()
        if link_match.group(2):
            fields["model_id"] = link_match.group(2).strip()
        # Vendor + license text is everything after the link
        rest = cell0[link_match.end():].strip()
    else:
        fields["name"] = cell0
        rest = ""

    # Extract vendor from "Vendor · License"
    vm = _VENDOR_RE.match(rest)
    if vm:
        vendor = vm.group(1).strip()
        if vendor and vendor.lower() not in ("n/a", "unknown"):
            fields["creator"] = vendor

    # Derive slug if not from link
    if "model_id" not in fields:
        fields["model_id"] = _derive_slug(fields["name"])

    return fields


def _parse_score_cell(cell: str) -> tuple[Optional[int], Optional[int]]:
    """Parse score cell like '1502±4' or '1567+10/-10'."""
    cell = cell.strip()
    # Remove trailing annotations
    cell = re.sub(r"\s*(Preliminary|final|confirmed)\s*", "", cell, flags=re.IGNORECASE)
    if "±" in cell:
        parts = cell.split("±")
        score = _parse_int(parts[0])
        ci = _parse_int(parts[1]) if len(parts) > 1 else None
        return score, ci
    # Code leaderboard format: "1567+10/-10"
    m = re.match(r"(\d+)\+(\d+)/-(\d+)", cell)
    if m:
        score = _parse_int(m.group(1))
        # Average the +/- as a single CI value
        ci = (_parse_int(m.group(2)) + _parse_int(m.group(3))) // 2
        return score, ci
    # Plain integer
    score = _parse_int(cell)
    return score, None


def _parse_price_cell(cell: str) -> tuple[Optional[float], Optional[float]]:
    """Parse price cell like '$5 / $25' or 'N/A'."""
    cell = cell.strip()
    if cell.upper() in ("N/A", "—", "-", ""):
        return None, None
    parts = cell.split("/")
    inp = _parse_price(parts[0]) if len(parts) > 0 else None
    out = _parse_price(parts[1]) if len(parts) > 1 else None
    return inp, out


def _parse_markdown_table(markdown: str, leaderboard: str) -> list[RawModelRecord]:
    """Parse the Arena leaderboard markdown table."""
    records: list[RawModelRecord] = []

    # Find the table: lines starting with "| Rank |"
    lines = markdown.split("\n")
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("| Rank |"):
            header_idx = i
            break

    if header_idx is None:
        log.warning("Could not find leaderboard table in markdown")
        return []

    # Parse header to find column positions
    header_cells = [c.strip() for c in lines[header_idx].split("|")]
    # Skip the empty first/last from split
    header_cells = [c for c in header_cells if c]

    # Map header names to indices
    col_map: dict[str, int] = {}
    for idx, h in enumerate(header_cells):
        h_lower = h.lower().replace("$", "").strip()
        if "rank" in h_lower and "spread" not in h_lower:
            col_map["rank"] = idx
        elif "model" in h_lower:
            col_map["model"] = idx
        elif "score" in h_lower:
            col_map["score"] = idx
        elif "votes" in h_lower:
            col_map["votes"] = idx
        elif "price" in h_lower:
            col_map["price"] = idx
        elif "context" in h_lower:
            col_map["context"] = idx

    # Data rows start after header + separator
    data_start = header_idx + 2

    for line in lines[data_start:]:
        if not line.startswith("|"):
            break

        cells = [c.strip() for c in line.split("|")]
        # Remove leading/trailing empty strings from split
        cells = [c for c in cells if c or cells.index(c) > 0]

        # Get cell by column
        def _cell(col_name: str) -> str:
            idx = col_map.get(col_name)
            if idx is not None and idx < len(cells):
                return cells[idx]
            return ""

        model_cell = _cell("model")
        if not model_cell:
            continue

        fields = _parse_table_cell(model_cell, leaderboard)

        # Score: Elo ± CI
        score_cell = _cell("score")
        if score_cell:
            score, ci = _parse_score_cell(score_cell)
            if leaderboard == "text":
                fields["arena_elo"] = score
                fields["arena_ci"] = ci
            elif leaderboard == "code":
                fields["arena_coding_elo"] = score

        # Votes
        votes_cell = _cell("votes")
        if votes_cell:
            fields["arena_votes"] = _parse_int(votes_cell)

        # Price
        price_cell = _cell("price")
        if price_cell:
            pin, pout = _parse_price_cell(price_cell)
            if pin is not None:
                fields["price_input"] = pin
            if pout is not None:
                fields["price_output"] = pout

        # Context
        ctx_cell = _cell("context")
        if ctx_cell:
            ctx = _parse_context(ctx_cell)
            if ctx is not None:
                fields["context_window_k"] = ctx

        try:
            records.append(RawModelRecord(**fields))
        except Exception as exc:
            log.debug("Skipping Arena model %s: %s", fields.get("name", "?"), exc)

    return records


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def fetch_all_arena() -> list[RawModelRecord]:
    """Fetch text and code leaderboards from Arena AI via Jina Reader."""
    all_records: list[RawModelRecord] = []

    for lb, url in [("text", ARENA_TEXT_URL), ("code", ARENA_CODE_URL)]:
        log.info("Fetching Arena %s leaderboard via Jina Reader...", lb)
        markdown = _fetch_markdown(url)
        if markdown is None:
            log.warning("Failed to fetch Arena %s, skipping", lb)
            continue
        records = _parse_markdown_table(markdown, lb)
        log.info("Arena %s: %d models", lb, len(records))
        all_records.extend(records)

    return all_records
