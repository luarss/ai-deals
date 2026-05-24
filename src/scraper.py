"""Fetch pages from artificialanalysis.ai and extract model data.

The site is a Next.js app with server-rendered HTML. Each page exposes a
<table> with a two-row header and one row per model. We parse that table
by locating column positions from row 1's header text, then reading data
from rows 2+.

Page-specific column schemas (discovered empirically):

leaderboard (/leaderboards/models):
  0: Model name  1: Context  2: Creator  3: Intelligence  4: Blended price
  5: Speed(t/s)  6: Latency  7: Total response  8: Further Analysis (slug hrefs)

coding (/models/capabilities/coding):
  Similar structure; "Coding Index" column may appear in position 3 or 4.

livecodebench (/evaluations/livecodebench):
  0: Model name  1: Score(%)  2+: other columns
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup, Tag

from src.models import RawModelRecord

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://artificialanalysis.ai/",
}

PAGES: dict[str, str] = {
    # Server-renders a full HTML table with 200+ models (intelligence, price, speed, context)
    "leaderboard": "https://artificialanalysis.ai/leaderboards/models",
    # Coding/livecodebench pages are client-side rendered (no table/useful JSON-LD via HTTP)
    # Add more pages here when AA adds server-rendered leaderboards for other capabilities
}


# ---------------------------------------------------------------------------
# HTTP fetch
# ---------------------------------------------------------------------------


def fetch_html(url: str, retries: int = 3, backoff: float = 2.0) -> str:
    """Fetch a URL with exponential backoff. Returns "" on final failure."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else 0
            if status == 429:
                wait = backoff * (2**attempt)
                log.warning("Rate-limited (%s), retrying in %.0fs", url, wait)
                time.sleep(wait)
            else:
                log.warning("HTTP %d from %s (attempt %d/%d)", status, url, attempt + 1, retries)
                if attempt == retries - 1:
                    return ""
        except requests.RequestException as exc:
            log.warning("Request error (%s): %s (attempt %d/%d)", url, exc, attempt + 1, retries)
            if attempt == retries - 1:
                return ""
        time.sleep(backoff)
    return ""


# ---------------------------------------------------------------------------
# Cell-value helpers
# ---------------------------------------------------------------------------


def _cell_text(cell: Tag) -> str:
    """Return cleaned text content of a table cell."""
    return cell.get_text(" ", strip=True)


def _parse_float(text: str) -> Optional[float]:
    """Strip currency/percent symbols and parse as float. Returns None on failure."""
    cleaned = text.strip().replace("$", "").replace("%", "").replace(",", "")
    if cleaned in ("", "—", "-", "N/A", "n/a"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_context(text: str) -> Optional[int]:
    """Parse context window strings like '922k', '1M', '10M' into K-token integers."""
    text = text.strip().lower().replace(",", "")
    if not text or text in ("—", "-"):
        return None
    try:
        if text.endswith("m"):
            return int(float(text[:-1]) * 1000)
        if text.endswith("k"):
            return int(float(text[:-1]))
        return int(float(text))
    except ValueError:
        return None


def _slug_from_href(href: str) -> str:
    """Extract model slug from '/models/gpt-5-5' or '/models/gpt-5-5/providers'."""
    match = re.search(r"/models/([^/?#]+)", href)
    return match.group(1) if match else ""


# ---------------------------------------------------------------------------
# Column-position detection
# ---------------------------------------------------------------------------

# Maps normalised header keywords → RawModelRecord field name
HEADER_FIELD_MAP: dict[str, str] = {
    "model": "name",
    "intelligence index": "intelligence_score",
    "intelligence": "intelligence_score",
    "blended": "price_blended",
    "price": "price_blended",
    "median": "output_speed_tps",
    "tokens/s": "output_speed_tps",
    "speed": "output_speed_tps",
    "context window": "context_window_k",
    "context": "context_window_k",
    "creator": "creator",
    "coding index": "coding_index",
    "coding": "coding_index",
    "livecodebench": "livecodebench_score",
    "score": "livecodebench_score",  # used only on livecodebench page
}


def _header_to_field(header_text: str) -> Optional[str]:
    """Match a header cell's text to a model field name."""
    h = header_text.lower().strip()
    # Prefer longer, more specific keys first to avoid false matches
    for key in sorted(HEADER_FIELD_MAP, key=len, reverse=True):
        if key in h:
            return HEADER_FIELD_MAP[key]
    return None


def _detect_columns(header_row: Tag) -> dict[int, str]:
    """Return a mapping from column index → field name from a header <tr>."""
    field_map: dict[int, str] = {}
    for idx, cell in enumerate(header_row.find_all(["th", "td"])):
        field = _header_to_field(_cell_text(cell))
        if field and field not in field_map.values():
            field_map[idx] = field
        elif field == "livecodebench_score" and "livecodebench_score" not in field_map.values():
            # Allow a second "score" column on the livecodebench page
            field_map[idx] = field
    return field_map


# ---------------------------------------------------------------------------
# Table parser
# ---------------------------------------------------------------------------


def _is_real_header_row(row: Tag) -> bool:
    """Return True if this row is the actual column header row (not group headers).

    The group-header row (row 0) has category labels like "Intelligence", "Price",
    "Speed" but NOT "Model". The real column-header row (row 1) contains "Model"
    and has no colspan > 1 cells. Data rows use <td> not <th>.
    """
    has_th = bool(row.find("th"))
    text = _cell_text(row).lower()
    # "model" appears only in the real column header row, not in the group header row
    return has_th and "model" in text


def extract_table(html: str, page_key: str) -> list[RawModelRecord]:
    """Parse the main data table. Handles the 2-row header structure."""
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    if not tables:
        log.debug("No <table> found on %s", page_key)
        return []

    # Prefer the table with the most rows
    table = max(tables, key=lambda t: len(t.find_all("tr")))
    rows = table.find_all("tr")

    # Find the row that contains real column headers
    header_row_idx = 0
    for i, row in enumerate(rows[:3]):  # check first 3 rows only
        if _is_real_header_row(row):
            header_row_idx = i
            break

    header_row = rows[header_row_idx]
    field_map = _detect_columns(header_row)

    if not field_map:
        log.debug("Could not detect columns on %s", page_key)
        return []

    log.debug("Column map for %s: %s", page_key, field_map)

    records: list[RawModelRecord] = []
    data_rows = rows[header_row_idx + 1:]

    for row in data_rows:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        kwargs: dict = {"source_pages": [page_key]}

        # Extract slug from ANY cell's anchor hrefs (the last column has /models/ links)
        slug = ""
        for cell in cells:
            for anchor in cell.find_all("a", href=True):
                s = _slug_from_href(anchor["href"])
                if s and "provider" not in anchor["href"]:
                    slug = s
                    break
            if slug:
                break

        for idx, field in field_map.items():
            if idx >= len(cells):
                continue
            cell = cells[idx]

            raw = _cell_text(cell)

            if field == "name":
                # Strip SVG/icon text noise — keep the first non-empty segment
                parts = [p.strip() for p in raw.split() if p.strip()]
                # The model name is the leading words before any icon text
                name_parts = []
                for p in parts:
                    if re.match(r"^[A-Za-z0-9().+\-_/ ]+$", p):
                        name_parts.append(p)
                    else:
                        break
                val: object = " ".join(name_parts) if name_parts else raw
                kwargs[field] = val
            elif field == "context_window_k":
                kwargs[field] = _parse_context(raw)
            elif field == "creator":
                # Ignore tiny icon filenames that appear in alt text
                creator_text = re.sub(r"\.(svg|png|jpg|webp)", "", raw, flags=re.I).strip()
                if creator_text:
                    kwargs[field] = creator_text
            else:
                kwargs[field] = _parse_float(raw)

        name = kwargs.get("name")
        if not name or not isinstance(name, str) or not name.strip():
            continue

        # Derive slug if not found in hrefs
        if not slug:
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

        kwargs["model_id"] = slug
        kwargs["name"] = name.strip()

        try:
            records.append(RawModelRecord(**kwargs))
        except Exception as exc:
            log.debug("Skipping bad row (%s): %s", name, exc)

    return records


# ---------------------------------------------------------------------------
# FAQ JSON-LD extraction (secondary data source)
# ---------------------------------------------------------------------------


def extract_faq_jsonld(html: str, page_key: str) -> list[RawModelRecord]:
    """Extract lightweight model records from FAQPage JSON-LD (top models only).

    The FAQ contains text like:
      "top models by Intelligence Index are: 1. GPT-5.5 (xhigh) (60), 2. ..."
    We parse this for name + intelligence score as a cross-check.
    """
    import json as _json

    soup = BeautifulSoup(html, "lxml")
    records: list[RawModelRecord] = []

    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            blob = _json.loads(tag.string or "")
        except Exception:
            continue

        if blob.get("@type") != "FAQPage":
            continue

        for item in blob.get("mainEntity", []):
            answer_text = item.get("acceptedAnswer", {}).get("text", "")
            # Pattern: "1. Model Name (score),"
            for match in re.finditer(r"\d+\.\s+(.+?)\s+\((\d+)\)", answer_text):
                raw_name, score_str = match.group(1).strip(), match.group(2)
                try:
                    score = float(score_str)
                except ValueError:
                    continue
                slug = re.sub(r"[^a-z0-9]+", "-", raw_name.lower()).strip("-")
                records.append(
                    RawModelRecord(
                        model_id=slug,
                        name=raw_name,
                        intelligence_score=score,
                        source_pages=[page_key],
                    )
                )

    return records


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def scrape_page(page_key: str, url: str) -> list[RawModelRecord]:
    """Scrape one page: HTML table primary, FAQ JSON-LD as supplement."""
    log.info("Fetching %s ...", url)
    html = fetch_html(url)
    if not html:
        log.warning("Empty response for %s, skipping", page_key)
        return []

    records = extract_table(html, page_key)
    log.info("Table parse: %d records from %s", len(records), page_key)

    # Supplement with FAQ JSON-LD for intelligence scores (leaderboard page only)
    if page_key == "leaderboard" and len(records) < 5:
        faq_records = extract_faq_jsonld(html, page_key)
        log.info("FAQ JSON-LD supplement: %d records", len(faq_records))
        records.extend(faq_records)

    return records


def fetch_all_pages() -> list[RawModelRecord]:
    """Scrape all configured pages and return combined raw records."""
    all_records: list[RawModelRecord] = []
    for page_key, url in PAGES.items():
        try:
            records = scrape_page(page_key, url)
            all_records.extend(records)
        except Exception as exc:
            log.error("Failed to scrape %s: %s", page_key, exc)
    return all_records
