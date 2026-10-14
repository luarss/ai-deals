"""Fetch pages from artificialanalysis.ai and extract model data.

The site is a Next.js App Router app. The leaderboard page streams its full
model dataset inside React Server Component flight chunks
(``self.__next_f.push([1, "..."])`` scripts). That payload carries raw
per-model fields (``intelligenceIndex``, ``price1mInputTokens``,
``price1mOutputTokens``, ``cacheHitPrice``, ``medianOutputTokensPerSecond``,
``contextWindowTokens`` ...) and is the primary source.

The server-rendered <table> is kept as a fallback. Its current columns are:
  Model | Context Window | Creator | Artificial Analysis Intelligence Index |
  Cost per Task USD | Median Tokens/s | Latency First Chunk (s) |
  Total Response (s) | Further Analysis (slug hrefs)
"Cost per Task" is not a per-token price, so the table yields no prices.
Estimated intelligence values render with a trailing "*" (e.g. "35 *").

The leaderboard no longer publishes a per-model Coding Index or LiveCodeBench
score, so those fields stay empty.
"""

from __future__ import annotations

import json
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
    match = re.match(r"^-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    return float(match.group(0))


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
    soup = BeautifulSoup(html, "lxml")
    records: list[RawModelRecord] = []

    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            blob = json.loads(tag.string or "")
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
# Embedded RSC flight payload (primary data source)
# ---------------------------------------------------------------------------

FLIGHT_PUSH_PATTERN = re.compile(r"^\s*self\.__next_f\.push\((.*)\)\s*;?\s*$", re.S)
MODELS_ARRAY_PATTERN = re.compile(r'"models"\s*:\s*\[')
MODEL_DATA_KEYS = ("intelligenceIndex", "price1mInputTokens", "medianOutputTokensPerSecond")


def _flight_payload(html: str) -> str:
    """Concatenate the string chunks pushed to ``self.__next_f`` by the page."""
    soup = BeautifulSoup(html, "lxml")
    parts: list[str] = []
    for script in soup.find_all("script"):
        match = FLIGHT_PUSH_PATTERN.match(script.string or "")
        if not match:
            continue
        try:
            chunk = json.loads(match.group(1))
        except ValueError:
            continue
        if len(chunk) > 1 and chunk[0] == 1 and isinstance(chunk[1], str):
            parts.append(chunk[1])
    return "".join(parts)


def _is_model_data(entry: object) -> bool:
    return isinstance(entry, dict) and "slug" in entry and any(key in entry for key in MODEL_DATA_KEYS)


def _find_model_dataset(payload: str) -> list[dict]:
    """Return the largest embedded ``"models": [...]`` array of model data objects."""
    decoder = json.JSONDecoder()
    best: list[dict] = []
    for match in MODELS_ARRAY_PATTERN.finditer(payload):
        try:
            candidate, _ = decoder.raw_decode(payload, match.end() - 1)
        except ValueError:
            continue
        models = [entry for entry in candidate if _is_model_data(entry)]
        if len(models) > len(best):
            best = models
    return best


def _optional_float(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _rounded(value: object, digits: int) -> Optional[float]:
    number = _optional_float(value)
    return None if number is None else round(number, digits)


def blended_price(
    price_input: Optional[float],
    price_output: Optional[float],
    cache_hit_price: Optional[float],
) -> Optional[float]:
    """Artificial Analysis 7:2:1 cached-input/input/output blend in $/1M tokens.

    Without a published cache-hit price, cached input is billed at the input rate.
    """
    if price_input is None or price_output is None:
        return None
    cached_input = cache_hit_price if cache_hit_price is not None else price_input
    return round((7 * cached_input + 2 * price_input + price_output) / 10, 6)


def _context_window_k(tokens: object) -> Optional[int]:
    token_count = _optional_float(tokens)
    if token_count is None or token_count <= 0:
        return None
    return round(token_count / 1000)


def _record_from_model_data(entry: dict, page_key: str) -> Optional[RawModelRecord]:
    slug = entry.get("slug")
    name = entry.get("shortName") or entry.get("name")
    if not slug or not name:
        return None
    price_input = _optional_float(entry.get("price1mInputTokens"))
    price_output = _optional_float(entry.get("price1mOutputTokens"))
    cache_hit_price = _optional_float(entry.get("cacheHitPrice"))
    return RawModelRecord(
        model_id=slug,
        name=name.strip(),
        creator=entry.get("modelCreatorName"),
        intelligence_score=_rounded(entry.get("intelligenceIndex"), 1),
        price_input=price_input,
        price_output=price_output,
        price_blended=blended_price(price_input, price_output, cache_hit_price),
        output_speed_tps=_rounded(entry.get("medianOutputTokensPerSecond"), 1),
        context_window_k=_context_window_k(entry.get("contextWindowTokens")),
        source_pages=[page_key],
    )


def extract_embedded_models(html: str, page_key: str) -> list[RawModelRecord]:
    """Parse the model dataset embedded in the page's RSC flight payload.

    Deprecated models are dropped, matching the rows the leaderboard table shows.
    """
    dataset = _find_model_dataset(_flight_payload(html))
    records: list[RawModelRecord] = []
    seen_slugs: set[str] = set()
    for entry in dataset:
        if entry.get("deprecated") or entry.get("slug") in seen_slugs:
            continue
        record = _record_from_model_data(entry, page_key)
        if record is None:
            continue
        seen_slugs.add(record.model_id)
        records.append(record)
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

    records = extract_embedded_models(html, page_key)
    log.info("Embedded payload parse: %d records from %s", len(records), page_key)
    if not records:
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
