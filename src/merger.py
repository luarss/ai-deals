"""Deduplicate and merge RawModelRecord lists from multiple pages.

Deduplication pipeline:
  1. Slug from href (primary key) — most reliable
  2. Normalised name → canonical slug registry (secondary)
  3. difflib fuzzy match ≥ 0.82 (tertiary fallback)

Field coalescion priority: leaderboard > coding > livecodebench
"""

from __future__ import annotations

import logging
import re
from difflib import get_close_matches
from typing import Any

from src.models import MergedModel, RawModelRecord

log = logging.getLogger(__name__)

# Lower page_key = higher priority when coalescing fields
PAGE_PRIORITY: dict[str, int] = {
    "leaderboard": 0,
    "coding": 1,
    "livecodebench": 2,
    "arena-text": 3,
    "arena-code": 4,
    "openrouter": 5,
}

NULLABLE_FIELDS = (
    "creator",
    "intelligence_score",
    "price_input",
    "price_output",
    "price_blended",
    "output_speed_tps",
    "context_window_k",
    "coding_index",
    "livecodebench_score",
    # Arena AI
    "arena_elo",
    "arena_ci",
    "arena_votes",
    "arena_coding_elo",
    # OpenRouter
    "openrouter_price_input",
    "openrouter_price_output",
    "openrouter_model_id",
    "uptime_30m",
    "uptime_1d",
)

# List fields: take the first non-empty list across the priority-sorted group
LIST_FIELDS = (
    "input_modalities",
    "output_modalities",
)


# ---------------------------------------------------------------------------
# Name normalisation helpers
# ---------------------------------------------------------------------------


def normalise_name(name: str) -> str:
    """Strip date/version noise and return a compact lowercase key.

    Examples:
        "GPT-4o Mini (Nov '24)" → "gpt4o mini"
        "claude-3-7-sonnet-20250219" → "claude37sonnet"
        "DeepSeek V3.1" → "deepseek v31"
        "GPT-5.5 (xhigh)" → "gpt55 xhigh"
    """
    # Strip OpenRouter "provider/" prefix (e.g. "openai/gpt-5.5" → "gpt-5.5")
    name = re.sub(r"^[a-z0-9_-]+/", "", name)
    # Remove date-like parentheticals: (Nov '24), (Feb 2026), (20250219)
    # Keep variant annotations: (xhigh), (high), (Thinking), (Max), etc.
    name = re.sub(
        r"\s*\((?:"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[^)]*"  # month-based
        r"|\d{4,}[^)]*"  # year-based or pure numeric
        r")\)",
        "",
        name,
        flags=re.IGNORECASE,
    )
    # Convert remaining parentheticals to space-separated words: (xhigh) → xhigh
    name = re.sub(r"\(([^)]+)\)", r"\1", name)
    # Remove trailing version strings: v1.2, -20250219, .0
    name = re.sub(r"[-_]\d{6,}", "", name)  # -20250219
    name = re.sub(r"\s+v\d[\d.]*\b", "", name, flags=re.IGNORECASE)
    # Strip hyphens, underscores, dots — collapse to alphanumeric + spaces
    name = re.sub(r"[-_.]", " ", name)
    name = re.sub(r"[^a-z0-9 ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


def derive_slug(name: str) -> str:
    """Derive a URL-slug-style identifier from a display name."""
    return normalise_name(name).replace(" ", "-")


# ---------------------------------------------------------------------------
# Coalescion
# ---------------------------------------------------------------------------


def _priority(record: RawModelRecord) -> int:
    """Return the minimum page priority across all source pages (lower = higher priority)."""
    if not record.source_pages:
        return 99
    return min(PAGE_PRIORITY.get(p, 50) for p in record.source_pages)


def coalesce_fields(group: list[RawModelRecord]) -> dict[str, Any]:
    """Merge a group of records into one dict, preferring higher-priority sources."""
    # Sort by priority ascending (highest priority first)
    sorted_group = sorted(group, key=_priority)

    merged: dict[str, Any] = {
        "model_id": sorted_group[0].model_id,
        "name": sorted_group[0].name,
        "source_pages": sorted(
            {p for rec in group for p in rec.source_pages},
            key=lambda p: PAGE_PRIORITY.get(p, 50),
        ),
    }

    for field in NULLABLE_FIELDS:
        for rec in sorted_group:
            val = getattr(rec, field, None)
            if val is not None:
                merged[field] = val
                break

    for field in LIST_FIELDS:
        for rec in sorted_group:
            val = getattr(rec, field, [])
            if val:
                merged[field] = val
                break

    # Prefer the longest/most-specific name across the group
    names = [rec.name for rec in group if rec.name]
    if names:
        merged["name"] = max(names, key=len)

    return merged


# ---------------------------------------------------------------------------
# Slug resolution
# ---------------------------------------------------------------------------


def _resolve_slug(
    record: RawModelRecord,
    slug_registry: dict[str, str],  # normalised_name → canonical_slug
    known_slugs: set[str],
) -> str:
    """Return the canonical slug for a record, updating registries as needed."""
    norm = normalise_name(record.name)

    # 1. If normalised name already matches an existing group, use that slug
    if norm in slug_registry:
        return slug_registry[norm]

    # 2. Use the record's own slug if it looks like a real slug (contains a letter)
    if record.model_id and re.search(r"[a-z]", record.model_id):
        canonical = record.model_id
        slug_registry[norm] = canonical
        known_slugs.add(canonical)
        return canonical

    # 3. Fuzzy match against known normalised names
    candidates = list(slug_registry.keys())
    if candidates:
        matches = get_close_matches(norm, candidates, n=1, cutoff=0.82)
        if matches:
            canonical = slug_registry[matches[0]]
            slug_registry[norm] = canonical
            log.debug("Fuzzy matched '%s' → '%s'", record.name, canonical)
            return canonical

    # 4. Mint a new slug from the name
    new_slug = derive_slug(record.name)
    slug_registry[norm] = new_slug
    known_slugs.add(new_slug)
    return new_slug


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def merge_records(records: list[RawModelRecord]) -> list[MergedModel]:
    """Deduplicate and merge raw records into a list of MergedModel instances."""
    if not records:
        return []

    slug_registry: dict[str, str] = {}  # normalised_name → canonical_slug
    known_slugs: set[str] = set()
    groups: dict[str, list[RawModelRecord]] = {}

    for rec in records:
        slug = _resolve_slug(rec, slug_registry, known_slugs)
        # Ensure the record carries the resolved slug
        rec_copy = rec.model_copy(update={"model_id": slug})
        groups.setdefault(slug, []).append(rec_copy)

    merged: list[MergedModel] = []
    for slug, group in groups.items():
        fields = coalesce_fields(group)
        try:
            model = MergedModel(**fields)
            merged.append(model)
        except Exception as exc:
            log.warning("Could not create MergedModel for slug '%s': %s", slug, exc)

    log.info("Merged %d raw records → %d unique models", len(records), len(merged))
    return merged
