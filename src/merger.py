"""Deduplicate and merge RawModelRecord lists from multiple sources.

Grouping (one group = one output row, keyed by a stable ``model_id``):
  1. A record whose slug is already a group joins that group.
  2. Otherwise a record whose ``normalise_name`` key is registered joins that
     group, unless the group already holds a record from the same source page
     under a different slug (one source never lists the same model twice).
  3. Otherwise the record's own slug starts a new group; slug-less records
     fall back to a guarded fuzzy match and finally a slug minted from the name.

``normalise_name`` drops "Creator: " / "provider/" prefixes and date stamps but
keeps variant tags, so AA reasoning-effort rows ("(xhigh)", "(non-reasoning)")
and OpenRouter tiers ("(batch)", ":free") stay distinct rows.

OpenRouter linking: a group with no OpenRouter data inherits the OpenRouter
price, id and uptime of the single non-variant OpenRouter model whose
``base_model_key`` (variant tags, noise tokens and vendor words stripped)
matches one of the group's names or slugs. Keys are tried strict first, then
loose (date stamps, "preview"/"exp", trailing ".0" dropped); an ambiguous key
is skipped.

Field coalescion priority: leaderboard > coding > livecodebench > arena > openrouter
"""

from __future__ import annotations

import logging
import re
from difflib import get_close_matches
from typing import Any

from src.models import MergedModel, RawModelRecord

log = logging.getLogger(__name__)

PAGE_PRIORITY: dict[str, int] = {
    "leaderboard": 0,
    "coding": 1,
    "livecodebench": 2,
    "arena-text": 3,
    "arena-code": 4,
    "openrouter": 5,
}

OPENROUTER_PAGE = "openrouter"

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
    "arena_elo",
    "arena_ci",
    "arena_votes",
    "arena_coding_elo",
    "openrouter_price_input",
    "openrouter_price_output",
    "openrouter_model_id",
    "uptime_30m",
    "uptime_1d",
)

LIST_FIELDS = (
    "input_modalities",
    "output_modalities",
)

INHERITED_OPENROUTER_FIELDS = (
    "openrouter_price_input",
    "openrouter_price_output",
    "openrouter_model_id",
    "uptime_30m",
    "uptime_1d",
)

EFFORT_TAGS = (
    "max",
    "xhigh",
    "high",
    "medium",
    "low",
    "minimal",
    "adaptive",
    "reasoning",
    "non-reasoning",
    "thinking",
    "non-thinking",
)

OPENROUTER_VARIANT_TAGS = ("batch", "free", "nitro", "extended", "floor", "online", "exacto")

VENDOR_WORDS = frozenset(
    {"nvidia", "meta", "google", "openai", "anthropic", "amazon", "ibm", "microsoft", "mistralai"}
)

NOISE_TOKENS = frozenset({"instruct", "it"})

LOOSE_NOISE_TOKENS = frozenset({"preview", "exp", "experimental"})

_CREATOR_PREFIX_RE = re.compile(r"^[^:/()]{1,40}:\s+")
_PROVIDER_PREFIX_RE = re.compile(r"^~?[a-z0-9_.-]+/")
_OPENROUTER_TAG_RE = re.compile(r":([a-z-]+)$")
_EFFORT_PAREN_RE = re.compile(
    r"\s*\((?:" + "|".join(re.escape(tag) for tag in EFFORT_TAGS) + r")(?:\s+with\s+fallback)?\)",
    flags=re.IGNORECASE,
)
_VARIANT_PAREN_RE = re.compile(
    r"\s*\((?:" + "|".join(OPENROUTER_VARIANT_TAGS) + r")\)",
    flags=re.IGNORECASE,
)
_DATE_TOKEN_RE = re.compile(r"^(?:\d{4}|\d{6}|\d{8})$")
_SLUG_EFFORT_WORDS = frozenset(tag for tag in EFFORT_TAGS if "-" not in tag)


def strip_creator_prefix(name: str) -> str:
    return _CREATOR_PREFIX_RE.sub("", name.strip())


def normalise_name(name: str) -> str:
    """Return a compact lowercase identity key that keeps variant tags.

    Examples:
        "GPT-4o Mini (Nov '24)" -> "gpt 4o mini"
        "Anthropic: Claude Haiku 5.5" -> "claude haiku 5 5"
        "Claude Haiku 5.5 (xhigh)" -> "claude haiku 5 5 xhigh"
        "anthropic/claude-haiku-5.5:batch" -> "claude haiku 5 5 batch"
    """
    name = strip_creator_prefix(name)
    name = _PROVIDER_PREFIX_RE.sub("", name)
    name = _OPENROUTER_TAG_RE.sub(r" \1", name)
    name = re.sub(
        r"\s*\((?:"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[^)]*"
        r"|\d{4,}[^)]*"
        r")\)",
        "",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(r"\(([^)]+)\)", r"\1", name)
    name = name.replace("+", " plus ")
    name = re.sub(r"[-_]\d{6,}", "", name)
    name = re.sub(r"[-_.]", " ", name)
    name = re.sub(r"[^a-z0-9 ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


def derive_slug(name: str) -> str:
    """Derive a URL-slug-style identifier from a display name."""
    return normalise_name(name).replace(" ", "-")


def is_openrouter_variant(record: RawModelRecord) -> bool:
    model_id = record.openrouter_model_id or ""
    if model_id.startswith("~") or ":" in model_id:
        return True
    return bool(_VARIANT_PAREN_RE.search(record.name))


def _strip_trailing_effort_words(tokens: list[str]) -> list[str]:
    while len(tokens) > 1:
        if len(tokens) > 2 and tokens[-2] == "non" and tokens[-1] in ("reasoning", "thinking"):
            tokens = tokens[:-2]
        elif tokens[-1] in _SLUG_EFFORT_WORDS:
            tokens = tokens[:-1]
        else:
            break
    return tokens


def _loosen(tokens: list[str]) -> list[str]:
    loose: list[str] = []
    for token in tokens:
        if _DATE_TOKEN_RE.match(token) or token in LOOSE_NOISE_TOKENS:
            continue
        if token == "0" and loose and loose[-1].isdigit():
            continue
        loose.append(token)
    return loose


def base_model_key(name: str, loose: bool = False, is_slug: bool = False) -> str:
    """Return the variant-free identity of a model name or slug.

    Examples:
        "Claude Haiku 5.5 (xhigh with fallback)" -> "claude haiku 5 5"
        "DeepSeek: DeepSeek V4.1 Flash (batch)" -> "deepseek v4 1 flash"
        "nvidia-nemotron-3-nano-30b-a3b-reasoning" (slug) -> "nemotron 3 nano 30b a3b"
        "DeepSeek V4 Flash Vision Exp" (loose) -> "deepseek v4 flash vision"
    """
    name = strip_creator_prefix(name)
    name = _OPENROUTER_TAG_RE.sub("", name)
    name = _EFFORT_PAREN_RE.sub("", name)
    name = _VARIANT_PAREN_RE.sub("", name)
    tokens = normalise_name(name).split()
    tokens = [t for t in tokens if t not in NOISE_TOKENS]
    if len(tokens) > 1 and tokens[-1] == "v1":
        tokens = tokens[:-1]
    if is_slug:
        tokens = _strip_trailing_effort_words(tokens)
    if len(tokens) > 1 and tokens[0] in VENDOR_WORDS:
        tokens = tokens[1:]
    if loose:
        tokens = _loosen(tokens)
    return " ".join(tokens)


def _priority(record: RawModelRecord) -> int:
    """Return the minimum page priority across all source pages (lower = higher priority)."""
    if not record.source_pages:
        return 99
    return min(PAGE_PRIORITY.get(p, 50) for p in record.source_pages)


def _sorted_pages(pages: set[str]) -> list[str]:
    return sorted(pages, key=lambda page: (PAGE_PRIORITY.get(page, 50), page))


def coalesce_fields(group: list[RawModelRecord]) -> dict[str, Any]:
    """Merge a group of records into one dict, preferring higher-priority sources."""
    sorted_group = sorted(group, key=_priority)

    merged: dict[str, Any] = {
        "model_id": sorted_group[0].model_id,
        "name": sorted_group[0].name,
        "source_pages": _sorted_pages({p for rec in group for p in rec.source_pages}),
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

    return merged


def _shares_source_under_other_slug(record: RawModelRecord, group: list[RawModelRecord]) -> bool:
    pages = set(record.source_pages)
    return any(
        member.model_id != record.model_id and pages & set(member.source_pages)
        for member in group
    )


def _resolve_slug(
    record: RawModelRecord,
    slug_registry: dict[str, str],
    groups: dict[str, list[RawModelRecord]],
) -> str:
    """Return the canonical slug for a record, updating the name registry as needed."""
    norm = normalise_name(record.name)
    has_real_slug = bool(record.model_id and re.search(r"[a-z]", record.model_id))

    if has_real_slug and record.model_id in groups:
        slug_registry.setdefault(norm, record.model_id)
        return record.model_id

    registered = slug_registry.get(norm)
    if registered and not _shares_source_under_other_slug(record, groups[registered]):
        return registered

    if has_real_slug:
        slug_registry.setdefault(norm, record.model_id)
        return record.model_id

    matches = get_close_matches(norm, list(slug_registry), n=1, cutoff=0.82)
    if matches:
        canonical = slug_registry[matches[0]]
        if not _shares_source_under_other_slug(record, groups[canonical]):
            slug_registry[norm] = canonical
            log.debug("Fuzzy matched '%s' -> '%s'", record.name, canonical)
            return canonical

    new_slug = derive_slug(record.name)
    slug_registry.setdefault(norm, new_slug)
    return new_slug


def _record_keys(record: RawModelRecord, loose: bool) -> list[str]:
    keys = [base_model_key(record.name, loose=loose)]
    if record.openrouter_model_id:
        keys.append(base_model_key(record.openrouter_model_id, loose=loose))
    if record.model_id:
        keys.append(base_model_key(record.model_id, loose=loose, is_slug=True))
    return [key for key in dict.fromkeys(keys) if key]


def _build_openrouter_index(
    groups: dict[str, list[RawModelRecord]], loose: bool
) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for slug, group in groups.items():
        for record in group:
            if OPENROUTER_PAGE not in record.source_pages or is_openrouter_variant(record):
                continue
            for key in _record_keys(record, loose):
                index.setdefault(key, set()).add(slug)
    return index


def _find_openrouter_parent(
    group: list[RawModelRecord], indexes: list[dict[str, set[str]]]
) -> str | None:
    for loose, index in enumerate(indexes):
        for record in group:
            for key in _record_keys(record, bool(loose)):
                candidates = index.get(key, set())
                if len(candidates) == 1:
                    return next(iter(candidates))
    return None


def _has_openrouter_price(fields: dict[str, Any]) -> bool:
    return (
        fields.get("openrouter_price_input") is not None
        or fields.get("openrouter_price_output") is not None
    )


def inherit_openrouter_fields(
    groups: dict[str, list[RawModelRecord]], coalesced: dict[str, dict[str, Any]]
) -> int:
    """Copy OpenRouter price/uptime from a matching base model into groups lacking them."""
    indexes = [_build_openrouter_index(groups, loose=False), _build_openrouter_index(groups, loose=True)]
    linked = 0
    for slug, fields in coalesced.items():
        if _has_openrouter_price(fields):
            continue
        parent_slug = _find_openrouter_parent(groups[slug], indexes)
        if parent_slug is None or parent_slug == slug:
            continue
        parent = coalesced[parent_slug]
        if not _has_openrouter_price(parent):
            continue
        for field in INHERITED_OPENROUTER_FIELDS:
            if fields.get(field) is None and parent.get(field) is not None:
                fields[field] = parent[field]
        fields["source_pages"] = _sorted_pages(set(fields["source_pages"]) | {OPENROUTER_PAGE})
        linked += 1
        log.debug("Linked '%s' to OpenRouter parent '%s'", slug, parent_slug)
    return linked


def merge_records(records: list[RawModelRecord]) -> list[MergedModel]:
    """Deduplicate and merge raw records into a list of MergedModel instances."""
    if not records:
        return []

    slug_registry: dict[str, str] = {}
    groups: dict[str, list[RawModelRecord]] = {}

    for rec in records:
        slug = _resolve_slug(rec, slug_registry, groups)
        groups.setdefault(slug, []).append(rec)

    coalesced = {slug: {**coalesce_fields(group), "model_id": slug} for slug, group in groups.items()}
    linked = inherit_openrouter_fields(groups, coalesced)

    merged: list[MergedModel] = []
    for slug, fields in coalesced.items():
        try:
            merged.append(MergedModel(**fields))
        except Exception as exc:
            log.warning("Could not create MergedModel for slug '%s': %s", slug, exc)

    log.info(
        "Merged %d raw records -> %d unique models (%d linked to an OpenRouter base model)",
        len(records),
        len(merged),
        linked,
    )
    return merged
