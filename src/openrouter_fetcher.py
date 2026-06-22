"""Fetch model pricing from OpenRouter API.

API: https://openrouter.ai/api/v1/models (no auth required)
Returns per-model pricing (per-token), context length, model ID, modalities.
Uptime is fetched in parallel from /api/v1/models/{id}/endpoints.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

from src.models import RawModelRecord

log = logging.getLogger(__name__)

OR_API = "https://openrouter.ai/api/v1/models"

HEADERS = {
    "User-Agent": "ai-deals/1.0",
    "Accept": "application/json",
}

# Map OpenRouter provider prefixes to display names
PROVIDER_MAP: dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "meta": "Meta",
    "meta-llama": "Meta",
    "deepseek": "DeepSeek",
    "qwen": "Alibaba",
    "mistral": "Mistral",
    "mistralai": "Mistral",
    "x-ai": "xAI",
    "amazon": "Amazon",
    "cohere": "Cohere",
    "moonshot": "Moonshot",
    "minimax": "MiniMax",
    "01-ai": "01.AI",
    "nvidia": "NVIDIA",
    "microsoft": "Microsoft",
    "gryphe": "Gryphe",
    "perplexity": "Perplexity",
    "ai21": "AI21",
    "sophosympatheia": "Sophosympatheia",
    "thedrummer": "TheDrummer",
    "undi95": "Undi95",
    "sao10k": "Sao10k",
    "cognitivecomputations": "Cognitive Computations",
    "liquid": "Liquid AI",
    "databricks": "Databricks",
    "NousResearch": "Nous Research",
    "inception": "Inception",
    "upstage": "Upstage",
    "snowflake": "Snowflake",
    "inflection": "Inflection",
    "kimi": "Moonshot",
    "z-ai": "Z.ai",
    "moonshotai": "Moonshot",
    "ling": "Ling",
    "alibaba": "Alibaba",
    "baidu": "Baidu",
    "or": "OpenRouter",
    "openrouter": "OpenRouter",
    "tülu3": "Ai2",
    "allenai": "Ai2",
}


def _extract_creator(model_id: str) -> Optional[str]:
    """Extract creator from OpenRouter model ID prefix (e.g. 'openai/gpt-5.5' → 'OpenAI')."""
    if "/" not in model_id:
        return None
    prefix = model_id.split("/")[0].lower()
    return PROVIDER_MAP.get(prefix)


def _derive_slug(name_or_id: str) -> str:
    """Derive a URL-slug from a model name or OpenRouter ID."""
    # Strip provider prefix
    if "/" in name_or_id:
        name_or_id = name_or_id.split("/", 1)[1]
    slug = name_or_id.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


# ---------------------------------------------------------------------------
# HTTP fetch
# ---------------------------------------------------------------------------


def fetch_openrouter_models() -> Optional[list[dict]]:
    """Fetch model list from OpenRouter API. Returns raw data list or None on failure."""
    for attempt in range(3):
        try:
            resp = requests.get(OR_API, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else 0
            if status == 429:
                wait = 2.0 * (2**attempt)
                log.warning("OpenRouter rate-limited, retrying in %.0fs", wait)
                time.sleep(wait)
            else:
                log.warning("OpenRouter HTTP %d (attempt %d/3)", status, attempt + 1)
                if attempt == 2:
                    return None
        except requests.RequestException as exc:
            log.warning("OpenRouter request error: %s (attempt %d/3)", exc, attempt + 1)
            if attempt == 2:
                return None
        time.sleep(2.0)
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_openrouter_models(raw_data: list[dict]) -> list[RawModelRecord]:
    """Map OpenRouter API model dicts to RawModelRecord instances.

    OpenRouter returns per-token prices; we convert to per-1M-token prices
    to match Artificial Analysis conventions.
    """
    records: list[RawModelRecord] = []
    for m in raw_data:
        model_id = m.get("id", "").strip()
        if not model_id:
            continue

        name = m.get("name", model_id).strip()

        pricing = m.get("pricing", {})
        prompt_str = pricing.get("prompt", "0")
        completion_str = pricing.get("completion", "0")

        try:
            prompt_price = float(prompt_str)
            completion_price = float(completion_str)
        except (ValueError, TypeError):
            continue

        # Skip models with no pricing
        if prompt_price <= 0 and completion_price <= 0:
            continue

        # Convert per-token to per-1M-tokens
        or_input = round(prompt_price * 1_000_000, 4) if prompt_price > 0 else None
        or_output = round(completion_price * 1_000_000, 4) if completion_price > 0 else None

        creator = _extract_creator(model_id)
        context_length = m.get("context_length")
        or_ctx = round(context_length / 1000) if isinstance(context_length, (int, float)) and context_length > 0 else None

        architecture = m.get("architecture", {})
        input_mods: list[str] = architecture.get("input_modalities") or []
        output_mods: list[str] = architecture.get("output_modalities") or []

        slug = _derive_slug(model_id)

        try:
            records.append(
                RawModelRecord(
                    model_id=slug,
                    name=name,
                    creator=creator,
                    openrouter_price_input=or_input,
                    openrouter_price_output=or_output,
                    context_window_k=or_ctx,
                    openrouter_model_id=model_id,
                    input_modalities=input_mods,
                    output_modalities=output_mods,
                    source_pages=["openrouter"],
                )
            )
        except Exception as exc:
            log.debug("Skipping OpenRouter model %s: %s", model_id, exc)

    return records


def _fetch_endpoint_uptime(model_id: str) -> tuple[str, float | None, float | None]:
    """Fetch best uptime across all endpoints for one OpenRouter model ID."""
    url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if not resp.ok:
            return model_id, None, None
        endpoints = resp.json().get("data", {}).get("endpoints", [])
        up30 = [e["uptime_last_30m"] for e in endpoints if e.get("uptime_last_30m") is not None]
        up1d = [e["uptime_last_1d"] for e in endpoints if e.get("uptime_last_1d") is not None]
        return model_id, (max(up30) if up30 else None), (max(up1d) if up1d else None)
    except Exception as exc:
        log.debug("Uptime fetch failed for %s: %s", model_id, exc)
        return model_id, None, None


def _enrich_uptime(records: list[RawModelRecord]) -> None:
    """Fetch uptime stats for all OpenRouter records in parallel (mutates in place)."""
    or_records = [r for r in records if r.openrouter_model_id]
    if not or_records:
        return

    log.info("Fetching uptime for %d OpenRouter models in parallel...", len(or_records))
    id_to_record: dict[str, RawModelRecord] = {r.openrouter_model_id: r for r in or_records}  # type: ignore[index]

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(_fetch_endpoint_uptime, mid): mid for mid in id_to_record}
        for future in as_completed(futures):
            mid, u30, u1d = future.result()
            rec = id_to_record[mid]
            rec.uptime_30m = u30
            rec.uptime_1d = u1d

    fetched = sum(1 for r in or_records if r.uptime_30m is not None)
    log.info("Uptime fetched for %d/%d models", fetched, len(or_records))


def fetch_all_openrouter() -> list[RawModelRecord]:
    """Fetch and parse all OpenRouter models, enriching with uptime in parallel."""
    log.info("Fetching OpenRouter models...")
    raw = fetch_openrouter_models()
    if raw is None:
        log.warning("Failed to fetch OpenRouter models, skipping")
        return []
    records = parse_openrouter_models(raw)
    log.info("OpenRouter: %d models with pricing", len(records))
    _enrich_uptime(records)
    return records
