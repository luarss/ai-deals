"""Fetch model pricing from OpenRouter API.

API: https://openrouter.ai/api/v1/models (no auth required)
Returns per-model pricing (per-token), context length, model ID.
"""

from __future__ import annotations

import logging
import re
import time
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
                    source_pages=["openrouter"],
                )
            )
        except Exception as exc:
            log.debug("Skipping OpenRouter model %s: %s", model_id, exc)

    return records


def fetch_all_openrouter() -> list[RawModelRecord]:
    """Fetch and parse all OpenRouter models."""
    log.info("Fetching OpenRouter models...")
    raw = fetch_openrouter_models()
    if raw is None:
        log.warning("Failed to fetch OpenRouter models, skipping")
        return []
    records = parse_openrouter_models(raw)
    log.info("OpenRouter: %d models with pricing", len(records))
    return records
