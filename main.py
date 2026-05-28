"""Orchestrator: scrape → merge → score → insights → write archives → render shell."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.arena_fetcher import fetch_all_arena
from src.insights import generate_insights
from src.merger import merge_records
from src.openrouter_fetcher import fetch_all_openrouter
from src.renderer import render_app_shell
from src.scorer import annotate_sizes, compute_top_models, rank_models
from src.scraper import fetch_all_pages
from src.writer import write_archives

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_PATH = Path("docs/index.html")


def main() -> None:
    log.info("=== AI Deals pipeline starting ===")

    # 1. Scrape all sources
    log.info("Step 1/6: Fetching from artificialanalysis.ai")
    aa_records = fetch_all_pages()
    if not aa_records:
        log.error("No Artificial Analysis records scraped — aborting.")
        sys.exit(1)
    log.info("Artificial Analysis: %d raw records", len(aa_records))

    # 2. Arena AI
    log.info("Step 2/6: Fetching from Arena AI")
    try:
        arena_records = fetch_all_arena()
    except Exception as exc:
        log.warning("Arena AI fetch failed (%s), continuing without it", exc)
        arena_records = []
    log.info("Arena AI: %d raw records", len(arena_records))

    # 3. OpenRouter pricing
    log.info("Step 3/6: Fetching from OpenRouter")
    try:
        or_records = fetch_all_openrouter()
    except Exception as exc:
        log.warning("OpenRouter fetch failed (%s), continuing without it", exc)
        or_records = []

    all_records = aa_records + arena_records + or_records
    log.info("Total raw records: %d", len(all_records))

    # 4. Merge & deduplicate
    log.info("Step 4/6: Merging and deduplicating")
    merged = merge_records(all_records)
    if not merged:
        log.error("Merge produced 0 models — aborting.")
        sys.exit(1)
    log.info("Unique models after merge: %d", len(merged))

    # 5. Score & rank
    log.info("Step 5/6: Scoring and ranking")
    annotate_sizes(merged)
    ranked = rank_models(merged)

    top_models = compute_top_models(ranked, "aa")
    arena_top = compute_top_models(ranked, "arena")
    log.info(
        "AA top deal: %s (score=%.2f) | Arena top: %s (score=%.2f)",
        top_models[0].name if top_models else "none",
        top_models[0].composite_deal_score or 0 if top_models else 0,
        arena_top[0].name if arena_top else "none",
        arena_top[0].arena_composite_score or 0 if arena_top else 0,
    )

    # 6. AI insights
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if api_key:
        log.info("Step 6/6: Generating AI insights via DeepSeek")
        insights = generate_insights(top_models[:10], api_key)
    else:
        log.warning("Step 6/6: DEEPSEEK_API_KEY not set — skipping AI insights")
        insights = ""

    # 7. Write archives + app shell
    log.info("Writing JSON archives and HTML app shell")
    generated_at = datetime.now(timezone.utc)
    write_archives(ranked, insights, generated_at, top_models=top_models, arena_top_models=arena_top)

    html = render_app_shell()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    log.info("Written to %s (%s bytes)", OUTPUT_PATH, f"{len(html):,}")
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
