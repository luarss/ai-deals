"""Orchestrator: scrape → merge → score → insights → render → write."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.insights import generate_insights
from src.merger import merge_records
from src.renderer import render
from src.scorer import rank_models
from src.scraper import fetch_all_pages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_PATH = Path("docs/index.html")


def main() -> None:
    log.info("=== AI Deals pipeline starting ===")

    # 1. Scrape
    log.info("Step 1/5: Fetching pages from artificialanalysis.ai")
    raw_records = fetch_all_pages()
    if not raw_records:
        log.error("No records scraped — aborting. Check network or site structure.")
        sys.exit(1)
    log.info("Scraped %d raw records", len(raw_records))

    # 2. Merge & deduplicate
    log.info("Step 2/5: Merging and deduplicating")
    merged = merge_records(raw_records)
    if not merged:
        log.error("Merge produced 0 models — aborting.")
        sys.exit(1)
    log.info("Unique models after merge: %d", len(merged))

    # 3. Score & rank
    log.info("Step 3/5: Scoring and ranking")
    ranked = rank_models(merged)
    top_models = [m for m in ranked if m.composite_deal_score is not None]
    log.info(
        "Top deal: %s (score=%.2f)",
        ranked[0].name if ranked else "none",
        ranked[0].composite_deal_score or 0 if ranked else 0,
    )

    # 4. AI insights
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if api_key:
        log.info("Step 4/5: Generating AI insights via DeepSeek")
        insights = generate_insights(top_models[:10], api_key)
    else:
        log.warning("Step 4/5: DEEPSEEK_API_KEY not set — skipping AI insights")
        insights = ""

    # 5. Render HTML
    log.info("Step 5/5: Rendering HTML")
    html = render(
        top_models=top_models,
        all_models=ranked,
        insights_text=insights,
        generated_at=datetime.now(timezone.utc),
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    log.info("Written to %s (%s bytes)", OUTPUT_PATH, f"{len(html):,}")
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
