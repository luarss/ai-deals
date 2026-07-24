"""Write JSON archives and maintain the date manifest."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from src.models import ArchivePayload, BandDeal, Changelog, CheapestAbove, MergedModel
from src.timeseries import build_timeseries

log = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).parent.parent / "docs"


def write_archives(
    models: list[MergedModel],
    insights: str,
    generated_at: datetime,
    top_models: list[MergedModel] | None = None,
    arena_top_models: list[MergedModel] | None = None,
    bands: list[BandDeal] | None = None,
    cheapest_above: list[CheapestAbove] | None = None,
    changelog: Changelog | None = None,
) -> None:
    date_str = generated_at.strftime("%Y-%m-%d")
    generated_str = generated_at.strftime("%Y-%m-%d %H:%M UTC")

    hero = top_models[0] if top_models else None
    runners_up = top_models[1:4] if top_models else []
    arena_hero = arena_top_models[0] if arena_top_models else None
    arena_runners_up = arena_top_models[1:4] if arena_top_models else []

    # Explicit frontier arrays for the chart, sorted by price ascending for the
    # step-line. Empty when annotate_frontier found no candidates.
    frontier_intel = sorted(
        (m for m in models if m.is_frontier_intel),
        key=lambda m: m._raw_blended_price() or 0,
    )
    frontier_arena = sorted(
        (m for m in models if m.is_frontier_arena),
        key=lambda m: m._raw_blended_price() or 0,
    )

    payload = ArchivePayload(
        date=date_str,
        generated_at=generated_str,
        insights=insights,
        total_models=len(models),
        models=models,
        hero=hero,
        runners_up=runners_up,
        arena_hero=arena_hero,
        arena_runners_up=arena_runners_up,
        frontier_intel=frontier_intel,
        frontier_arena=frontier_arena,
        bands=bands or [],
        cheapest_above=cheapest_above or [],
        changelog=changelog,
    )

    _write_json(DOCS_DIR / "data" / "latest.json", payload)
    _write_json(DOCS_DIR / "archive" / f"{date_str}.json", payload)
    _update_manifest(date_str)

    log.info("Archives written for %s (%d models)", date_str, len(models))

    # Rebuild the longitudinal timeseries LAST, so today's freshly-written
    # <date>.json is included in the rebuild (spec 04 §5.4 — ordering matters).
    write_timeseries(generated_at)


def write_timeseries(generated_at: datetime) -> None:
    """Full deterministic rebuild of docs/data/timeseries.json (spec 04)."""
    ts = build_timeseries(DOCS_DIR / "archive", generated_at)
    path = DOCS_DIR / "data" / "timeseries.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ts, ensure_ascii=False), encoding="utf-8")
    log.info(
        "Timeseries written: %d days, %d model series",
        len(ts["days"]),
        len(ts["models"]),
    )


def _write_json(path: Path, payload: ArchivePayload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = payload.model_dump(mode="json")
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _update_manifest(date_str: str) -> None:
    manifest_path = DOCS_DIR / "archive" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dates = set(manifest.get("dates", []))
    else:
        dates = set()

    dates.add(date_str)
    manifest = {"dates": sorted(dates, reverse=True)}
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
