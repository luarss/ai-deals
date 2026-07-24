"""Longitudinal time-series builder (spec 04).

Deterministically rebuilds ``docs/data/timeseries.json`` from every archive in
``docs/archive/`` (``20*.json`` glob excludes ``manifest.json``). The daily
pipeline calls this via ``src.writer.write_timeseries`` after the day's archive
is written; ``scripts/build_timeseries.py`` runs the one-time backfill.

Design constraints (see plans/04-trends.md):
  * RAW metrics only — never composite/day-relative scores (§2.1).
  * Effective price via the shared ladder in ``src.pricing`` (§2.2), read from
    plain dicts because archives are heterogeneous JSON (§2.3 schema drift).
  * Cumulative "at or above T" intelligence bands on the SAME thresholds the
    site's bands use (``src.bands.INTELLIGENCE_THRESHOLDS``) so "cheapest >=40"
    matches everywhere (00-overview §2).
  * Full rebuild every run — idempotent and self-healing (§2.6).
"""

from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from src.bands import INTELLIGENCE_THRESHOLDS
from src.pricing import effective_price

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MIN_DAYS = 14  # per-model series inclusion cutoff (§2.4)
WINDOW_DAYS = 180  # most-recent points kept per model series (§2.5)


def _threshold_key(t: float) -> str:
    """Stable JSON band key: '25' not '25.0' (matches the site's '>=T' chips)."""
    return str(int(t)) if float(t).is_integer() else str(t)


def _day_aggregate(
    date: str, pairs: list[tuple[float, float]], thresholds: list[float]
) -> dict[str, Any]:
    """Per-day cumulative band aggregates from (intelligence, price) pairs.

    For each threshold T: cheapest (min) and median price among models with
    ``intelligence >= T``, plus the qualifying count. Empty band -> cheapest and
    median are ``None`` with ``count == 0`` (never crashes, never interpolates).
    """
    bands: dict[str, Any] = {}
    for t in thresholds:
        prices = [p for (i, p) in pairs if i >= t]
        if prices:
            bands[_threshold_key(t)] = {
                "cheapest": round(min(prices), 4),
                "median": round(statistics.median(prices), 4),
                "count": len(prices),
            }
        else:
            bands[_threshold_key(t)] = {"cheapest": None, "median": None, "count": 0}
    return {"date": date, "priced_models": len(pairs), "bands": bands}


def _select_series(
    hist: dict[str, list[dict[str, Any]]],
    meta: dict[str, dict[str, Any]],
    min_days: int,
    window_days: int,
) -> dict[str, Any]:
    """Filter/trim per-model series: drop <min_days history, window to the tail.

    Keys are emitted in sorted model_id order for deterministic output.
    """
    out: dict[str, Any] = {}
    for mid in sorted(hist):
        points = sorted(hist[mid], key=lambda p: p["d"])
        if len(points) < min_days:
            continue
        info = meta.get(mid, {})
        out[mid] = {
            "name": info.get("name"),
            "creator": info.get("creator"),
            "points": points[-window_days:],
        }
    return out


def build_timeseries(archive_dir: Path, generated_at: datetime) -> dict[str, Any]:
    """Build the full timeseries payload from all archives on disk.

    Pure aside from reading archive files; one malformed archive is logged and
    skipped rather than breaking the whole build.
    """
    files = sorted(Path(archive_dir).glob("20*.json"))  # excludes manifest.json
    days: list[dict[str, Any]] = []
    hist: dict[str, list[dict[str, Any]]] = {}  # model_id -> points
    meta: dict[str, dict[str, Any]] = {}  # model_id -> {name, creator}

    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — one bad file must not kill the build
            log.warning("Skipping malformed archive %s: %s", f.name, exc)
            continue

        date = payload.get("date")
        if not date:
            log.warning("Archive %s has no date field; skipping", f.name)
            continue

        pairs: list[tuple[float, float]] = []  # (intelligence, price) this day
        for m in payload.get("models", []):
            intel = m.get("intelligence_score")
            price = effective_price(m)
            if intel is None or not price or price <= 0:
                continue
            pairs.append((intel, price))
            mid = m.get("model_id")  # absent in the 4 oldest archives -> no series
            if mid:
                hist.setdefault(mid, []).append(
                    {"d": date, "i": intel, "p": round(price, 4)}
                )
                meta[mid] = {"name": m.get("name"), "creator": m.get("creator")}

        days.append(_day_aggregate(date, pairs, INTELLIGENCE_THRESHOLDS))

    days.sort(key=lambda d: d["date"])
    models = _select_series(hist, meta, MIN_DAYS, WINDOW_DAYS)

    thresholds_out = [
        int(t) if float(t).is_integer() else t for t in INTELLIGENCE_THRESHOLDS
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at.strftime("%Y-%m-%d %H:%M UTC"),
        "thresholds": thresholds_out,
        "days": days,
        "models": models,
    }
