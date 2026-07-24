"""Compute a day-over-day changelog from the current models vs. the previous archive.

The differ joins today's merged+scored models against the most recent previous
archive on ``model_id`` (minted by the merger, stable across days per spec 03).
It emits a structured :class:`~src.models.Changelog`: new/removed models, price
and intelligence changes, low-priority speed changes, frontier-deal rank moves,
and Pareto-frontier entries/exits (spec 01 fields).

Two safety mechanisms shape the output:

* **Outage defense** — a source that vanishes overnight (spec 03 §Outage
  detection) would otherwise register hundreds of fake removals; per-source and
  global caps suppress those.
* **scoring_version guard** — the 62 pre-v2 archives lack the v2 knee/frontier
  fields, so comparing v1→v2 score-derived events (rank/frontier) would flood
  the changelog. When the previous archive's ``scoring_version`` differs from
  today's, those events are suppressed. Raw-field events (new/removed/price/
  intelligence/speed) are always valid.

``compute_changelog`` never raises: any internal error is logged and an empty
``Changelog(first_run=True)`` is returned so a diff failure never aborts the
pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from src.models import ChangeEvent, Changelog, MergedModel
from src.pricing import effective_price
from src.scorer import SCORING_VERSION

log = logging.getLogger(__name__)
DOCS_DIR = Path(__file__).parent.parent / "docs"

# --- thresholds (see spec 03 §Threshold calibration) ---
PRICE_REL_THRESHOLD = 0.05      # 5% — floor vs rounding; prices don't jitter
INTEL_ABS_THRESHOLD = 0.5       # points on 0-60 scale
SPEED_REL_THRESHOLD = 0.25      # 25% — above p95(18%) speed jitter; low priority
SPEED_EVENT_CAP = 15            # keep only the 15 biggest speed moves
RANK_MOVE_MIN = 3               # positions, within top 20
RANK_TOP_N = 10                 # entering/leaving this window is always an event
RANK_WINDOW = 20                # only consider the top-20 of the deal list
REMOVAL_SANITY_CAP_FRAC = 0.05  # >5% of prev N (or 20) => outage
REMOVAL_SANITY_CAP_MIN = 20
SOURCE_DEGRADE_MIN = 50         # a source w/ >=50 models yesterday, 0 today => degraded


def load_previous_archive(
    today: str, archive_dir: Path | None = None
) -> Optional[dict]:
    """Return the parsed archive for the most recent date strictly before ``today``.

    Reads ``manifest.json`` and walks dates newest-first, transparently skipping
    gaps (e.g. the 06-20 gap → previous-of-06-21 is 06-19). Returns None when no
    earlier archive exists (first run).
    """
    base = archive_dir or (DOCS_DIR / "archive")
    manifest = base / "manifest.json"
    if not manifest.exists():
        return None
    try:
        dates = sorted(
            json.loads(manifest.read_text(encoding="utf-8")).get("dates", []),
            reverse=True,
        )
    except (ValueError, OSError) as exc:
        log.warning("Could not read manifest for diff (%s): %s", type(exc).__name__, exc)
        return None
    for d in dates:
        if d < today:
            p = base / f"{d}.json"
            if p.exists():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except (ValueError, OSError) as exc:
                    log.warning("Could not read %s for diff: %s", p, exc)
                    return None
    return None


def _source_counts(models: list[dict]) -> dict[str, int]:
    """Count how many models each source_page contributed to."""
    counts: dict[str, int] = {}
    for m in models:
        for s in m.get("source_pages", []) or []:
            counts[s] = counts.get(s, 0) + 1
    return counts


def _frontier_deal_order(models: list[dict]) -> list[str]:
    """model_ids of the frontier deal list, hero first.

    Prefers spec 01's ``frontier_rank_intel`` (the knee ordering shown on the
    site). Falls back to the legacy ``size_tier == "frontier"`` + composite order
    for pre-frontier (v1) archives so the function is well-defined on both.
    """
    ranked = [m for m in models if m.get("frontier_rank_intel") is not None]
    if ranked:
        ranked.sort(key=lambda m: m["frontier_rank_intel"])
        return [m["model_id"] for m in ranked]
    # Legacy fallback (v1 archives): size_tier == "frontier" by composite desc.
    fr = [
        m
        for m in models
        if m.get("size_tier") == "frontier" and m.get("composite_deal_score") is not None
    ]
    fr.sort(key=lambda m: -(m.get("composite_deal_score") or 0))
    return [m["model_id"] for m in fr]


def compute_changelog(
    today_models: list[MergedModel],
    prev_archive: Optional[dict],
    today_scoring_version: int = SCORING_VERSION,
) -> Changelog:
    """Diff today's ranked models against the previous archive. Never raises."""
    try:
        return _compute_changelog(today_models, prev_archive, today_scoring_version)
    except Exception as exc:  # diff must never abort the pipeline
        log.exception("compute_changelog failed (%s); emitting empty changelog", exc)
        return Changelog(first_run=True)


def _compute_changelog(
    today_models: list[MergedModel],
    prev_archive: Optional[dict],
    today_scoring_version: int,
) -> Changelog:
    today = [m.model_dump(mode="json") for m in today_models]
    if prev_archive is None:
        return Changelog(compared_to=None, first_run=True)

    prev = prev_archive.get("models", [])
    prev_by_id = {m["model_id"]: m for m in prev}
    today_by_id = {m["model_id"]: m for m in today}
    prev_ids, today_ids = set(prev_by_id), set(today_by_id)

    cl = Changelog(compared_to=prev_archive.get("date"))

    # scoring_version guard: v1 archives lack the field (None). When it differs
    # from today's, suppress score-derived events (rank/frontier) — the raw
    # fields (price/intel/new/removed/speed) are still comparable.
    prev_version = prev_archive.get("scoring_version")
    cl.scoring_changed = prev_version != today_scoring_version

    # --- outage / source-degrade detection FIRST ---
    prev_src = _source_counts(prev)
    today_src = _source_counts(today)
    degraded_sources = {
        s
        for s, n in prev_src.items()
        if n >= SOURCE_DEGRADE_MIN and today_src.get(s, 0) == 0
    }

    # --- NEW models ---
    for mid in sorted(today_ids - prev_ids):
        m = today_by_id[mid]
        cl.new_models.append(
            ChangeEvent(
                model_id=mid,
                name=m["name"],
                kind="new",
                intelligence_score=m.get("intelligence_score"),
                price=effective_price(m),
                composite_deal_score=m.get("composite_deal_score"),
            )
        )

    # --- REMOVED models (two-layer outage guard) ---
    # Layer 1 (per-source): drop removals fully explained by a degraded source —
    # the model isn't gone, its only source failed to fetch. Layer 2 (global
    # cap) then applies to the *remaining* real removals: an unexplained mass
    # drop is a data-quality failure, not news, so the list is suppressed and
    # the day flagged "degraded". A drop fully explained by a source outage is
    # flagged "partial" (real removals under the cap are still reported).
    removed_ids = sorted(prev_ids - today_ids)
    real_removed = [
        mid
        for mid in removed_ids
        if not (
            (src := set(prev_by_id[mid].get("source_pages", []) or []))
            and src.issubset(degraded_sources)
        )
    ]
    over_cap = len(real_removed) > max(
        REMOVAL_SANITY_CAP_MIN, int(len(prev) * REMOVAL_SANITY_CAP_FRAC)
    )
    if over_cap:
        cl.data_quality = "degraded"  # suppress removals entirely
    else:
        for mid in real_removed:
            m = prev_by_id[mid]
            cl.removed_models.append(
                ChangeEvent(
                    model_id=mid,
                    name=m["name"],
                    kind="removed",
                    intelligence_score=m.get("intelligence_score"),
                    price=effective_price(m),
                )
            )
        if degraded_sources:
            cl.data_quality = "partial"
            cl.degraded_sources = sorted(degraded_sources)

    # --- field changes on COMMON models ---
    for mid in sorted(prev_ids & today_ids):
        p, t = prev_by_id[mid], today_by_id[mid]
        name = t["name"]

        # price (raw effective price; always valid across scoring versions)
        pp, tp = effective_price(p), effective_price(t)
        if pp and tp and pp > 0 and abs(tp - pp) / pp >= PRICE_REL_THRESHOLD:
            cl.price_changes.append(
                ChangeEvent(
                    model_id=mid,
                    name=name,
                    kind="price",
                    old=round(pp, 6),
                    new=round(tp, 6),
                    pct=round((tp - pp) / pp * 100, 1),
                    direction="cut" if tp < pp else "hike",
                )
            )

        # intelligence (raw field)
        pi, ti = p.get("intelligence_score"), t.get("intelligence_score")
        if pi is not None and ti is not None and abs(ti - pi) >= INTEL_ABS_THRESHOLD:
            cl.intelligence_changes.append(
                ChangeEvent(
                    model_id=mid,
                    name=name,
                    kind="intelligence",
                    old=pi,
                    new=ti,
                    delta=round(ti - pi, 1),
                    direction="up" if ti > pi else "down",
                )
            )

        # speed (raw field; low priority, high threshold)
        ps, ts = p.get("output_speed_tps"), t.get("output_speed_tps")
        if ps and ts and ps > 0 and abs(ts - ps) / ps >= SPEED_REL_THRESHOLD:
            cl.speed_changes.append(
                ChangeEvent(
                    model_id=mid,
                    name=name,
                    kind="speed",
                    old=round(ps),
                    new=round(ts),
                    pct=round((ts - ps) / ps * 100, 1),
                    direction="up" if ts > ps else "down",
                )
            )

    # --- score-derived events (rank moves + frontier entries/exits) ---
    # Suppressed when the previous archive used a different scoring_version.
    if not cl.scoring_changed:
        _compute_rank_changes(cl, prev, today, prev_by_id, today_by_id)
        _compute_frontier_changes(cl, prev_ids & today_ids, prev_by_id, today_by_id)

    _sort_events(cl)

    cl.total_events = (
        len(cl.new_models)
        + len(cl.removed_models)
        + len(cl.price_changes)
        + len(cl.intelligence_changes)
        + len(cl.speed_changes)
        + len(cl.rank_changes)
        + len(cl.frontier_changes)
    )
    return cl


def _compute_rank_changes(
    cl: Changelog,
    prev: list[dict],
    today: list[dict],
    prev_by_id: dict[str, Any],
    today_by_id: dict[str, Any],
) -> None:
    """Emit rank events: top-10 entries/exits and >=3-position moves in top 20."""
    prev_order = _frontier_deal_order(prev)
    today_order = _frontier_deal_order(today)
    prev_rank = {mid: i for i, mid in enumerate(prev_order)}
    today_rank = {mid: i for i, mid in enumerate(today_order)}

    seen: set[str] = set()
    for new_i, mid in enumerate(today_order[:RANK_WINDOW]):
        if mid not in prev_rank:
            continue
        old_i = prev_rank[mid]
        entered_top = old_i >= RANK_TOP_N and new_i < RANK_TOP_N
        big_move = abs(new_i - old_i) >= RANK_MOVE_MIN
        if entered_top or big_move:
            seen.add(mid)
            cl.rank_changes.append(
                ChangeEvent(
                    model_id=mid,
                    name=today_by_id[mid]["name"],
                    kind="rank",
                    old_rank=old_i + 1,
                    new_rank=new_i + 1,
                    direction="up" if new_i < old_i else "down",
                    entered_top=entered_top,
                )
            )

    # Models that LEFT the top 10 (were in prev top 10, now >10 or gone).
    for mid in prev_order[:RANK_TOP_N]:
        if mid in seen:
            continue
        new_i = today_rank.get(mid)
        if new_i is None or new_i >= RANK_TOP_N:
            cl.rank_changes.append(
                ChangeEvent(
                    model_id=mid,
                    name=prev_by_id[mid]["name"],
                    kind="rank",
                    old_rank=prev_rank[mid] + 1,
                    new_rank=(new_i + 1) if new_i is not None else None,
                    direction="down",
                    left_top=True,
                )
            )


def _compute_frontier_changes(
    cl: Changelog,
    common_ids: set[str],
    prev_by_id: dict[str, Any],
    today_by_id: dict[str, Any],
) -> None:
    """Emit "entered/left the intelligence frontier" events (spec 01 flag diff)."""
    for mid in sorted(common_ids):
        was = bool(prev_by_id[mid].get("is_frontier_intel"))
        now = bool(today_by_id[mid].get("is_frontier_intel"))
        if was == now:
            continue
        t = today_by_id[mid]
        cl.frontier_changes.append(
            ChangeEvent(
                model_id=mid,
                name=t["name"],
                kind="frontier",
                direction="entered" if now else "left",
                intelligence_score=t.get("intelligence_score"),
                price=effective_price(t),
            )
        )


def _sort_events(cl: Changelog) -> None:
    """Order each event list for deterministic, most-newsworthy-first output."""
    cl.price_changes.sort(key=lambda e: -abs(e.pct or 0))
    cl.intelligence_changes.sort(key=lambda e: -abs(e.delta or 0))
    cl.new_models.sort(key=lambda e: -(e.composite_deal_score or 0))
    cl.rank_changes.sort(
        key=lambda e: -abs((e.new_rank or 99) - (e.old_rank or 99))
    )
    cl.frontier_changes.sort(key=lambda e: (e.direction != "entered", e.name))
    cl.speed_changes.sort(key=lambda e: -abs(e.pct or 0))
    del cl.speed_changes[SPEED_EVENT_CAP:]
