"""Intelligence bands — the single source of truth for intelligence thresholds.

Per 00-overview §2 this module unifies the thresholds that spec 02 (bands) and
spec 04 (trends) both consume, so the site's "cheapest ≥40" number and the trend
line's "cheapest ≥40" line always share one constant.

Band membership is keyed on ``intelligence_score`` ONLY — never ``size_tier``.
Boundaries are left-closed / right-open on intelligence:
    budget    = [0, 25)
    workhorse = [25, 40)
    flagship  = [40, +inf)

The stable JSON ``id`` for the top band is ``flagship`` (not ``frontier``) so it
stays orthogonal to ``size_tier == "frontier"``; the display label is
"Frontier-class".
"""

from __future__ import annotations

from src.models import BandDeal, CheapestAbove, MergedModel

# Band definitions: (id, display_label, min_inclusive, max_exclusive_or_None).
# flagship -> workhorse -> budget order is preserved through the pipeline so the
# frontend can render a stable set of sections.
BANDS: list[tuple[str, str, float, float | None]] = [
    ("flagship", "Frontier-class", 40.0, None),
    ("workhorse", "Workhorse", 25.0, 40.0),
    ("budget", "Budget", 0.0, 25.0),
]

# Cumulative "cheapest at or above X" thresholds. Unified on [25, 40, 50] per
# 00-overview §2 (25/40 are the band floors; 50 is spec 04's frontier-class line).
# Spec 04 imports THIS constant so both features answer the same question.
INTELLIGENCE_THRESHOLDS: list[float] = [25.0, 40.0, 50.0]

BAND_MEMBER_LIMIT = 8  # top-N members surfaced per band for the UI


def band_of(intel: float | None) -> str | None:
    """Return the band id for an intelligence score, or None if unscored."""
    if intel is None:
        return None
    for band_id, _label, lo, hi in BANDS:
        if intel >= lo and (hi is None or intel < hi):
            return band_id
    return None


def _eligible(models: list[MergedModel]) -> list[MergedModel]:
    """Band-eligible pool: has both intelligence_score and composite_deal_score."""
    return [
        m
        for m in models
        if m.intelligence_score is not None and m.composite_deal_score is not None
    ]


def compute_bands(models: list[MergedModel]) -> list[BandDeal]:
    """Build per-band leaderboards from the AA-intelligence pool.

    Eligible = has intelligence_score AND composite_deal_score. Returns bands in
    flagship->workhorse->budget order. Empty bands are still returned (count=0)
    so the frontend can render a stable set of sections.

    - ``leader`` = highest composite_deal_score in the band, among uptime-passing
      members (reuses the spec 05 uptime gate). Falls back to the best composite
      overall if no member passes the gate, so a band is never leaderless.
    - ``cheapest`` = lowest RAW effective price in the band (unfloored; see
      00-overview §1). Displayed/compared prices are never floored.
    """
    from src.scorer import passes_uptime_gate

    eligible = _eligible(models)

    result: list[BandDeal] = []
    for band_id, label, lo, hi in BANDS:
        members = [
            m
            for m in eligible
            if m.intelligence_score is not None
            and m.intelligence_score >= lo
            and (hi is None or m.intelligence_score < hi)
        ]

        # Leader: best composite among uptime-passing members; fall back to the
        # best composite overall so the band is never leaderless.
        gated = [m for m in members if passes_uptime_gate(m)]
        leader_pool = gated or members
        by_composite = sorted(leader_pool, key=lambda m: -(m.composite_deal_score or 0.0))
        leader_id = by_composite[0].model_id if by_composite else None

        # member_ids: top-N by composite across ALL members (leader first).
        members_ranked = sorted(members, key=lambda m: -(m.composite_deal_score or 0.0))
        member_ids = [m.model_id for m in members_ranked[:BAND_MEMBER_LIMIT]]

        # Cheapest: lowest raw effective price in the band (missing prices last).
        priced = [m for m in members if m.effective_price() is not None]
        cheapest_id = None
        if priced:
            cheapest = min(priced, key=lambda m: m.effective_price() or float("inf"))
            cheapest_id = cheapest.model_id

        result.append(
            BandDeal(
                id=band_id,
                label=label,
                min_intelligence=lo,
                max_intelligence=hi,
                count=len(members),
                leader_id=leader_id,
                cheapest_id=cheapest_id,
                member_ids=member_ids,
            )
        )
    return result


def compute_cheapest_above(models: list[MergedModel]) -> list[CheapestAbove]:
    """For each threshold, the single cheapest model with intelligence >= threshold.

    Cumulative (not band-bounded): a higher-band model can legitimately be the
    cheapest answer for a lower threshold. Uses RAW effective price.
    """
    pool = _eligible(models)
    out: list[CheapestAbove] = []
    for t in INTELLIGENCE_THRESHOLDS:
        cands = [
            m
            for m in pool
            if m.intelligence_score is not None
            and m.intelligence_score >= t
            and m.effective_price() is not None
        ]
        if cands:
            best = min(cands, key=lambda m: m.effective_price() or float("inf"))
            price = best.effective_price()
            out.append(
                CheapestAbove(
                    threshold=t,
                    model_id=best.model_id,
                    effective_price=round(price, 4) if price is not None else None,
                )
            )
        else:
            out.append(CheapestAbove(threshold=t))
    return out
