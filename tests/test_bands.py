"""Unit tests for intelligence-band leaderboards (spec 02).

Thresholds live in src/bands.py (the single source of truth per 00-overview §2:
band edges 25/40, cumulative cheapest-above thresholds [25, 40, 50]).
"""

from __future__ import annotations

import json

import pytest

from src.bands import (
    BAND_MEMBER_LIMIT,
    BANDS,
    INTELLIGENCE_THRESHOLDS,
    band_of,
    compute_bands,
    compute_cheapest_above,
)
from src.models import ArchivePayload, BandDeal, CheapestAbove, MergedModel


def _m(name: str, **kw) -> MergedModel:
    return MergedModel(model_id=name.lower().replace(" ", "-"), name=name, **kw)


# --- 1. band_of boundaries (left-closed / right-open) -------------------------


def test_band_of_boundaries():
    assert band_of(40.0) == "flagship"
    assert band_of(39.9) == "workhorse"
    assert band_of(25.0) == "workhorse"
    assert band_of(24.9) == "budget"
    assert band_of(0.0) == "budget"
    assert band_of(60.0) == "flagship"


def test_band_of_none():
    assert band_of(None) is None


def test_unified_thresholds():
    # 00-overview §2 unifies on [25, 40, 50]; spec 04 imports this constant.
    assert INTELLIGENCE_THRESHOLDS == [25.0, 40.0, 50.0]
    # Internal band ids must not reuse the "frontier" token (size_tier footgun).
    assert [b[0] for b in BANDS] == ["flagship", "workhorse", "budget"]


# --- 2. compute_bands membership ----------------------------------------------


def test_compute_bands_membership_and_eligibility():
    models = [
        _m("Flag1", intelligence_score=45.0, price_blended=2.0, composite_deal_score=0.8),
        _m("Flag2", intelligence_score=40.0, price_blended=1.0, composite_deal_score=0.7),
        _m("Work1", intelligence_score=30.0, price_blended=0.5, composite_deal_score=0.6),
        _m("Bud1", intelligence_score=20.0, price_blended=0.1, composite_deal_score=0.5),
        _m("Bud2", intelligence_score=10.0, price_blended=0.05, composite_deal_score=0.4),
        # Excluded: no composite_deal_score.
        _m("NoComposite", intelligence_score=50.0, price_blended=1.0),
        # Excluded: no intelligence_score.
        _m("NoIntel", price_blended=1.0, composite_deal_score=0.9),
    ]
    bands = {b.id: b for b in compute_bands(models)}
    assert bands["flagship"].count == 2
    assert bands["workhorse"].count == 1
    assert bands["budget"].count == 2
    # Counts sum to the eligible total (5, excluding the two missing-signal ones).
    assert sum(b.count for b in bands.values()) == 5


def test_compute_bands_order():
    bands = compute_bands([])
    assert [b.id for b in bands] == ["flagship", "workhorse", "budget"]


# --- 3. leader = max composite, cheapest = min price (can differ) -------------


def test_leader_vs_cheapest_differ():
    models = [
        _m("HighComposite", intelligence_score=45.0, price_blended=2.0, composite_deal_score=0.9),
        _m("Cheap", intelligence_score=42.0, price_blended=0.1, composite_deal_score=0.3),
    ]
    flagship = next(b for b in compute_bands(models) if b.id == "flagship")
    assert flagship.leader_id == "highcomposite"
    assert flagship.cheapest_id == "cheap"


def test_leader_respects_uptime_gate():
    # Higher composite but failing uptime must not be leader when a passing
    # member exists; it still counts and can be the cheapest.
    models = [
        _m("Flaky", intelligence_score=45.0, price_blended=0.1, composite_deal_score=0.9, uptime_30m=90.0),
        _m("Solid", intelligence_score=44.0, price_blended=2.0, composite_deal_score=0.7, uptime_30m=99.0),
    ]
    flagship = next(b for b in compute_bands(models) if b.id == "flagship")
    assert flagship.leader_id == "solid"        # uptime-gated leader
    assert flagship.cheapest_id == "flaky"      # cheapest ignores the gate
    assert flagship.count == 2


# --- 4. empty band -------------------------------------------------------------


def test_empty_band_returned():
    models = [
        _m("Bud1", intelligence_score=10.0, price_blended=0.1, composite_deal_score=0.4),
    ]
    flagship = next(b for b in compute_bands(models) if b.id == "flagship")
    assert flagship.count == 0
    assert flagship.leader_id is None
    assert flagship.cheapest_id is None
    assert flagship.member_ids == []


# --- 5. compute_cheapest_above is cumulative ----------------------------------


def test_cheapest_above_cumulative():
    # A pricier flagship (intel 41) and a cheaper workhorse (intel 37).
    models = [
        _m("PriceyFlag", intelligence_score=41.0, price_blended=0.5, composite_deal_score=0.6),
        _m("CheapWork", intelligence_score=37.0, price_blended=0.187, composite_deal_score=0.6),
    ]
    ca = {c.threshold: c for c in compute_cheapest_above(models)}
    # >=25: cheaper lower-intel model wins (cumulative, not band-bounded).
    assert ca[25.0].model_id == "cheapwork"
    assert ca[25.0].effective_price == pytest.approx(0.187, abs=1e-4)
    # >=40: only the flagship qualifies.
    assert ca[40.0].model_id == "priceyflag"
    # >=50: nobody qualifies.
    assert ca[50.0].model_id is None
    assert ca[50.0].effective_price is None


def test_cheapest_above_uses_raw_price():
    # Sub-floor price must survive unfloored in cheapest_above (00-overview §1).
    models = [
        _m("SubFloor", intelligence_score=45.0, price_blended=0.02, composite_deal_score=0.6),
    ]
    ca = {c.threshold: c for c in compute_cheapest_above(models)}
    assert ca[40.0].effective_price == pytest.approx(0.02, abs=1e-4)


# --- 6. member_ids capped + ordered by composite desc -------------------------


def test_member_ids_capped_and_ordered():
    models = [
        _m(f"F{i}", intelligence_score=40.0 + i, price_blended=1.0, composite_deal_score=i / 100.0)
        for i in range(1, 12)  # 11 flagship models
    ]
    flagship = next(b for b in compute_bands(models) if b.id == "flagship")
    assert flagship.count == 11
    assert len(flagship.member_ids) == BAND_MEMBER_LIMIT  # capped at 8
    composites = [
        next(m for m in models if m.model_id == mid).composite_deal_score
        for mid in flagship.member_ids
    ]
    assert composites == sorted((c for c in composites if c is not None), reverse=True)
    assert flagship.member_ids[0] == flagship.leader_id


# --- 7. backward-compat: old archive lacks bands ------------------------------


def test_old_archive_loads_with_empty_bands():
    from tests.conftest import REPO_ROOT

    old = REPO_ROOT / "docs" / "archive" / "2026-07-23.json"
    raw = json.loads(old.read_text(encoding="utf-8"))
    assert "bands" not in raw  # sanity: genuinely a pre-spec-02 archive
    payload = ArchivePayload.model_validate(raw)
    assert payload.bands == []
    assert payload.cheapest_above == []


# --- 8. round-trip: payload with bands serializes and reloads -----------------


def test_bands_round_trip():
    band = BandDeal(
        id="flagship",
        label="Frontier-class",
        min_intelligence=40.0,
        max_intelligence=None,
        count=2,
        leader_id="a",
        cheapest_id="b",
        member_ids=["a", "b"],
    )
    ca = CheapestAbove(threshold=40.0, model_id="a", effective_price=0.29)
    payload = ArchivePayload(
        date="2026-07-24",
        generated_at="2026-07-24 00:00 UTC",
        insights="",
        total_models=0,
        models=[],
        bands=[band],
        cheapest_above=[ca],
    )
    dumped = json.loads(json.dumps(payload.model_dump(mode="json")))
    reloaded = ArchivePayload.model_validate(dumped)
    assert reloaded.bands == [band]
    assert reloaded.cheapest_above == [ca]


# --- 9. real-data smoke -------------------------------------------------------


def test_real_data_bands():
    from tests.conftest import REPO_ROOT, reconstruct_models
    from src.scorer import annotate_sizes, rank_models

    models = reconstruct_models(REPO_ROOT / "docs" / "data" / "latest.json")
    annotate_sizes(models)
    ranked = rank_models(models)

    bands = compute_bands(ranked)
    assert [b.id for b in bands] == ["flagship", "workhorse", "budget"]
    assert all(b.count > 0 for b in bands)  # balanced, no empty band on real data

    by_id = {m.model_id: m for m in ranked}
    flagship = next(b for b in bands if b.id == "flagship")
    leader = by_id[flagship.leader_id]
    assert leader.intelligence_score is not None and leader.intelligence_score >= 40.0

    ca = {c.threshold: c for c in compute_cheapest_above(ranked)}
    assert set(ca) == {25.0, 40.0, 50.0}
    # The >=40 answer must itself be a >=40-intelligence model.
    top = by_id[ca[40.0].model_id]
    assert top.intelligence_score is not None and top.intelligence_score >= 40.0
