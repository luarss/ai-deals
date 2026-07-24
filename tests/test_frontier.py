"""Unit tests for the Pareto-frontier deal ranking (spec 01)."""

from __future__ import annotations

import pytest

from src.models import MergedModel
from src.scorer import (
    _collapse_near_duplicates,
    _DUP_Q_INTEL,
    _frontier_price,
    _knee_scores,
    _pareto_frontier,
    annotate_frontier,
    annotate_sizes,
    compute_top_models,
    rank_models,
)


def _m(name: str, **kw) -> MergedModel:
    return MergedModel(model_id=name.lower().replace(" ", "-"), name=name, **kw)


# --- 1. _pareto_frontier ------------------------------------------------------


def test_pareto_excludes_dominated():
    a, b, c = _m("A"), _m("B"), _m("C")
    # B (0.5, 40) dominates C (1.0, 30): cheaper AND smarter.
    cands = [(a, 1.0, 50.0), (b, 0.5, 40.0), (c, 1.0, 30.0)]
    names = {t[0].name for t in _pareto_frontier(cands)}
    assert names == {"A", "B"}
    assert "C" not in names


def test_pareto_keeps_ties():
    a, b = _m("A"), _m("B")
    # Identical points: neither strictly dominates → both kept.
    front = _pareto_frontier([(a, 1.0, 40.0), (b, 1.0, 40.0)])
    assert {t[0].name for t in front} == {"A", "B"}


# --- 2. Regression: real-data hero is DeepSeek V4 Flash, NOT Ling -------------


def test_real_frontier_hero_is_deepseek_not_ling():
    from tests.conftest import REPO_ROOT, reconstruct_models

    models = reconstruct_models(REPO_ROOT / "docs" / "data" / "latest.json")
    annotate_sizes(models)
    ranked = rank_models(models)
    annotate_frontier(ranked)

    front = [m for m in models if m.is_frontier_intel]
    assert len(front) == 12, f"expected 12-point intel frontier, got {len(front)}"

    hero = next(m for m in models if m.frontier_rank_intel == 1)
    assert "DeepSeek V4 Flash" in hero.name, f"hero was {hero.name}"

    ling = next(m for m in models if "Ling-2.6-flash" in m.name)
    # Ling stays on the frontier (it's an endpoint) but is no longer the knee.
    assert ling.frontier_rank_intel != 1
    assert ling.frontier_score_intel == pytest.approx(0.0, abs=1e-6)


# --- 3. _knee_scores: endpoints ~0, knee maximal ------------------------------


def test_knee_scores_endpoints_cancel():
    cheap, knee, exp = _m("Cheap"), _m("Knee"), _m("Exp")
    front = [(cheap, 0.02, 14.0), (knee, 0.13, 40.0), (exp, 23.0, 60.0)]
    scores = _knee_scores(front)
    assert scores[id(cheap)] == pytest.approx(0.0, abs=1e-6)
    assert scores[id(exp)] == pytest.approx(0.0, abs=1e-6)
    assert scores[id(knee)] > 0
    assert scores[id(knee)] == max(scores.values())


def test_knee_scores_single_point():
    only = _m("Only")
    scores = _knee_scores([(only, 1.0, 40.0)])
    # min==max on both axes → 0 - 0 = 0, no division error.
    assert scores[id(only)] == 0.0


# --- 4. _collapse_near_duplicates ---------------------------------------------


def test_collapse_near_duplicates_keeps_cheaper():
    a = _m("A", composite_deal_score=0.5)
    b = _m("B", composite_deal_score=0.9)
    far = _m("Far", composite_deal_score=0.1)
    # a and b are within ~2.3% price (log diff ~0.01) and 0.2 intel apart.
    p_a, p_b = 1.00, 1.00 * 10 ** 0.01
    front = [(a, p_a, 40.0), (b, p_b, 40.2), (far, 10.0, 55.0)]
    kept = _collapse_near_duplicates(front, _DUP_Q_INTEL, "composite_deal_score")
    names = {t[0].name for t in kept}
    assert names == {"A", "Far"}  # cheaper 'A' wins the cluster; 'Far' distinct


def test_collapse_distinct_points_all_kept():
    a, b, c = _m("A"), _m("B"), _m("C")
    front = [(a, 0.1, 20.0), (b, 1.0, 40.0), (c, 10.0, 55.0)]
    kept = _collapse_near_duplicates(front, _DUP_Q_INTEL, "composite_deal_score")
    assert len(kept) == 3


# --- 5. Missing-data handling -------------------------------------------------


def test_missing_intel_not_on_intel_frontier_but_on_arena():
    # Price + arena_coding_elo only, no intelligence_score.
    m = _m("ArenaOnly", price_blended=1.0, arena_coding_elo=1500.0)
    annotate_frontier([m])
    assert m.is_frontier_intel is False
    assert m.frontier_rank_intel is None
    assert m.is_frontier_arena is True
    assert m.frontier_rank_arena == 1


def test_missing_price_not_on_any_frontier():
    m = _m("NoPrice", intelligence_score=40.0, arena_coding_elo=1500.0)
    annotate_frontier([m])
    assert m.is_frontier_intel is False
    assert m.is_frontier_arena is False


# --- 6. compute_top_models: frontier order + fallback -------------------------


def test_compute_top_models_frontier_order():
    lo = _m("Lo", intelligence_score=30.0, price_blended=1.0)
    hi = _m("Hi", intelligence_score=50.0, price_blended=2.0)
    ranked = rank_models([lo, hi])
    annotate_frontier(ranked)
    top = compute_top_models(ranked, "aa")
    # Both on frontier; hero is rank 1.
    assert top[0].frontier_rank_intel == 1
    ranks = [m.frontier_rank_intel for m in top]
    assert ranks == sorted(r for r in ranks if r is not None)


def test_compute_top_models_fallback_when_no_frontier():
    """Without annotate_frontier, falls back to uptime-gated composite order."""
    a = _m("A", intelligence_score=50.0, price_blended=1.0)
    b = _m("B", intelligence_score=20.0, price_blended=1.0)
    down = _m("Down", intelligence_score=60.0, price_blended=1.0, uptime_30m=90.0)
    ranked = rank_models([a, b, down])
    # No annotate_frontier call → all frontier_rank_* None → fallback path.
    top = compute_top_models(ranked, "aa")
    names = [m.name for m in top]
    assert "Down" not in names  # uptime gate still applied in fallback
    # Ordered by composite desc.
    scores = [m.composite_deal_score for m in top]
    assert scores == sorted((s for s in scores if s is not None), reverse=True)


# --- 7. Guard: price <= 0 treated as missing (no log10 domain error) ----------


def test_zero_price_treated_as_missing():
    m = _m("ZeroPrice", intelligence_score=40.0, price_input=0.0, price_output=0.0)
    assert _frontier_price(m) is None
    annotate_frontier([m])  # must not raise
    assert m.is_frontier_intel is False


def test_frontier_price_uses_raw_not_floored():
    # Sub-floor price ($0.02) must survive on the frontier axis unfloored.
    m = _m("Cheap", intelligence_score=14.0, price_blended=0.02)
    assert _frontier_price(m) == pytest.approx(0.02)
    assert m._effective_blended_price() == pytest.approx(0.05)  # scoring floor
