"""Unit tests for the v2 scoring mechanics (spec 05)."""

from __future__ import annotations

import pytest

from src.models import PRICE_FLOOR, MergedModel
from src.scorer import (
    NEUTRAL,
    SCORING_VERSION,
    SP_REF,
    VS_REF,
    W_CODING,
    W_SPEED,
    W_VALUE,
    _size_tier,
    annotate_sizes,
    compute_top_models,
    is_deal_candidate,
    normalize_log,
    passes_uptime_gate,
    rank_models,
)


# --- 1. normalize_log ---------------------------------------------------------


def test_normalize_log_endpoints():
    assert normalize_log(5, (5, 500)) == 0.0
    assert normalize_log(500, (5, 500)) == 1.0


def test_normalize_log_midpoint():
    assert normalize_log(37, (5, 500)) == pytest.approx(0.435, abs=0.005)


def test_normalize_log_high_clamp():
    assert normalize_log(5000, (5, 500)) == 1.0


def test_normalize_log_low_clamp():
    assert normalize_log(1, (5, 500)) == 0.0


def test_normalize_log_missing_is_neutral():
    assert normalize_log(None, (5, 500)) == NEUTRAL == 0.5


# --- 2. Day-stability (the P1 regression) ------------------------------------


def _model(name: str, **kw) -> MergedModel:
    return MergedModel(model_id=name.lower().replace(" ", "-"), name=name, **kw)


def test_day_stability_shared_model_identical_score():
    """A model with a fixed value_score scores identically regardless of pool."""
    shared_a = _model("Shared", intelligence_score=40.0, price_blended=1.0)
    shared_b = _model("Shared", intelligence_score=40.0, price_blended=1.0)
    assert shared_a.value_score == shared_b.value_score

    pool_a = [shared_a, _model("Outlier", intelligence_score=14.0, price_blended=0.01)]
    pool_b = [
        shared_b,
        _model("BigA", intelligence_score=50.0, price_blended=0.02),
        _model("BigB", intelligence_score=5.0, price_blended=0.05),
    ]
    rank_models(pool_a)
    rank_models(pool_b)
    assert shared_a.composite_deal_score == shared_b.composite_deal_score


# --- 3. Outlier robustness ----------------------------------------------------


def test_outlier_does_not_squash_field():
    big = _model("Big", value_score=5000.0)
    mid = _model("Mid", value_score=500.0)
    normal = _model("Normal", value_score=180.0)
    # value_score set directly (bypass price); populate via attribute
    for m, vs in ((big, 5000.0), (mid, 500.0), (normal, 180.0)):
        m.value_score = vs
    rank_models([big, mid, normal])
    # both extreme value legs clamp to 1.0
    assert normalize_log(big.value_score, VS_REF) == 1.0
    assert normalize_log(mid.value_score, VS_REF) == 1.0
    # the mid-field model is not compressed toward 0
    assert normalize_log(normal.value_score, VS_REF) == pytest.approx(0.778, abs=0.005)


# --- 4. Price floor -----------------------------------------------------------


def test_price_floor_applied_to_value_score():
    m = _model(
        "Cheap",
        intelligence_score=14.0,
        openrouter_price_input=0.01,
        openrouter_price_output=0.03,
    )
    # effective price floored to 0.05 -> 14 / 0.05 = 280
    assert m.value_score == round(14 / PRICE_FLOOR, 4) == 280.0


def test_price_floor_does_not_touch_raw_price():
    m = _model(
        "Cheap",
        intelligence_score=14.0,
        openrouter_price_input=0.01,
        openrouter_price_output=0.03,
    )
    raw = m._raw_blended_price()
    assert raw is not None
    assert raw == pytest.approx((2 * 0.01 + 0.03) / 3)
    assert raw < PRICE_FLOOR  # raw is untouched; only the scoring price is floored


# --- 5. Neutral imputation ----------------------------------------------------


def test_neutral_imputation_missing_legs():
    m = _model("ValueOnly", value_score=180.0)
    m.value_score = 180.0  # coding_value and speed remain None
    rank_models([m])
    v = normalize_log(180.0, VS_REF)
    expected = round(W_VALUE * v + W_CODING * NEUTRAL + W_SPEED * NEUTRAL, 4)
    assert m.composite_deal_score == expected


def test_missing_coding_not_boosted_above_neutral_reporter():
    """A coding-missing model must not outscore an identical model that reports
    coding at exactly the neutral point."""
    missing = _model("Missing", value_score=180.0)
    missing.value_score = 180.0  # coding None
    # neutral-point coding: value that normalizes to 0.5 -> 10**((0.5*(log500-log5))+log5)
    import math

    neutral_cv = 10 ** (0.5 * (math.log10(500) - math.log10(5)) + math.log10(5))
    reporter = _model("Reporter", value_score=180.0)
    reporter.value_score = 180.0
    reporter.coding_value = neutral_cv
    rank_models([missing, reporter])
    assert missing.composite_deal_score == pytest.approx(
        reporter.composite_deal_score, abs=0.001
    )


# --- 6. Tier ------------------------------------------------------------------


def test_size_tier_unknown_when_no_size():
    assert _size_tier(None, "GPT-5.5") == "unknown"


def test_size_tier_from_param_count():
    m = _model("Qwen 7B")
    annotate_sizes([m])
    assert m.param_billions == 7.0
    assert m.size_tier == "small"


def test_size_tier_default_is_unknown():
    assert MergedModel(model_id="x", name="X").size_tier == "unknown"


# --- 7. Uptime gate -----------------------------------------------------------


def test_uptime_gate_no_data_passes():
    assert passes_uptime_gate(_model("A")) is True


def test_uptime_gate_at_threshold_passes():
    assert passes_uptime_gate(_model("A", uptime_30m=95.0)) is True


def test_uptime_gate_below_threshold_fails():
    assert passes_uptime_gate(_model("A", uptime_30m=92.65)) is False


def test_uptime_gate_uses_best_of_two():
    assert passes_uptime_gate(_model("A", uptime_1d=99.0)) is True
    assert passes_uptime_gate(_model("A", uptime_30m=90.0, uptime_1d=99.0)) is True


# --- 8. compute_top_models ----------------------------------------------------


def test_compute_top_models_includes_non_frontier_tier():
    large = _model("Big 70B", intelligence_score=40.0, price_blended=1.0)
    annotate_sizes([large])
    assert large.size_tier == "large"
    ranked = rank_models([large])
    top = compute_top_models(ranked, "aa")
    assert large in top


def test_compute_top_models_excludes_down_model():
    up = _model("Up", intelligence_score=40.0, price_blended=1.0, uptime_30m=100.0)
    down = _model("Down", intelligence_score=40.0, price_blended=1.0, uptime_30m=90.0)
    ranked = rank_models([up, down])
    top = compute_top_models(ranked, "aa")
    assert up in top
    assert down not in top


def test_is_deal_candidate_requires_score_and_uptime():
    down = _model("Down", intelligence_score=40.0, price_blended=1.0, uptime_30m=90.0)
    rank_models([down])
    assert is_deal_candidate(down, "aa") is False


def test_scoring_version_is_two():
    assert SCORING_VERSION == 2
