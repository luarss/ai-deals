"""Regression tests: re-score real archives and assert the known v2 outcome."""

from __future__ import annotations

import pytest

from src.scorer import annotate_sizes, compute_top_models, passes_uptime_gate, rank_models

from tests.conftest import REPO_ROOT, reconstruct_models

LATEST = REPO_ROOT / "docs" / "data" / "latest.json"
PREV = REPO_ROOT / "docs" / "archive" / "2026-06-23.json"


def _score(archive_path):
    models = reconstruct_models(archive_path)
    annotate_sizes(models)
    ranked = rank_models(models)
    return models, ranked


def test_ling_no_longer_first_deepseek_above_it():
    _, ranked = _score(LATEST)
    top = compute_top_models(ranked, "aa")
    names = [m.name for m in top]

    ling_idx = next(i for i, n in enumerate(names) if "Ling-2.6-flash" in n)
    deepseek_idx = next(i for i, n in enumerate(names) if "DeepSeek V4 Flash" in n)

    # Ling must no longer be the #1 deal (the P1/P2 outlier squash is fixed)
    assert ling_idx != 0, f"Ling regressed back to #1 (top: {names[:3]})"
    # DeepSeek V4 Flash ranks above Ling (its value leg 0.893 > Ling's 0.874)
    assert deepseek_idx < ling_idx, (
        f"DeepSeek (#{deepseek_idx + 1}) should rank above Ling (#{ling_idx + 1})"
    )


def test_top_field_not_one_model_dominated():
    _, ranked = _score(LATEST)
    top = compute_top_models(ranked, "aa")
    assert top[0].composite_deal_score and top[1].composite_deal_score
    ratio = top[1].composite_deal_score / top[0].composite_deal_score
    assert ratio > 0.6, f"field is outlier-dominated: 2nd/1st = {ratio:.3f}"


def test_all_top_models_pass_uptime_gate():
    _, ranked = _score(LATEST)
    top = compute_top_models(ranked, "aa")
    assert all(passes_uptime_gate(m) for m in top)


def test_scores_are_deterministic():
    _, ranked_a = _score(LATEST)
    _, ranked_b = _score(LATEST)
    a = [(m.name, m.composite_deal_score) for m in ranked_a]
    b = [(m.name, m.composite_deal_score) for m in ranked_b]
    assert a == b


@pytest.mark.skipif(not PREV.exists(), reason="previous archive not present")
def test_cross_day_comparability():
    """A model present in both archives with an unchanged raw value_score gets
    the same composite_deal_score on both days (day-stable normalization)."""
    models_new, _ = _score(LATEST)
    models_old, _ = _score(PREV)

    def index(models):
        out = {}
        for m in models:
            out[m.name] = m
        return out

    new_by_name = index(models_new)
    old_by_name = index(models_old)

    checked = 0
    for name, mn in new_by_name.items():
        mo = old_by_name.get(name)
        if mo is None:
            continue
        # same raw inputs -> same composite (compare the value/coding/speed triple)
        if (
            mn.value_score == mo.value_score
            and mn.coding_value == mo.coding_value
            and mn.output_speed_tps == mo.output_speed_tps
        ):
            assert mn.composite_deal_score == mo.composite_deal_score, name
            checked += 1

    assert checked > 0, "no shared model with identical raw inputs to compare"
