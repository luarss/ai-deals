"""Effective-price parity + ladder tests (spec 04 §8.1, 00-overview §1)."""

from __future__ import annotations

import pytest

from src.models import MergedModel
from src.pricing import effective_price

# Field-combination matrix: each dict is a raw archive-model shape.
CASES = [
    # only openrouter present — the 2026-07-24 case (price_blended null)
    {"openrouter_price_input": 0.3, "openrouter_price_output": 1.2},
    {"openrouter_price_input": 0.5},
    # price_blended = 0 must fall through, not win
    {"price_blended": 0, "price_input": 0.4, "price_output": 1.0},
    {"price_blended": None, "price_input": 0.4, "price_output": 1.0},
    # price_blended positive wins
    {"price_blended": 2.5, "price_input": 0.4, "price_output": 1.0},
    # only price_input
    {"price_input": 0.9},
    # input+output blend
    {"price_input": 1.0, "price_output": 4.0},
    # blended over openrouter
    {"price_blended": 1.1, "openrouter_price_input": 9.0, "openrouter_price_output": 9.0},
    # everything null -> None
    {},
    {"price_output": 5.0},  # output alone is NOT usable (matches ladder)
]


@pytest.mark.parametrize("rec", CASES)
def test_effective_price_parity_dict_vs_model(rec: dict) -> None:
    """src.pricing.effective_price(dict) == MergedModel raw price for same inputs."""
    model = MergedModel(model_id="x", name="X", **rec)
    from_dict = effective_price(rec)
    from_model = model._raw_blended_price()
    assert from_dict == from_model
    # public alias must also agree
    assert model.effective_price() == from_dict


@pytest.mark.parametrize("rec", CASES)
def test_effective_price_accepts_model_instance(rec: dict) -> None:
    model = MergedModel(model_id="x", name="X", **rec)
    assert effective_price(model) == effective_price(rec)


def test_specific_values() -> None:
    assert effective_price({"price_blended": 2.5}) == 2.5
    assert effective_price({"price_input": 1.0, "price_output": 4.0}) == (2 * 1.0 + 4.0) / 3
    assert effective_price({"price_input": 0.9}) == 0.9
    assert effective_price(
        {"openrouter_price_input": 0.3, "openrouter_price_output": 1.2}
    ) == (2 * 0.3 + 1.2) / 3
    assert effective_price({"openrouter_price_input": 0.5}) == 0.5
    assert effective_price({}) is None
    assert effective_price({"price_blended": 0}) is None
