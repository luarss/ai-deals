from __future__ import annotations

import json

import pytest

from src.insights import FALLBACK
from tests.conftest import REPO_ROOT

LATEST = REPO_ROOT / "docs" / "data" / "latest.json"
MIN_SCORED_MODELS = 30

pytestmark = pytest.mark.artifact


@pytest.fixture(scope="module")
def latest() -> dict:
    if not LATEST.exists():
        pytest.skip("docs/data/latest.json has not been published")
    return json.loads(LATEST.read_text(encoding="utf-8"))


def test_hero_present(latest):
    assert latest.get("hero"), "published payload has no hero"


def test_enough_scored_models(latest):
    scored = [m for m in latest["models"] if m.get("composite_deal_score") is not None]
    assert len(scored) >= MIN_SCORED_MODELS


def test_insights_non_empty(latest):
    insights = (latest.get("insights") or "").strip()
    assert insights
    assert insights != FALLBACK.strip()


def test_bands_non_empty(latest):
    bands = latest.get("bands") or []
    assert bands, "published payload has no bands"
    empty = [band["id"] for band in bands if not band.get("count")]
    assert not empty, f"empty bands: {empty}"


def test_every_model_identified(latest):
    unidentified = [
        index
        for index, model in enumerate(latest["models"])
        if not model.get("model_id") or not model.get("name")
    ]
    assert not unidentified
