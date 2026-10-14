"""Integration tests: frontier annotation + archive backward-compat (spec 01)."""

from __future__ import annotations

import json

from src import writer
from src.models import ArchivePayload
from src.scorer import (
    _frontier_price,
    annotate_frontier,
    annotate_sizes,
    compute_top_models,
    rank_models,
)
from tests.conftest import REPO_ROOT, reconstruct_models

PINNED_ARCHIVE = REPO_ROOT / "docs" / "archive" / "2026-07-24.json"
# 2026-05-24 (the very oldest) predates the required `model_id` field and never
# validated even before spec 01 — a separate pre-existing schema gap. 2026-05-25
# is the oldest archive that carries model_id yet lacks the frontier fields, so
# it is the correct target for testing the *new* fields' backward-compat.
OLD_ARCHIVE = REPO_ROOT / "docs" / "archive" / "2026-05-25.json"


def _pipeline(path):
    models = reconstruct_models(path)
    annotate_sizes(models)
    ranked = rank_models(models)
    annotate_frontier(ranked)
    return ranked


# --- 8. Measured counts + hero regression ------------------------------------


def test_pipeline_counts_and_hero():
    ranked = _pipeline(PINNED_ARCHIVE)

    intel_cands = [
        m for m in ranked if _frontier_price(m) is not None and m.intelligence_score is not None
    ]
    arena_cands = [
        m
        for m in ranked
        if _frontier_price(m) is not None
        and (m.arena_elo is not None or m.arena_coding_elo is not None)
    ]
    intel_front = [m for m in ranked if m.is_frontier_intel]
    arena_front = [m for m in ranked if m.is_frontier_arena]

    assert len(intel_cands) == 51
    assert len(intel_front) == 12
    assert len(arena_cands) == 93
    assert len(arena_front) == 8

    top = compute_top_models(ranked, "aa")
    assert "Ling-2.6-flash" not in top[0].name
    assert "DeepSeek V4 Flash" in top[0].name


# --- 9. Backward compatibility: old archive lacking new fields ----------------


def test_old_archive_validates():
    data = json.loads(OLD_ARCHIVE.read_text(encoding="utf-8"))
    # Archive predating the frontier fields — must still validate on read.
    assert not any("is_frontier_intel" in m for m in data["models"])
    payload = ArchivePayload(**data)
    # New top-level frontier arrays default to empty.
    assert payload.frontier_intel == []
    assert payload.frontier_arena == []
    # Its model objects lack the frontier flags → defaults applied.
    assert all(m.is_frontier_intel is False for m in payload.models)
    assert all(m.frontier_rank_intel is None for m in payload.models)
    assert all(m.frontier_score_arena is None for m in payload.models)


# --- 10. write_archives round-trip carries the six new fields ------------------


def test_write_archives_roundtrip(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    ranked = _pipeline(PINNED_ARCHIVE)
    top = compute_top_models(ranked, "aa")
    arena_top = compute_top_models(ranked, "arena")

    monkeypatch.setattr(writer, "DOCS_DIR", tmp_path)
    writer.write_archives(
        ranked,
        insights="",
        generated_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
        top_models=top,
        arena_top_models=arena_top,
    )

    data = json.loads((tmp_path / "data" / "latest.json").read_text(encoding="utf-8"))
    payload = ArchivePayload(**data)  # re-parses cleanly

    assert payload.hero is not None
    assert "DeepSeek V4 Flash" in payload.hero.name
    assert len(payload.frontier_intel) == 12
    assert len(payload.frontier_arena) == 8
    # Frontier arrays are price-ascending (raw price).
    prices = [m._raw_blended_price() for m in payload.frontier_intel]
    assert prices == sorted(p for p in prices if p is not None)

    always_present = {"is_frontier_intel", "is_frontier_arena"}
    for m in data["models"]:
        assert always_present.issubset(m.keys())
        assert None not in m.values()
    for m in data["models"]:
        if m["is_frontier_intel"]:
            assert {"frontier_rank_intel", "frontier_score_intel"}.issubset(m.keys())
        if m["is_frontier_arena"]:
            assert {"frontier_rank_arena", "frontier_score_arena"}.issubset(m.keys())


def test_written_archive_is_compact(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    ranked = _pipeline(PINNED_ARCHIVE)
    monkeypatch.setattr(writer, "DOCS_DIR", tmp_path)
    writer.write_archives(
        ranked,
        insights="",
        generated_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
        top_models=compute_top_models(ranked, "aa"),
    )

    text = (tmp_path / "archive" / "2026-07-24.json").read_text(encoding="utf-8")
    assert ", " not in text.replace('", "', "")
    assert '": ' not in text
    data = json.loads(text)
    assert None not in data["hero"].values()
    reparsed = ArchivePayload(**data)
    assert len(reparsed.models) == len(ranked)


def test_compact_archive_data_is_idempotent():
    data = {
        "date": "2026-07-24",
        "hero": {"model_id": "a", "name": "A", "price_blended": None},
        "arena_hero": None,
        "models": [{"model_id": "a", "name": "A", "price_blended": None, "source_pages": []}],
        "changelog": {"data_quality": None},
    }
    once = writer.compact_archive_data(data)
    twice = writer.compact_archive_data(once)

    assert once == twice
    assert once["hero"] == {"model_id": "a", "name": "A"}
    assert once["arena_hero"] is None
    assert once["models"] == [{"model_id": "a", "name": "A", "source_pages": []}]
    assert once["changelog"] == {"data_quality": None}


def _healthy_report_inputs():
    from src.bands import compute_bands

    ranked = _pipeline(PINNED_ARCHIVE)
    return {
        "aa_count": 300,
        "arena_count": 200,
        "openrouter_count": 400,
        "ranked": ranked,
        "bands": compute_bands(ranked),
        "insights": "Prices fell today.",
        "hero": compute_top_models(ranked, "aa")[0],
    }


def _failed_names(checks):
    return sorted(check.name for check in checks)


def test_health_report_counts():
    from src.health import build_health_report

    report = build_health_report(**_healthy_report_inputs())

    assert report.merged_count == len(_healthy_report_inputs()["ranked"])
    assert report.scored_count > 0
    assert report.frontier_count == 12
    assert set(report.band_counts) == {"flagship", "workhorse", "budget"}
    assert report.hard_failures == []


def test_health_hard_failures_abort(monkeypatch):
    import pytest

    from src.health import SKIP_ENV_VAR, build_health_report, enforce

    inputs = _healthy_report_inputs() | {"aa_count": 10, "openrouter_count": 0, "hero": None}
    report = build_health_report(**inputs)

    assert _failed_names(report.hard_failures) == ["aa", "hero", "openrouter"]
    monkeypatch.delenv(SKIP_ENV_VAR, raising=False)
    with pytest.raises(SystemExit) as excinfo:
        enforce(report)
    assert excinfo.value.code == 1

    monkeypatch.setenv(SKIP_ENV_VAR, "1")
    enforce(report)


def test_health_zero_scored_is_hard_failure():
    from src.health import build_health_report

    inputs = _healthy_report_inputs()
    for model in inputs["ranked"]:
        model.composite_deal_score = None
    report = build_health_report(**inputs)

    assert "scored_nonzero" in _failed_names(report.hard_failures)


def test_health_soft_failures_recorded_in_changelog(capsys):
    from src.health import apply_to_changelog, build_health_report, enforce, log_report
    from src.insights import FALLBACK
    from src.models import BandDeal, Changelog

    inputs = _healthy_report_inputs() | {
        "arena_count": 0,
        "insights": FALLBACK,
        "bands": [BandDeal(id="budget", label="Budget", min_intelligence=0, count=0)],
    }
    report = build_health_report(**inputs)
    changelog = Changelog(compared_to="2026-07-23")

    log_report(report)
    apply_to_changelog(report, changelog)
    enforce(report)

    soft = _failed_names(report.soft_failures)
    assert {"arena", "band:budget", "insights"}.issubset(soft)
    assert report.hard_failures == []
    assert changelog.data_quality == "partial"
    assert {"arena", "band:budget", "insights"}.issubset(changelog.degraded_sources)
    assert changelog.health["source_counts"]["arena"] == 0
    assert "::warning title=Data health::arena below floor" in capsys.readouterr().out


def test_health_keeps_existing_degraded_quality():
    from src.health import apply_to_changelog, build_health_report
    from src.models import Changelog

    report = build_health_report(**(_healthy_report_inputs() | {"arena_count": 0}))
    changelog = Changelog(data_quality="degraded", degraded_sources=["leaderboard"])

    apply_to_changelog(report, changelog)

    assert changelog.data_quality == "degraded"
    assert "leaderboard" in changelog.degraded_sources
    assert "arena" in changelog.degraded_sources
    payload = ArchivePayload(
        date="2026-07-24",
        generated_at="x",
        insights="",
        total_models=0,
        models=[],
        changelog=changelog,
    )
    assert ArchivePayload(**json.loads(payload.model_dump_json())).changelog.health == changelog.health
