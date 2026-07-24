"""Tests for the daily-diff engine (spec 03).

Pure functions, no network. Synthetic diffs cover every threshold and guard;
two cases exercise the real 07-23/07-24 archives and backward-compat.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src import writer
from src.differ import compute_changelog, load_previous_archive
from src.insights import build_user_prompt
from src.models import ArchivePayload, ChangeEvent, Changelog, MergedModel
from tests.conftest import REPO_ROOT, reconstruct_models


# ── helpers ──────────────────────────────────────────────────────────────────


def mm(model_id: str, name: str | None = None, **fields) -> MergedModel:
    """Build a MergedModel with sensible defaults for a diff test."""
    return MergedModel(model_id=model_id, name=name or model_id, **fields)


def prev_archive(models: list[dict], date: str = "2026-07-23", scoring_version: int = 2) -> dict:
    """Build a previous-archive dict (as read from disk)."""
    return {"date": date, "scoring_version": scoring_version, "models": models}


def row(model_id: str, name: str | None = None, **fields) -> dict:
    """A previous-archive model row (plain dict)."""
    d = {"model_id": model_id, "name": name or model_id, "source_pages": ["leaderboard"]}
    d.update(fields)
    return d


# ── 1. New model detected ────────────────────────────────────────────────────


def test_new_model_detected():
    prev = prev_archive([row("a", intelligence_score=40.0), row("b", intelligence_score=30.0)])
    today = [mm("a", intelligence_score=40.0), mm("b", intelligence_score=30.0),
             mm("c", name="C", intelligence_score=48.0, price_input=0.4, price_output=0.5)]
    cl = compute_changelog(today, prev)
    assert [e.model_id for e in cl.new_models] == ["c"]
    assert cl.new_models[0].kind == "new"
    assert cl.new_models[0].price is not None
    assert not cl.removed_models


# ── 2. Removed model detected ────────────────────────────────────────────────


def test_removed_model_detected():
    prev = prev_archive([row("a"), row("b"), row("c", intelligence_score=20.0, price_input=1.0)])
    today = [mm("a"), mm("b")]
    cl = compute_changelog(today, prev)
    assert [e.model_id for e in cl.removed_models] == ["c"]
    assert cl.removed_models[0].kind == "removed"
    assert cl.data_quality is None


# ── 3. Price change threshold ────────────────────────────────────────────────


def test_price_cut_above_threshold_and_noise_below():
    prev = prev_archive([
        row("cut", price_blended=0.28),
        row("noise", price_blended=1.00),
    ])
    today = [mm("cut", price_blended=0.21), mm("noise", price_blended=0.99)]
    cl = compute_changelog(today, prev)
    ids = {e.model_id: e for e in cl.price_changes}
    assert "noise" not in ids  # 1% < 5% floor
    assert ids["cut"].direction == "cut"
    assert ids["cut"].pct == -25.0


# ── 4. Intelligence change threshold ─────────────────────────────────────────


def test_intelligence_change_threshold():
    prev = prev_archive([row("up", intelligence_score=45.0), row("noise", intelligence_score=45.0)])
    today = [mm("up", intelligence_score=47.0), mm("noise", intelligence_score=45.2)]
    cl = compute_changelog(today, prev)
    ids = {e.model_id: e for e in cl.intelligence_changes}
    assert "noise" not in ids  # 0.2 < 0.5
    assert ids["up"].delta == 2.0
    assert ids["up"].direction == "up"


# ── 5. Speed noise suppressed ────────────────────────────────────────────────


def test_speed_noise_suppressed():
    prev = prev_archive([row("jit", output_speed_tps=100.0), row("big", output_speed_tps=100.0)])
    today = [mm("jit", output_speed_tps=112.0), mm("big", output_speed_tps=140.0)]
    cl = compute_changelog(today, prev)
    ids = {e.model_id for e in cl.speed_changes}
    assert "jit" not in ids  # +12% < 25%
    assert "big" in ids      # +40% >= 25%


# ── 6. Rank moves ────────────────────────────────────────────────────────────


def test_rank_moves():
    # prev order: 12 frontier models ranked 1..12; today reshuffles.
    prev_rows = [row(f"m{i}", frontier_rank_intel=i) for i in range(1, 13)]
    prev = prev_archive(prev_rows)
    # today: m12 jumps to rank 8 (entered top 10); m5 -> rank 6 (small move);
    # m3 drops out of the frontier entirely (left top 10). Reassign 1..12.
    today_ranks = {
        "m1": 1, "m2": 2, "m4": 3, "m6": 4, "m5": 6, "m7": 5,
        "m8": 7, "m12": 8, "m9": 9, "m10": 10, "m11": 11,
    }  # m3 absent from frontier
    today = [mm(mid, frontier_rank_intel=r) for mid, r in today_ranks.items()]
    cl = compute_changelog(today, prev)
    by_id = {e.model_id: e for e in cl.rank_changes}
    assert by_id["m12"].entered_top is True
    assert "m5" not in by_id  # moved only 1 position
    assert by_id["m3"].left_top is True


# ── 7. Outage guard (per-source) ─────────────────────────────────────────────


def test_outage_guard_per_source():
    # 300 arena-text-only models yesterday, all gone today (source down).
    prev_rows = [row(f"at{i}", source_pages=["arena-text"]) for i in range(300)]
    prev_rows += [row("keep", source_pages=["leaderboard"])]
    prev = prev_archive(prev_rows)
    today = [mm("keep")]
    cl = compute_changelog(today, prev)
    assert cl.data_quality == "partial"
    assert cl.degraded_sources == ["arena-text"]
    assert cl.removed_models == []  # arena-text removals suppressed


# ── 8. Outage guard (global cap) ─────────────────────────────────────────────


def test_outage_guard_global_cap():
    # 500 mixed-source models, >5% (and >20) vanish across many sources — but
    # every source is still present today (no per-source outage), so the drop is
    # unexplained → degraded.
    prev_rows = [row(f"m{i}", source_pages=[f"src{i % 5}"]) for i in range(500)]
    prev = prev_archive(prev_rows)
    today = [mm(f"m{i}", source_pages=[f"src{i % 5}"]) for i in range(400)]  # 100 removed >> max(20, 25)
    cl = compute_changelog(today, prev)
    assert cl.data_quality == "degraded"
    assert cl.removed_models == []


# ── 9. First run / missing previous ──────────────────────────────────────────


def test_first_run():
    cl = compute_changelog([mm("a"), mm("b")], None)
    assert cl.first_run is True
    assert cl.total_events == 0
    assert cl.new_models == [] and cl.removed_models == []


def test_load_previous_missing_manifest(tmp_path):
    assert load_previous_archive("2026-07-24", archive_dir=tmp_path) is None


# ── 10. Gap handling (06-20 gap) ─────────────────────────────────────────────


def test_load_previous_skips_gap(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"dates": ["2026-06-21", "2026-06-19", "2026-06-18"]})
    )
    for d in ("2026-06-21", "2026-06-19", "2026-06-18"):
        (tmp_path / f"{d}.json").write_text(json.dumps({"date": d, "models": []}))
    prev = load_previous_archive("2026-06-21", archive_dir=tmp_path)
    assert prev is not None and prev["date"] == "2026-06-19"  # skips missing 06-20


# ── 11. Real-archive smoke test ──────────────────────────────────────────────


def test_real_archive_smoke():
    ranked = reconstruct_models(REPO_ROOT / "docs" / "archive" / "2026-07-24.json")
    from src.scorer import annotate_frontier, annotate_sizes, rank_models

    annotate_sizes(ranked)
    ranked = rank_models(ranked)
    annotate_frontier(ranked)
    prev = json.loads((REPO_ROOT / "docs" / "archive" / "2026-07-23.json").read_text())
    cl = compute_changelog(ranked, prev)
    assert any(e.model_id == "agnes-2-5-pro-alpha" for e in cl.new_models)
    assert cl.data_quality is None
    assert cl.compared_to == "2026-07-23"
    # 07-23 is scoring_version 1; the guard suppresses rank/frontier events.
    assert cl.scoring_changed is True
    assert cl.rank_changes == [] and cl.frontier_changes == []


# ── 12. Backward compatibility ───────────────────────────────────────────────


def test_old_archive_changelog_is_none():
    data = json.loads(
        (REPO_ROOT / "docs" / "archive" / "2026-06-01.json").read_text(encoding="utf-8")
    )
    assert "changelog" not in data
    payload = ArchivePayload(**data)
    assert payload.changelog is None


# ── scoring_version guard ────────────────────────────────────────────────────


def test_scoring_version_guard_suppresses_score_events():
    # 12 ranked models; "a" sits at rank 12 (out of top 10) and is off-frontier.
    prev_rows = [row("a", frontier_rank_intel=12, is_frontier_intel=False)]
    prev_rows += [row(f"m{i}", frontier_rank_intel=i, is_frontier_intel=True) for i in range(1, 12)]
    # Today "a" jumps to rank 8 (entered top 10) AND onto the frontier.
    today = [mm("a", frontier_rank_intel=8, is_frontier_intel=True)]
    today += [mm(f"m{i}", frontier_rank_intel=i, is_frontier_intel=True) for i in range(1, 8)]
    today += [mm(f"m{i}", frontier_rank_intel=i + 1, is_frontier_intel=True) for i in range(8, 12)]

    # prev v1 (field absent) vs today v2 → suppressed.
    prev_v1 = {"date": "2026-07-23", "models": prev_rows}  # no scoring_version key
    cl = compute_changelog(today, prev_v1)
    assert cl.scoring_changed is True
    assert cl.rank_changes == [] and cl.frontier_changes == []

    # matching versions → events fire.
    prev_v2 = prev_archive(prev_rows, scoring_version=2)
    cl2 = compute_changelog(today, prev_v2)
    assert cl2.scoring_changed is False
    assert any(e.model_id == "a" and e.entered_top for e in cl2.rank_changes)
    assert any(e.model_id == "a" and e.direction == "entered" for e in cl2.frontier_changes)


# ── changelog round-trip through the writer ──────────────────────────────────


def test_changelog_roundtrip_through_writer(tmp_path, monkeypatch):
    ranked = reconstruct_models(REPO_ROOT / "docs" / "data" / "latest.json")
    from src.scorer import annotate_frontier, annotate_sizes, compute_top_models, rank_models

    annotate_sizes(ranked)
    ranked = rank_models(ranked)
    annotate_frontier(ranked)
    top = compute_top_models(ranked, "aa")
    prev = json.loads((REPO_ROOT / "docs" / "archive" / "2026-07-23.json").read_text())
    cl = compute_changelog(ranked, prev)

    monkeypatch.setattr(writer, "DOCS_DIR", tmp_path)
    writer.write_archives(
        ranked, insights="", generated_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
        top_models=top, changelog=cl,
    )
    data = json.loads((tmp_path / "data" / "latest.json").read_text(encoding="utf-8"))
    payload = ArchivePayload(**data)
    assert payload.changelog is not None
    assert payload.changelog.compared_to == "2026-07-23"
    assert payload.changelog.total_events == cl.total_events
    assert len(payload.changelog.new_models) == len(cl.new_models)


# ── insights prompt behaviour ────────────────────────────────────────────────


def test_insights_prompt_includes_sections():
    cl = Changelog(compared_to="2026-07-23")
    cl.price_changes.append(
        ChangeEvent(
            model_id="x", name="ModelX", kind="price", old=0.28, new=0.21, pct=-25.0, direction="cut"
        )
    )
    prompt = build_user_prompt([], cl, "2026-07-24")
    assert "PRICE CHANGES" in prompt
    assert "ModelX" in prompt
    assert "Changes since 2026-07-23" in prompt


def test_insights_prompt_quiet_day():
    cl = Changelog(compared_to="2026-07-23")  # no events
    prompt = build_user_prompt([], cl, "2026-07-24")
    assert "market was quiet" in prompt
    # every section renders "none"
    assert prompt.count("- none") >= 6
