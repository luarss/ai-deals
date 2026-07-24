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

LATEST = REPO_ROOT / "docs" / "data" / "latest.json"
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
    ranked = _pipeline(LATEST)

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

    ranked = _pipeline(LATEST)
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

    new_fields = {
        "is_frontier_intel",
        "frontier_rank_intel",
        "frontier_score_intel",
        "is_frontier_arena",
        "frontier_rank_arena",
        "frontier_score_arena",
    }
    for m in data["models"]:
        assert new_fields.issubset(m.keys())
