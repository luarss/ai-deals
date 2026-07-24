"""Time-series builder tests (spec 04 §8)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.bands import INTELLIGENCE_THRESHOLDS
from src.timeseries import (
    MIN_DAYS,
    WINDOW_DAYS,
    _day_aggregate,
    _select_series,
    build_timeseries,
)
from tests.conftest import REPO_ROOT

GEN_AT = datetime(2026, 7, 24, 2, 17, tzinfo=timezone.utc)


def _write_archive(dir_: Path, date: str, models: list[dict]) -> None:
    payload = {"date": date, "generated_at": date, "insights": "", "models": models}
    (dir_ / f"{date}.json").write_text(json.dumps(payload), encoding="utf-8")


# ── §8.3 aggregate correctness ────────────────────────────────────────────
def test_day_aggregate_correctness() -> None:
    pairs = [(45, 2.0), (42, 0.5), (30, 0.1), (10, 0.01)]
    day = _day_aggregate("2026-07-01", pairs, [25.0, 40.0, 50.0])
    assert day["priced_models"] == 4
    b = day["bands"]
    assert b["40"]["cheapest"] == 0.5
    assert b["40"]["count"] == 2
    assert b["40"]["median"] == 1.25  # median of [0.5, 2.0]
    assert b["50"] == {"cheapest": None, "median": None, "count": 0}
    assert b["25"]["count"] == 3  # 45, 42, 30


# ── §8.4 empty / sparse day ───────────────────────────────────────────────
def test_empty_day_no_crash() -> None:
    day = _day_aggregate("2026-07-01", [], INTELLIGENCE_THRESHOLDS)
    assert day["priced_models"] == 0
    for t in INTELLIGENCE_THRESHOLDS:
        key = str(int(t))
        assert day["bands"][key] == {"cheapest": None, "median": None, "count": 0}


# ── thresholds match the single source of truth (00-overview §2) ──────────
def test_thresholds_match_bands_module(tmp_path: Path) -> None:
    _write_archive(tmp_path, "2026-07-01", [{"intelligence_score": 45, "price_blended": 1.0}])
    ts = build_timeseries(tmp_path, GEN_AT)
    assert ts["thresholds"] == [int(t) for t in INTELLIGENCE_THRESHOLDS]
    keys = set(ts["days"][0]["bands"].keys())
    assert keys == {str(int(t)) for t in INTELLIGENCE_THRESHOLDS}


# ── §8.2 schema-drift tolerance ───────────────────────────────────────────
def test_schema_drift_old_and_new_shapes(tmp_path: Path) -> None:
    # 2026-05-24-shaped: 16 fields, no model_id, no openrouter_*
    old_model = {
        "name": "Qwen3.5 0.8B", "creator": "Alibaba",
        "intelligence_score": 45.0, "price_input": None, "price_output": None,
        "price_blended": 0.5, "output_speed_tps": None, "context_window_k": 262,
        "coding_index": None, "livecodebench_score": None, "source_pages": [],
        "value_score": 90.0, "coding_value": None, "composite_deal_score": 1.0,
        "param_billions": None, "size_tier": "nano",
    }
    _write_archive(tmp_path, "2026-05-24", [old_model])

    # 2026-07-24-shaped: price_blended null, only openrouter_* set, has model_id
    new_model = {
        "model_id": "deepseek-v4-flash", "name": "DeepSeek: DeepSeek V4 Flash",
        "creator": "DeepSeek", "intelligence_score": 47.0, "price_blended": None,
        "openrouter_price_input": 0.05, "openrouter_price_output": 0.11,
    }
    _write_archive(tmp_path, "2026-05-28", [new_model])

    ts = build_timeseries(tmp_path, GEN_AT)

    # Both days produce aggregates.
    assert len(ts["days"]) == 2
    old_day = next(d for d in ts["days"] if d["date"] == "2026-05-24")
    assert old_day["priced_models"] == 1
    assert old_day["bands"]["40"]["cheapest"] == 0.5

    # New-shape model counted in aggregates via openrouter fallback.
    new_day = next(d for d in ts["days"] if d["date"] == "2026-05-28")
    assert new_day["priced_models"] == 1

    # Old model (no model_id) yields no series; new model would need >= MIN_DAYS.
    # With one day it is excluded, but its model_id was tracked (no exception).
    assert "deepseek-v4-flash" not in ts["models"]  # only 1 day < MIN_DAYS


# ── §8.5 series windowing & inclusion ─────────────────────────────────────
def test_series_inclusion_and_windowing() -> None:
    # short model: 10 days -> excluded
    short = [{"d": f"2026-05-{d:02d}", "i": 40.0, "p": 1.0} for d in range(1, 11)]
    # long model: 200 days -> trimmed to WINDOW_DAYS, ascending
    long_pts = []
    base = datetime(2026, 1, 1)
    for k in range(200):
        day = base.fromordinal(base.toordinal() + k).strftime("%Y-%m-%d")
        long_pts.append({"d": day, "i": 50.0, "p": 2.0})
    hist = {"short": short, "long": long_pts}
    meta = {"short": {"name": "S", "creator": "c"}, "long": {"name": "L", "creator": "c"}}
    out = _select_series(hist, meta, MIN_DAYS, WINDOW_DAYS)
    assert "short" not in out
    assert "long" in out
    assert len(out["long"]["points"]) == WINDOW_DAYS
    dates = [p["d"] for p in out["long"]["points"]]
    assert dates == sorted(dates)  # ascending


# ── §8.6 gap handling ─────────────────────────────────────────────────────
def test_gap_produces_no_entry_and_dates_ascending(tmp_path: Path) -> None:
    for date in ["2026-06-19", "2026-06-21", "2026-06-22"]:  # 06-20 missing
        _write_archive(tmp_path, date, [{"intelligence_score": 45, "price_blended": 1.0}])
    ts = build_timeseries(tmp_path, GEN_AT)
    dates = [d["date"] for d in ts["days"]]
    assert "2026-06-20" not in dates
    assert dates == sorted(dates)


# ── determinism (00-overview / §9 idempotence) ────────────────────────────
def test_determinism(tmp_path: Path) -> None:
    for d in range(1, 20):
        _write_archive(
            tmp_path, f"2026-06-{d:02d}",
            [{"model_id": "m1", "name": "M1", "creator": "c",
              "intelligence_score": 45.0, "price_blended": float(d)}],
        )
    a = build_timeseries(tmp_path, GEN_AT)
    b = build_timeseries(tmp_path, GEN_AT)
    assert json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)


# ── §8.7 full-corpus smoke over the real archives ─────────────────────────
def test_full_corpus_smoke() -> None:
    archive_dir = REPO_ROOT / "docs" / "archive"
    n_files = len(list(archive_dir.glob("20*.json")))
    ts = build_timeseries(archive_dir, GEN_AT)
    assert len(ts["days"]) == n_files
    raw = json.dumps(ts, ensure_ascii=False)
    assert len(raw.encode("utf-8")) < 400_000  # size guard
    for day in ts["days"]:
        assert set(day["bands"].keys()) == {str(int(t)) for t in INTELLIGENCE_THRESHOLDS}
    # every included model series meets the min-days cutoff and is windowed
    for series in ts["models"].values():
        assert len(series["points"]) >= MIN_DAYS
        assert len(series["points"]) <= WINDOW_DAYS
