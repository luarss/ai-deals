# Spec 04 — Trends (longitudinal price/intelligence history)

Status: proposed
Owner: TBD
Related specs: `02-intelligence-bands` (share band definitions), `05-normalized-scores`
(day-relative composite scores are being fixed; trends must NOT depend on them),
`03-daily-diff` (both read consecutive archives; share the effective-price helper).

---

## 1. Problem

The pipeline writes one JSON archive per day (`docs/archive/<date>.json`, plus
`docs/data/latest.json`) and never reads them back. After ~60 days we have a rich
longitudinal dataset (43,703 model-rows across 61 files) and **zero** longitudinal
insight surfaced to users. The site only ever shows a single day.

The headline story this data can tell is **"the cost of capability is falling"** —
the cheapest price to buy inference at or above a given intelligence level, tracked
over time — plus **per-model price/intelligence history** (sparklines on table rows).

Serving trends by fetching every archive client-side is a non-starter: 61 files ×
~500 KB ≈ 30 MB per page load. We need a single **precomputed time-series artifact**
built once per day and fetched once by the browser.

### Measured constraints (from the real 61 archives, 2026-05-24 → 2026-07-24)

| Fact | Value |
|---|---|
| Archive files | 61 (one gap: 2026-06-20 missing) |
| Archive size range | 88.9 KB (2026-05-24) → 742 KB (2026-07-15), latest 428 KB |
| Total model-rows across all files | 43,703 |
| Time to load+parse all 61 files (python3) | **0.28 s** — backfill is cheap |
| Precomputed timeseries size (aggregates only) | **18.2 KB** raw |
| Precomputed timeseries (aggregates + all models with ≥14 days history, 140 models) | **236.7 KB** raw / **18.1 KB** gzipped |
| Precomputed timeseries (aggregates + curated ~12 notable models) | **48.7 KB** raw / **3.0 KB** gzipped |

GitHub Pages serves gzip, so even the 237 KB variant is one ~18 KB transfer. This is
the crux of the design: replace 30 MB of client fetches with a single small file.

---

## 2. Design (with rationale + findings)

### 2.1 Use RAW metrics, never composite scores

`composite_deal_score` / `arena_composite_score` are computed in `src/scorer.py` by
min-max normalising each day's population **independently** (see `_normalise`,
`_build_range`). A score of 0.8 on Monday and 0.8 on Tuesday are NOT comparable —
they are ranks within that day's cohort. Plotting them over time is meaningless and
actively misleading. (This day-relative flaw is the subject of spec 05.)

**Therefore trends use only raw, absolute metrics:** `intelligence_score`,
effective blended price ($/1M tok), and `output_speed_tps`. These are provider-
reported absolutes and are comparable across days. The timeseries builder must not
read any `*_score` / `*_value` field.

### 2.2 Effective blended price (critical — do not read `price_blended` directly)

**Finding — data anomaly:** on the latest archive (2026-07-24) `price_blended` is
`null`/`0` for **all 523 models** (an AA scraper regression), yet `price_input`,
`price_output`, and the OpenRouter prices are present. Coverage of `price_blended`
also swings wildly across days (164 → 128 → 142 → 169 → 0). If aggregates keyed on
`price_blended` alone, the headline chart would show gaps and a cliff to zero on the
latest day.

The builder MUST replicate the existing fallback ladder already implemented in two
places — `MergedModel._effective_blended_price` (`src/models.py:89`) and the JS
`effPrice` (`templates/index.html:400`):

```
price_blended (if > 0)
  → (2*price_input + price_output) / 3
  → price_input
  → (2*openrouter_price_input + openrouter_price_output) / 3
  → openrouter_price_input
  → None (skip model for that day)
```

To avoid a third copy drifting, extract this into a shared helper (see §5.1) and have
`models.py` delegate to it. The backfill/build reads it from plain dicts (archives are
JSON, not `MergedModel` instances), so the helper takes a `dict`.

### 2.3 Schema drift across archives (empirically checked: 2026-05-24 vs 2026-07-24)

The archive schema grew over time. Older files have **fewer fields**; the builder must
treat every field as optional and use `.get()`.

**Top-level keys:**
- 2026-05-24: `date, generated_at, insights, total_models, models, hero, runners_up`
- 2026-07-24: adds `arena_hero, arena_runners_up`

**Per-model fields:** 16 fields (old) → 31 fields (new). Present in new but absent in
the 4 oldest files (2026-05-24 → 2026-05-27):
`model_id, arena_elo, arena_ci, arena_votes, arena_coding_elo, arena_value,
arena_coding_value, arena_composite_score, openrouter_price_input,
openrouter_price_output, openrouter_model_id, input_modalities, output_modalities,
uptime_30m, uptime_1d`.

**Consequences for the builder:**
- **`model_id` is absent in the 4 oldest archives** (2026-05-24 → 05-27), present from
  2026-05-28 onward. Per-model sparkline series are keyed by `model_id`, so those series
  simply **start 2026-05-28**. This is acceptable and must be handled gracefully (skip
  models with no `model_id` when building per-model series).
- **Do not key models by `name`** for series: the name format changed from bare
  (`"Qwen3.5 0.8B"`) to `"Creator: Model"` (`"DeepSeek: DeepSeek V4 Flash"`,
  `"inclusionAI: Ling-2.6-flash"`). `model_id` is the only stable key
  (e.g. `deepseek-v4-flash` is identical across all days that have it).
- **Per-day aggregates do NOT need `model_id`** — they only need `(intelligence, price)`
  pairs — so aggregates cover all 61 days including the 4 oldest.
- **Intelligence coverage is sparse and volatile** (210 → 163 → 49 → 83 → 115 models
  with `intelligence_score` on sampled days). Aggregates must tolerate days with few or
  zero qualifying models per band and emit `null` cheapest/median with `count: 0` rather
  than crashing or interpolating.

### 2.4 Model-series inclusion policy

Including every model that ever appeared bloats the file with one-day noise. Measured
tradeoffs (aggregates + per-model series):

| Inclusion cutoff | Models | Raw | Gzipped |
|---|---|---|---|
| ≥14 days of history | 140 | 236.7 KB | 18.1 KB |
| ≥21 days | 123 | 223.1 KB | 16.9 KB |
| ≥30 days | 66 | 164.3 KB | 11.9 KB |
| curated ~12 notable | 12 | 48.7 KB | 3.0 KB |

**Decision:** include per-model series for any model with **≥14 days** of history
(140 models today, 18 KB gzipped). This gives a sparkline for almost every long-lived
table row while dropping transient noise. Cheap now; see §2.5 for growth bounding.

### 2.5 Growth bounding

Per-model points grow linearly with days tracked. At 1 year: ~140 models × 365 days ×
~30 bytes ≈ 1.5 MB raw (~100 KB gzipped) — still one small transfer. To keep it bounded
regardless, the builder **windows per-model `points` to the most recent 180 days**.
Day-level aggregates are kept for the full history (they are tiny: 18 KB for all days).

### 2.6 Build strategy: full rebuild, not incremental append

The load+parse of all 61 archives takes **0.28 s**. The daily job already reads/writes
`docs/`. So rather than maintaining a fragile incremental append (which would need
back-patching when a day's data corrects, and re-deriving the ≥14-day inclusion set),
**rebuild `timeseries.json` from scratch every run** from the freshly-written archive
plus all existing archives. Simple, idempotent, self-healing, and negligible cost.
(An incremental path is documented in §6 as an optional optimisation but is not needed.)

---

## 3. Time-series schema (`docs/data/timeseries.json`)

Exact shape the frontend consumes. All prices are effective blended $/1M tokens,
rounded to 4 dp. `null` means "no qualifying data that day".

```json
{
  "schema_version": 1,
  "generated_at": "2026-07-24 02:17 UTC",
  "thresholds": [20, 30, 40, 50],
  "days": [
    {
      "date": "2026-05-24",
      "priced_models": 160,
      "bands": {
        "20": { "cheapest": 0.02,   "median": 0.80,   "count": 44 },
        "30": { "cheapest": 0.13,   "median": 1.80,   "count": 35 },
        "40": { "cheapest": 0.13,   "median": 2.56,   "count": 24 },
        "50": { "cheapest": 2.25,   "median": 4.67,   "count": 13 }
      }
    }
  ],
  "models": {
    "deepseek-v4-flash": {
      "name": "DeepSeek: DeepSeek V4 Flash",
      "creator": "DeepSeek",
      "points": [
        { "d": "2026-05-28", "i": 47.0, "p": 0.06 },
        { "d": "2026-05-29", "i": 47.0, "p": 0.06 }
      ]
    }
  }
}
```

Field contract:
- `thresholds`: intelligence cut points shared with spec 02 (see §7). Band key `"T"`
  aggregates all models with `intelligence_score >= T` (cumulative "at or above",
  not exclusive buckets — this is what "cheapest ≥40-intelligence model" means).
- `days[].bands[T].cheapest`: min effective price among models with `intelligence >= T`.
- `days[].bands[T].median`: median effective price among those models.
- `days[].bands[T].count`: how many models qualified (drives the "N models" caption
  and lets the frontend hide bands with too-few points).
- `days[].priced_models`: total models that had both an intelligence score and a
  usable price that day (denominator / sanity signal).
- `days` is sorted ascending by date. The 2026-06-20 gap is simply an absent entry;
  the frontend must plot by actual date, not array index (see §6 gap handling).
- `models[id].points`: ascending by date, windowed to last 180 days. `i` =
  intelligence_score, `p` = effective price. Points where either is missing are omitted.

---

## 4. Backfill script design (`scripts/build_timeseries.py`)

A standalone script that produces `docs/data/timeseries.json` from all archives on
disk. Used both for the one-time backfill and (imported) by the daily pipeline (§5).

```
scripts/build_timeseries.py
  main():
    ts = build_timeseries(archive_dir=docs/archive, generated_at=now)
    write docs/data/timeseries.json  (json.dumps, ensure_ascii=False, no indent)
```

Core function (place the reusable logic in `src/timeseries.py`, see §5.2):

```python
def build_timeseries(archive_dir: Path, generated_at: datetime) -> dict:
    files = sorted(archive_dir.glob("20*.json"))   # excludes manifest.json
    days = []
    hist: dict[str, list[dict]] = {}      # model_id -> points
    meta: dict[str, dict] = {}            # model_id -> {name, creator}
    for f in files:
        payload = json.loads(f.read_text())
        date = payload["date"]
        pairs = []                        # (intelligence, price) for this day
        for m in payload.get("models", []):
            intel = m.get("intelligence_score")
            price = effective_price(m)    # shared helper, §5.1
            if intel is None or not price or price <= 0:
                continue
            pairs.append((intel, price))
            mid = m.get("model_id")       # absent in 4 oldest files → skip series
            if mid:
                hist.setdefault(mid, []).append({"d": date, "i": intel, "p": round(price, 4)})
                meta[mid] = {"name": m.get("name"), "creator": m.get("creator")}
        days.append(_day_aggregate(date, pairs, THRESHOLDS))
    models = _select_series(hist, meta, min_days=14, window_days=180)
    return {
        "schema_version": 1,
        "generated_at": generated_at.strftime("%Y-%m-%d %H:%M UTC"),
        "thresholds": THRESHOLDS,
        "days": days,
        "models": models,
    }
```

- `_day_aggregate(date, pairs, thresholds)`: for each `T`, filter pairs with `i >= T`,
  compute `min` / `statistics.median` of prices, `count`. Empty → `cheapest/median = None,
  count = 0`. Also emits `priced_models = len(pairs)`.
- `_select_series(hist, meta, min_days, window_days)`: drop models with
  `len(points) < min_days`; keep only the last `window_days` points (points already
  ascending by date); attach `name`/`creator` from `meta`.
- Robustness: wrap per-file parse in try/except and `log.warning` + skip on malformed
  files, so one bad archive can't break the whole build.

Run once to backfill: `uv run python scripts/build_timeseries.py`.

---

## 5. Exact changes per file

### 5.1 `src/pricing.py` (NEW) — single source of truth for effective price

Extract the fallback ladder so builder, model, and (optionally) diff share it.

```python
def effective_price(rec: dict) -> float | None:
    """Effective blended $/1M tokens from a raw archive-model dict."""
    pb = rec.get("price_blended")
    if pb is not None and pb > 0:
        return pb
    pi, po = rec.get("price_input"), rec.get("price_output")
    if pi is not None and po is not None:
        return (2 * pi + po) / 3
    if pi is not None:
        return pi
    ori, oro = rec.get("openrouter_price_input"), rec.get("openrouter_price_output")
    if ori is not None and oro is not None:
        return (2 * ori + oro) / 3
    if ori is not None:
        return ori
    return None
```

Then refactor `MergedModel._effective_blended_price` (`src/models.py:89-103`) to call
`effective_price(self.model_dump())` (or keep its logic and add a module-level shared
constant test asserting parity — see §8). Not strictly required for trends to work, but
prevents a third divergent copy.

### 5.2 `src/bands.py` (NEW / shared with spec 02) — threshold definitions

```python
INTELLIGENCE_THRESHOLDS = [20, 30, 40, 50]   # cumulative "≥T" cut points
```

Spec 02 (intelligence-bands) must import the same list so table band chips and trend
bands never diverge. If spec 02 lands first, import from wherever it defines them and
delete this stub. See §7.

### 5.3 `src/timeseries.py` (NEW) — build logic

Contains `build_timeseries`, `_day_aggregate`, `_select_series` from §4, importing
`effective_price` (§5.1) and `INTELLIGENCE_THRESHOLDS` (§5.2). Pure functions, no I/O
except reading archive files, so it is unit-testable.

### 5.4 `src/writer.py` — write timeseries each run

Add a function and call it from `write_archives` (or from `main.py`, see 5.5). After the
day's archive is written to disk (so the current day is included in the rebuild):

```python
from src.timeseries import build_timeseries

def write_timeseries(generated_at: datetime) -> None:
    ts = build_timeseries(DOCS_DIR / "archive", generated_at)
    path = DOCS_DIR / "data" / "timeseries.json"
    path.write_text(json.dumps(ts, ensure_ascii=False), encoding="utf-8")
    log.info("Timeseries written: %d days, %d model series", len(ts["days"]), len(ts["models"]))
```

Call it at the end of `write_archives` (after `_update_manifest`), passing
`generated_at`. It re-reads the just-written `<date>.json`, so ordering matters:
archive write → timeseries build.

### 5.5 `main.py` — no change required

`write_archives` already receives `generated_at` (`main.py:96`); the timeseries write
is triggered inside it. If preferred, call `write_timeseries(generated_at)` explicitly in
`main.py` right after `write_archives(...)` and log at step granularity. Either is fine;
keep it inside `write_archives` for a single write path.

### 5.6 `.github/workflows/daily.yml` — no change required

`uv run python main.py` now also emits `docs/data/timeseries.json`; the existing
`git add docs/` picks it up. Confirm the file is committed on first run.

### 5.7 `templates/index.html` — frontend (see §6 for the data contract)

---

## 6. Frontend changes (`templates/index.html`)

### 6.1 Data loading

`timeseries.json` is small (~18 KB gzipped) and independent of the day selector, so
load it once at init alongside `latest.json` and `manifest.json`
(`templates/index.html:658-665`). Store in a module-level `var trends = null;`. If the
fetch fails, hide the trends section and log — never block the main table.

```js
Promise.all([
  fetch('data/latest.json').then(r => r.json()),
  fetch('archive/manifest.json').then(r => r.ok ? r.json() : {dates:[]}).catch(() => ({dates:[]})),
  fetch('data/timeseries.json').then(r => r.ok ? r.json() : null).catch(() => null)
]).then(([data, manifest, ts]) => { trends = ts; buildDateSelector(manifest); renderPage(data); renderTrends(ts); });
```

### 6.2 Rendering approach

No chart library is currently loaded (Tailwind CDN only). Keep zero new dependencies:
render charts as **inline SVG** built in JS (same spirit as the existing hand-rolled
DOM rendering). This keeps the ~43 KB template dependency-free.

### 6.3 Chart 1 — "The cost of capability is falling" (headline line chart)

New `<section id="trends-section">` placed directly under the hero/runners-up, above the
"All Models" table (insert after line 116, before line 118).

- **X axis:** date (use real dates from `days[].date`; the 2026-06-20 gap must render as
  a break in the line, not a straight interpolation — plot points at x = date, connect
  only consecutive-ish points, or draw markers + gap-aware polyline).
- **Y axis:** effective price $/1M tokens, **log scale** (prices span 0.02 → ~5, and the
  whole point is exponential decline; log makes the fall linear-ish and readable).
- **Series:** one line per threshold, `days[].bands[T].cheapest`, for `T` in
  `thresholds`. Skip null points. Legend: "cheapest ≥20", "≥30", "≥40", "≥50".
- **Caption:** e.g. "Cheapest price to run inference at each intelligence level. The
  ≥40-intelligence floor fell from \$X to \$Y over N days." Compute X/Y from first and
  last non-null `bands["40"].cheapest`.
- **Data contract consumed:** `trends.days[i].date`, `trends.days[i].bands[T].cheapest`,
  `trends.days[i].bands[T].count` (gray out / dashed a series segment where `count` is
  very low, e.g. `< 3`, so a one-model day doesn't look authoritative).

### 6.4 Chart 2 — median price per band (companion, optional toggle)

Same axes, plotting `bands[T].median` instead of `cheapest`. A toggle
("cheapest / median") switches the series source. Median is more robust to a single
cheap outlier; cheapest is the punchier headline. Ship cheapest by default with a toggle.

### 6.5 Chart 3 — per-model price sparklines on table rows

Augment the Model column in `renderTable` (`templates/index.html:474-490`). For each row
whose model has a series in `trends.models[model_id]`, append a tiny inline-SVG sparkline
of `points[].p` (last ~30 points), ~80×20 px, after the model name.

- **Row keying:** add `data-model-id` to each `<tr>` (`m.model_id`, may be null for
  ancient re-renders — guard). Look up `trends.models[m.model_id]`.
- **Sparkline contract:** `trends.models[id].points` → array of `{d, i, p}`; use `p`
  (effective price) for the y-values, min/max normalised within the sparkline. Color:
  green if price trend is down (last < first), red if up — reinforces "cheaper = better".
- Graceful absence: models without a series (short-lived, or pre-`model_id` only) render
  no sparkline. No layout shift beyond the reserved 80 px.
- Optional hover tooltip: intelligence `i` and price `p` at the hovered date.

### 6.6 Methodology note

Add one line to the methodology section (`templates/index.html:173-180`): trends use
raw intelligence and effective blended price (never the day-relative deal score), and
"≥T intelligence" bands are cumulative.

---

## 7. Integration with parallel specs

- **02-intelligence-bands** — MUST share `INTELLIGENCE_THRESHOLDS` (§5.2). The trend
  band keys (`"20"/"30"/"40"/"50"`) and the table's band filter chips must use one
  definition. Coordinate: whichever spec lands first owns `src/bands.py`; the other
  imports it. If spec 02 chooses different cut points, this spec adopts them (the schema
  is threshold-agnostic — `thresholds` is a data field, not hardcoded in the frontend).
- **03-daily-diff** — both read consecutive archives and both need effective price.
  Share `src/pricing.effective_price` (§5.1). Keep the artifacts separate
  (`timeseries.json` vs the diff output); a diff can optionally be derived from the last
  two `days[]` entries but is out of scope here.
- **05-normalized-scores** — this spec deliberately avoids composite scores (§2.1). When
  spec 05 introduces cross-day-comparable normalised scores, a future iteration MAY add a
  normalised-score series to `timeseries.json`; until then, raw metrics only. No blocking
  dependency in either direction.

---

## 8. Test plan

No test suite exists today; add `tests/` with pytest (dev dependency).

1. **`effective_price` parity** — table-driven: assert `src.pricing.effective_price`
   returns the same value as `MergedModel._effective_blended_price` for a matrix of
   field combinations, including: only `openrouter_*` present (the 2026-07-24 case),
   `price_blended = 0` (must fall through), only `price_input`, all null → `None`.
2. **Schema-drift robustness** — feed `build_timeseries` a fixture dir containing a
   minimal 2026-05-24-shaped model (16 fields, no `model_id`, no `openrouter_*`) and a
   2026-07-24-shaped one (31 fields, `price_blended` null but `openrouter_*` set).
   Assert: no exception; the old-shape day still produces band aggregates; the old-shape
   model produces no per-model series (no `model_id`); the new-shape model with only
   OpenRouter prices IS counted in aggregates and gets a series point.
3. **Aggregate correctness** — hand-built pairs `[(45, 2.0), (42, 0.5), (30, 0.1),
   (10, 0.01)]`: `bands["40"].cheapest == 0.5`, `count == 2`, `median == 1.25`;
   `bands["50"]` → `cheapest None, count 0`.
4. **Empty / sparse day** — a day with zero priced models yields all bands
   `null/0` and `priced_models == 0`, no crash.
5. **Series windowing & inclusion** — a model with 10 days is excluded (min 14); a model
   with 200 days is trimmed to the last 180; points remain ascending by date.
6. **Gap handling** — with 2026-06-20 absent, `days` has no entry for it and dates remain
   strictly ascending (frontend plots by date).
7. **Full-corpus smoke** — run `build_timeseries` over the real `docs/archive` and assert
   `len(days) == number of archive files`, output JSON < 300 KB, and every
   `days[].bands` has all four threshold keys. (Guards against size regressions.)
8. **Frontend contract (manual/lightweight)** — load `timeseries.json` in the browser,
   confirm the headline chart renders 4 lines, the gap shows a break, and a known
   long-lived model (`deepseek-v4-flash`) shows a sparkline; a one-day model shows none.

---

## 9. Dependencies / integration notes

- **No new runtime Python deps** — uses stdlib `json`, `statistics`, `pathlib`,
  `datetime`. Add `pytest` as a dev dependency only.
- **No new JS deps** — inline SVG, no chart library; preserves the dependency-free
  ~43 KB static template.
- **File write ordering** — archive `<date>.json` must be written before the timeseries
  rebuild so the current day is included (§5.4).
- **CI** — `git add docs/` already stages the new `docs/data/timeseries.json`; verify it
  is committed on the first scheduled run.
- **Backfill step** — run `uv run python scripts/build_timeseries.py` once and commit the
  generated `docs/data/timeseries.json` so the feature is live before the next daily run.
- **Idempotence** — the build is a full deterministic rebuild; re-running produces
  identical output (modulo `generated_at`).
```
