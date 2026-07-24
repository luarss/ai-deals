# 03 — Daily Diff / Changelog

## Problem

The tracker ships a full snapshot every day, but ~99% of each day's `docs/archive/<date>.json`
is byte-for-byte identical to the previous day. The *news* — a price cut, a new model, a model
disappearing, a frontier/ranking move — is the actual product of a daily tracker, and it is
never computed or surfaced. The site headline is a static "Best Deal Today" hero that rarely
changes. Nothing tells a returning visitor "what changed since yesterday."

This spec adds a **diff engine** that compares today's merged+scored models against the most
recent previous archive, emits a structured **changelog**, embeds it in the archive JSON
(backward compatible), makes it the **headline** of the site, and rewrites the DeepSeek
insights prompt to summarize the diff instead of restating the leaderboard.

---

## Design

### Identity: how models are matched across days

`model_id` is the join key. It is minted in `src/merger.py::_resolve_slug` (slug from href →
normalised-name registry → fuzzy match ≥0.82 → derived slug) and persisted in every archive.

**Empirical churn (measured with `python3` over adjacent archives):**

| Pair | prev N | today N | common | added | removed |
|---|---|---|---|---|---|
| 07-22 → 07-23 | 521 | 522 | 521 | 1 | 0 |
| 07-23 → 07-24 | 522 | 523 | 522 | 1 | 0 |
| 07-13 → 07-14 | 913 | 915 | 912 | 3 | 1 |
| 06-17 → 06-18 | 741 | 742 | 740 | 2 | 1 |

On normal days, `model_id` churn is **0–3 models/day** and there are **zero duplicate
`model_id`s within a single archive**. Identity is stable enough to diff directly on `model_id`
with **no fuzzy re-matching at diff time** — the merger already did the fuzzy work, and re-fuzzing
would risk masking genuine renames. The one caveat is genuine slug renames (e.g.
`gpt-4o-mini-2024-07-18` → `gpt-4o-mini` on 07-14, which shows as 1 add + 1 remove). We accept
these as a paired add/remove; they are rare (~1/week) and honest to report.

### Threshold calibration (measured day-over-day noise)

Distribution of `abs(pct change)` for models present on both days, across 6 normal (non-outage)
adjacent pairs:

| field | n | % that changed | median | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| `price_blended` | 535 | **0.2%** | 0 | 0 | 0 | 0 | 100% |
| `price_input` | 1396 | **0.3%** | 0 | 0 | 0 | 0 | 25% |
| `price_output` | 1396 | **0.4%** | 0 | 0 | 0 | 0 | 40% |
| `openrouter_price_input` | 1833 | **1.4%** | 0 | 0 | 0 | 5.8% | 90% |
| `intelligence_score` | 616 | **0.5%** | 0 | 0 | 0 | 0 | 20% |
| `output_speed_tps` | 789 | **82.9%** | 2.7% | 13% | **18%** | 39% | 72% |
| `arena_elo` | 1467 | **4.0%** | 0 | 0 | 0 | 0.09% | 0.3% |

**Conclusions that drive the thresholds:**

1. **Prices essentially never jitter.** When a price moves at all it moves by a large,
   real amount (median nonzero change 16–18%, often 25–100%). There is no float noise to
   filter. Threshold is set at **≥5% relative** purely as a floor against rounding artifacts;
   in practice it catches every real price move. This is a *real signal every time.*
2. **Intelligence score never jitters** (0.5% of models, and only in whole/large steps).
   Threshold: **≥0.5 points absolute** (scale is 0–60).
3. **Speed is extremely noisy** — 83% of models change *every day*, median 2.7%, p95 ~18%.
   Speed is a sampled measurement, not a quote. A naive speed diff would fire on hundreds of
   models daily and drown the signal. Threshold: **≥25% relative** (above p95≈18%, near p99≈39%)
   **and** speed events are **low priority** (never headlined, table-only, capped list).
4. **Arena Elo drifts slowly** — magnitudes are tiny (p99 = 0.09%), so absolute-Elo change
   events are useless. We surface **arena rank moves** instead, not absolute Elo deltas.
5. **Rank moves of ±1–2 positions are noise** driven by the speed jitter feeding
   `composite_deal_score` (observed: top-10 frontier list only swaps adjacent neighbours
   day-to-day). We only emit rank events for **entering/leaving the top 10** or a move of
   **≥3 positions within the top 20** of the frontier deal list.

### Outage detection (mass fake removals)

Two archives in the sample show catastrophic drops: **07-17 → 07-18** (334 removed) and
**07-20 → 07-21** (399 removed). Root cause, from `source_pages` counts: the entire
`arena-text` source (374 records) failed to fetch on those days. Those "removed" models are
**not gone** — a source was down.

Defense (two layers):
- **Per-source guard.** Compute, for prev and today, the set of `source_pages` present and
  the model count contributed by each. If any source that had ≥50 models yesterday has
  **0 models today**, mark that source `degraded` and **suppress all removal events for models
  whose `source_pages` (in the previous archive) were a subset of the degraded source(s)**.
- **Global sanity cap.** If total removals exceed `max(20, 5% of prev N)`, set
  `changelog.data_quality = "degraded"`, suppress the `removed` section entirely, and show a
  banner instead of a removal list. (Both outage days blow past this cap: 334/399 ≫ ~45.)

### Where the diff runs

In `main.py`, after `rank_models(...)` and before `write_archives(...)`. The differ loads the
previous archive **from disk** (`docs/archive/`), choosing the most recent date **strictly
before today** from `manifest.json` (this transparently handles the **06-20 gap** — on 06-21 the
"previous" is 06-19). The computed `Changelog` is passed into `write_archives` and embedded in
the payload; DeepSeek insights are then generated *from the changelog*.

---

## Exact changes per file

### NEW: `src/differ.py`

```python
"""Compute a day-over-day changelog from the current models vs. the previous archive."""
from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Optional
from src.models import MergedModel, Changelog, ChangeEvent  # new models, see below

log = logging.getLogger(__name__)
DOCS_DIR = Path(__file__).parent.parent / "docs"

# --- thresholds (see spec §Threshold calibration) ---
PRICE_REL_THRESHOLD = 0.05      # 5% — floor vs rounding; prices don't jitter
INTEL_ABS_THRESHOLD = 0.5       # points on 0-60 scale
SPEED_REL_THRESHOLD = 0.25      # 25% — above p95(18%) speed jitter; low priority
RANK_MOVE_MIN = 3               # positions, within top 20
RANK_TOP_N = 10                 # entering/leaving this window is always an event
REMOVAL_SANITY_CAP_FRAC = 0.05  # >5% of prev N (or 20) => outage
REMOVAL_SANITY_CAP_MIN = 20
SOURCE_DEGRADE_MIN = 50         # a source w/ >=50 models yesterday, 0 today => degraded


def effective_price(m: dict | MergedModel) -> Optional[float]:
    """Mirror models.MergedModel._effective_blended_price for dict or model input."""
    g = (lambda k: getattr(m, k, None)) if isinstance(m, MergedModel) else m.get
    pb = g("price_blended")
    if pb and pb > 0: return pb
    pi, po = g("price_input"), g("price_output")
    if pi is not None and po is not None: return (2*pi + po) / 3
    if pi is not None: return pi
    opi, opo = g("openrouter_price_input"), g("openrouter_price_output")
    if opi is not None and opo is not None: return (2*opi + opo) / 3
    if opi is not None: return opi
    return None


def load_previous_archive(today: str) -> Optional[dict]:
    """Return the parsed archive for the most recent date strictly before `today`."""
    manifest = DOCS_DIR / "archive" / "manifest.json"
    if not manifest.exists():
        return None
    dates = sorted(json.loads(manifest.read_text()).get("dates", []), reverse=True)
    for d in dates:
        if d < today:
            p = DOCS_DIR / "archive" / f"{d}.json"
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
    return None


def _frontier_deal_order(models: list[dict]) -> list[str]:
    """model_ids of frontier-tier models ordered by composite_deal_score desc (mirror scorer)."""
    fr = [m for m in models if m.get("size_tier") == "frontier"
          and m.get("composite_deal_score") is not None]
    fr.sort(key=lambda m: -(m.get("composite_deal_score") or 0))
    return [m["model_id"] for m in fr]


def compute_changelog(
    today_models: list[MergedModel],
    prev_archive: Optional[dict],
    today_date: str,
) -> Changelog:
    """Diff today's ranked models against the previous archive. Never raises."""
    today = [m.model_dump(mode="json") for m in today_models]
    if prev_archive is None:
        return Changelog(compared_to=None, first_run=True)  # empty, first_run flag set

    prev = prev_archive.get("models", [])
    prev_by_id = {m["model_id"]: m for m in prev}
    today_by_id = {m["model_id"]: m for m in today}
    prev_ids, today_ids = set(prev_by_id), set(today_by_id)

    cl = Changelog(compared_to=prev_archive.get("date"))

    # --- outage / source-degrade detection FIRST ---
    def source_counts(models):
        c = {}
        for m in models:
            for s in m.get("source_pages", []):
                c[s] = c.get(s, 0) + 1
        return c
    prev_src, today_src = source_counts(prev), source_counts(today)
    degraded_sources = {
        s for s, n in prev_src.items()
        if n >= SOURCE_DEGRADE_MIN and today_src.get(s, 0) == 0
    }

    # --- NEW models ---
    for mid in sorted(today_ids - prev_ids):
        m = today_by_id[mid]
        cl.new_models.append(ChangeEvent(
            model_id=mid, name=m["name"], kind="new",
            intelligence_score=m.get("intelligence_score"),
            price=effective_price(m),
            composite_deal_score=m.get("composite_deal_score"),
        ))

    # --- REMOVED models (with outage guards) ---
    removed_ids = sorted(prev_ids - today_ids)
    over_cap = len(removed_ids) > max(REMOVAL_SANITY_CAP_MIN,
                                      int(len(prev) * REMOVAL_SANITY_CAP_FRAC))
    if over_cap:
        cl.data_quality = "degraded"           # suppress removals entirely
    else:
        for mid in removed_ids:
            m = prev_by_id[mid]
            src = set(m.get("source_pages", []))
            if src and src.issubset(degraded_sources):
                continue                        # fake removal: its only source is down
            cl.removed_models.append(ChangeEvent(
                model_id=mid, name=m["name"], kind="removed",
                intelligence_score=m.get("intelligence_score"),
                price=effective_price(m),
            ))
    if degraded_sources and cl.data_quality is None:
        cl.data_quality = "partial"
        cl.degraded_sources = sorted(degraded_sources)

    # --- field changes on COMMON models ---
    for mid in sorted(prev_ids & today_ids):
        p, t = prev_by_id[mid], today_by_id[mid]
        name = t["name"]
        # price
        pp, tp = effective_price(p), effective_price(t)
        if pp and tp and pp > 0 and abs(tp - pp) / pp >= PRICE_REL_THRESHOLD:
            cl.price_changes.append(ChangeEvent(
                model_id=mid, name=name, kind="price",
                old=round(pp, 6), new=round(tp, 6),
                pct=round((tp - pp) / pp * 100, 1),
                direction="cut" if tp < pp else "hike",
            ))
        # intelligence
        pi, ti = p.get("intelligence_score"), t.get("intelligence_score")
        if pi is not None and ti is not None and abs(ti - pi) >= INTEL_ABS_THRESHOLD:
            cl.intelligence_changes.append(ChangeEvent(
                model_id=mid, name=name, kind="intelligence",
                old=pi, new=ti, delta=round(ti - pi, 1),
                direction="up" if ti > pi else "down",
            ))
        # speed (low priority, high threshold)
        ps, ts = p.get("output_speed_tps"), t.get("output_speed_tps")
        if ps and ts and ps > 0 and abs(ts - ps) / ps >= SPEED_REL_THRESHOLD:
            cl.speed_changes.append(ChangeEvent(
                model_id=mid, name=name, kind="speed",
                old=round(ps), new=round(ts),
                pct=round((ts - ps) / ps * 100, 1),
                direction="up" if ts > ps else "down",
            ))

    # --- rank moves in frontier deal list ---
    prev_order = _frontier_deal_order(prev)
    today_order = _frontier_deal_order(today)
    prev_rank = {mid: i for i, mid in enumerate(prev_order)}
    for new_i, mid in enumerate(today_order[:20]):
        if mid not in prev_rank:
            continue
        old_i = prev_rank[mid]
        entered_top = old_i >= RANK_TOP_N and new_i < RANK_TOP_N
        left_top = old_i < RANK_TOP_N and new_i >= RANK_TOP_N  # handled below via prev scan
        big_move = abs(new_i - old_i) >= RANK_MOVE_MIN and new_i < 20
        if entered_top or big_move:
            cl.rank_changes.append(ChangeEvent(
                model_id=mid, name=today_by_id[mid]["name"], kind="rank",
                old_rank=old_i + 1, new_rank=new_i + 1,
                direction="up" if new_i < old_i else "down",
                entered_top=entered_top,
            ))
    # models that LEFT the top 10 (were in prev top 10, now >10 or gone)
    for mid in prev_order[:RANK_TOP_N]:
        new_i = today_order.index(mid) if mid in today_order else None
        if new_i is None or new_i >= RANK_TOP_N:
            cl.rank_changes.append(ChangeEvent(
                model_id=mid, name=prev_by_id[mid]["name"], kind="rank",
                old_rank=prev_rank[mid] + 1,
                new_rank=(new_i + 1) if new_i is not None else None,
                direction="down", left_top=True,
            ))

    cl.total_events = (len(cl.new_models) + len(cl.removed_models)
                       + len(cl.price_changes) + len(cl.intelligence_changes)
                       + len(cl.speed_changes) + len(cl.rank_changes))
    return cl
```

Sort each event list before returning for deterministic output:
`price_changes` by `abs(pct)` desc, `intelligence_changes` by `abs(delta)` desc,
`new_models` by `composite_deal_score` desc, `rank_changes` by magnitude of move desc,
`speed_changes` capped to top 15 by `abs(pct)`.

### `src/models.py` — add two models, extend `ArchivePayload`

```python
class ChangeEvent(BaseModel):
    model_id: str
    name: str
    kind: str  # new | removed | price | intelligence | speed | rank
    # populated per-kind; all optional so one class serves every event type
    intelligence_score: Optional[float] = None
    price: Optional[float] = None
    composite_deal_score: Optional[float] = None
    old: Optional[float] = None
    new: Optional[float] = None
    delta: Optional[float] = None
    pct: Optional[float] = None
    direction: Optional[str] = None          # cut/hike, up/down
    old_rank: Optional[int] = None
    new_rank: Optional[int] = None
    entered_top: bool = False
    left_top: bool = False

class Changelog(BaseModel):
    compared_to: Optional[str] = None         # previous archive date, or None
    first_run: bool = False
    data_quality: Optional[str] = None        # None | "partial" | "degraded"
    degraded_sources: list[str] = Field(default_factory=list)
    total_events: int = 0
    new_models: list[ChangeEvent] = Field(default_factory=list)
    removed_models: list[ChangeEvent] = Field(default_factory=list)
    price_changes: list[ChangeEvent] = Field(default_factory=list)
    intelligence_changes: list[ChangeEvent] = Field(default_factory=list)
    speed_changes: list[ChangeEvent] = Field(default_factory=list)
    rank_changes: list[ChangeEvent] = Field(default_factory=list)
```

Add to `ArchivePayload` (optional → backward compatible; old archives parse fine, field
defaults to `None`):
```python
    changelog: Optional["Changelog"] = None
```

### `src/writer.py` — accept and embed the changelog

```python
def write_archives(models, insights, generated_at,
                   top_models=None, arena_top_models=None,
                   changelog=None):                       # NEW param
    ...
    payload = ArchivePayload(
        ...,
        changelog=changelog,                              # NEW
    )
```
No other writer change; `model_dump(mode="json")` serializes the nested changelog automatically.

### `main.py` — wire it in

```python
from src.differ import load_previous_archive, compute_changelog

# after: ranked = rank_models(merged); top_models/arena_top computed
today_str = generated_at.strftime("%Y-%m-%d")   # move generated_at up before this block
prev_archive = load_previous_archive(today_str)
changelog = compute_changelog(ranked, prev_archive, today_str)
log.info("Changelog: %d events (%d new, %d removed, %d price, %d intel, %d rank), quality=%s",
         changelog.total_events, len(changelog.new_models), len(changelog.removed_models),
         len(changelog.price_changes), len(changelog.intelligence_changes),
         len(changelog.rank_changes), changelog.data_quality or "ok")

# insights now summarize the diff
insights = generate_insights(top_models[:10], changelog, api_key) if api_key else ""

write_archives(ranked, insights, generated_at,
               top_models=top_models, arena_top_models=arena_top,
               changelog=changelog)
```
(`compute_changelog` never raises; on any internal error it should log and return
`Changelog(first_run=True)` so the pipeline never aborts over a diff failure.)

### `src/insights.py` — summarize the diff (see New insights prompt below)

Signature change: `generate_insights(top_models, changelog, api_key)` and a new
`build_user_prompt(top_models, changelog, as_of)`.

---

## Archive schema changes (backward compatible)

One new top-level optional key `changelog` in each archive and `latest.json`:

```jsonc
"changelog": {
  "compared_to": "2026-07-23",
  "first_run": false,
  "data_quality": null,                 // or "partial" | "degraded"
  "degraded_sources": [],               // e.g. ["arena-text"] when partial
  "total_events": 4,
  "new_models": [
    {"model_id":"agnes-2-5-pro-alpha","name":"Agnes 2.5 Pro (alpha)","kind":"new",
     "intelligence_score":48.0,"price":0.42,"composite_deal_score":0.71}
  ],
  "removed_models": [ /* ChangeEvent kind=removed */ ],
  "price_changes": [
    {"model_id":"deepseek-v4-flash","name":"DeepSeek V4 Flash","kind":"price",
     "old":0.28,"new":0.21,"pct":-25.0,"direction":"cut"}
  ],
  "intelligence_changes": [
    {"model_id":"...","name":"...","kind":"intelligence","old":45.0,"new":47.0,
     "delta":2.0,"direction":"up"}
  ],
  "speed_changes": [ /* capped 15, kind=speed, old/new/pct/direction */ ],
  "rank_changes": [
    {"model_id":"...","name":"...","kind":"rank","old_rank":12,"new_rank":8,
     "direction":"up","entered_top":true,"left_top":false}
  ]
}
```

- **Backward compatibility:** older archives (05-24 … today) lack the key; `ArchivePayload`
  defaults `changelog=None`. The frontend must treat a missing/`null` changelog as "no data"
  and fall back to the existing hero. Do **not** backfill old archives (optional one-off script
  could, but out of scope).
- `latest.json` and `archive/<date>.json` carry identical `changelog` blocks (same writer path).

---

## Frontend changes (`templates/index.html`)

### Data contract (JS consumes `data.changelog`)

```
data.changelog?               // object | null | undefined  → if falsy, render nothing, keep hero
  .compared_to                // string date, e.g. "2026-07-23" (for "vs Jul 23" label)
  .first_run                  // bool → show "First snapshot — no prior day to compare" note
  .data_quality               // null | "partial" | "degraded"
  .degraded_sources           // string[]
  .total_events               // int  → if 0 (and not first_run) show "No notable changes today"
  .new_models[]               // {name, intelligence_score, price, composite_deal_score}
  .removed_models[]           // {name, intelligence_score, price}
  .price_changes[]            // {name, old, new, pct, direction:"cut"|"hike"}
  .intelligence_changes[]     // {name, old, new, delta, direction:"up"|"down"}
  .speed_changes[]            // {name, old, new, pct, direction}
  .rank_changes[]             // {name, old_rank, new_rank, direction, entered_top, left_top}
```

### UI

Add a new **`#changelog-section`** as the **first** child of `<main>`, *above* `#hero-section`
(lines 93–99). This becomes the site headline: **"What changed today"**.

- Section header: `What changed today` + subtitle `vs {compared_to}` (or `First snapshot` when
  `first_run`, or `No notable changes since {compared_to}` when `total_events === 0`).
- A compact **summary strip** of pill counters, only rendering non-zero groups:
  `🆕 3 new · 💸 2 price cuts · 📈 1 smarter · ⬆ 1 entered top 10 · ➖ 1 removed`.
- Grouped cards below, in priority order: **Price cuts → New models → Rank/frontier moves →
  Intelligence changes → Removed → Speed (collapsed `<details>`, capped 15)**.
  - Price cut row: `DeepSeek V4 Flash  $0.28 → $0.21  (−25%)` with green for cut, amber for hike
    (reuse `fmtPrice`).
  - New model row: name + `intel 48.0 · $0.42 · deal 0.71` badges.
  - Rank row: `Model ▲ #12 → #8` with an `Entered top 10` / `Left top 10` chip.
- **Data-quality banner:** if `data_quality === "degraded"`, show an amber banner
  *"A data source was unavailable today; removals are hidden to avoid false alarms."* and skip
  the removed group. If `"partial"`, show a small note listing `degraded_sources`.
- Reuse existing dark-theme classes (`bg-gray-900 border border-gray-800 rounded-xl p-7`) so it
  matches the insights card.

### JS wiring

- New `function renderChangelog(changelog, comparedToLabel)`; call it from `renderPage(data)`
  (line ~591, before/above `renderInsights`). If `!changelog`, `add('hidden')` on the section
  and return (old snapshots and outage-suppressed days degrade gracefully).
- All values already numeric; reuse `fmtPrice`, `fmtScore`, `escapeHtml`. No new fetch — the
  changelog rides inside the already-fetched archive JSON, so the date-selector "time travel"
  works for free (each historical archive shows *its* changelog vs *its* previous day).
- Keep the existing `#hero-section`; the changelog sits above it as the lede, hero remains the
  "current best deal."

---

## New insights prompt (`src/insights.py`)

Replace `SYSTEM_PROMPT` and `build_user_prompt`. The model summarizes the **diff**; if there are
no events, it says so in one line rather than inventing news.

**New `SYSTEM_PROMPT` (draft, use verbatim):**

```
You are the editor of a daily AI-model market brief. You are given a structured changelog of
what changed in the AI model market since the previous day, plus the current top value models
for context. Write a short, punchy daily update for developers who track model pricing and
capability.

Rules:
- Lead with the single most important change (a price cut, a strong new model, or a frontier
  ranking move). Put the most newsworthy item first.
- Report only what is in the changelog. Do NOT restate a full leaderboard. Do NOT invent
  changes that are not listed. Use the exact model names and numbers provided.
- Quantify: name the model, the old and new value, and the percentage or point change.
- If the changelog is empty or marked no-change, write a single sentence saying the market was
  quiet today and (optionally) name the current best-value model for reference. Do not pad.
- If a data source was degraded, add one sentence noting some data was unavailable today.
- 120–200 words, 2–3 short paragraphs, plain prose, no headers, no bullet points, no markdown.
- Direct and factual. No hype, no marketing language.
```

**New `build_user_prompt(top_models, changelog, as_of)` (produces the user message):**

```
Date: {as_of}. Changes since {changelog.compared_to or "N/A"}.
Data quality: {changelog.data_quality or "ok"}{, unavailable sources: ...degraded_sources if any}.

PRICE CHANGES:
  - {name}: ${old} -> ${new} ({pct:+.0f}%, {direction})        # each price_changes item, or "none"
NEW MODELS:
  - {name}: intelligence {intelligence_score}, price ${price}, deal {composite_deal_score}
INTELLIGENCE CHANGES:
  - {name}: {old} -> {new} ({delta:+.1f} pts, {direction})
RANK / FRONTIER MOVES:
  - {name}: #{old_rank} -> #{new_rank} ({"entered top 10"|"left top 10"|direction})
REMOVED MODELS:
  - {name} (was intelligence {intelligence_score}, ${price})
SPEED CHANGES (secondary, top 5 only):
  - {name}: {old} -> {new} t/s ({pct:+.0f}%)

For reference only (do not just restate this list), current top value models:
  1. {name}: intelligence {..}, ${..}/1M, deal {..}
  ... (top 5)

If every section above is "none", say the market was quiet today.
Write the brief now.
```

`FALLBACK` stays. `generate_insights` keeps its try/except → FALLBACK behaviour; only the
message construction and the extra `changelog` argument change. Keep `deepseek-v4-flash`,
`temperature=0.4`; `max_tokens` can drop to ~500 (shorter output).

---

## Test plan

New `tests/test_differ.py` (pure functions, no network):

1. **New model detected** — prev has {A,B}, today {A,B,C} → one `new_models` event for C.
2. **Removed model detected** — prev {A,B,C}, today {A,B} → one `removed_models` event for C
   (below sanity cap, no degraded source).
3. **Price cut above threshold** — A price 0.28→0.21 (−25%) → event, `direction="cut"`,
   `pct=-25.0`. Price 1.00→0.99 (−1%) → **no** event (below 5%).
4. **Intelligence change** — 45→47 → event delta +2.0; 45→45.2 → no event (<0.5).
5. **Speed noise suppressed** — 100→112 t/s (+12%, within observed p90) → **no** event;
   100→140 (+40%) → event.
6. **Rank move** — model at #12 → #8 → `entered_top=True`; #7 → #8 → no event (move <3 and
   still inside top 10 window on both sides); model in prev top10 now #14 → `left_top=True`.
7. **Outage guard (per-source)** — build prev with 300 `arena-text`-only models, today with
   those absent → those removals suppressed, `data_quality="partial"`,
   `degraded_sources=["arena-text"]`.
8. **Outage guard (global cap)** — today missing >5% of prev models across mixed sources →
   `data_quality="degraded"`, `removed_models` empty.
9. **First run / missing previous** — `load_previous_archive` returns None →
   `Changelog(first_run=True)`, all lists empty, `total_events=0`.
10. **06-20 gap** — feed a manifest with dates skipping 06-20; asking previous-of-06-21 returns
    the 06-19 archive.
11. **Real-archive smoke test** — load `docs/archive/2026-07-23.json` and `2026-07-24.json`,
    run `compute_changelog`; assert `agnes-2-5-pro-alpha` appears in `new_models`, total events
    small, `data_quality is None`.
12. **Backward-compat** — parse an old archive (e.g. `2026-06-01.json`) through `ArchivePayload`
    → succeeds with `changelog is None`.

Insights: extend/add a test that `build_user_prompt` includes changelog sections and that an
empty changelog yields the "quiet day" instruction (mock the OpenAI client, assert on prompt).

Manual/QA: run `python main.py` locally (or a dry-run harness) and open `docs/index.html`;
confirm the changelog section renders above the hero, the outage banner appears when fed a
degraded day, and the date-selector shows each archive's own changelog.

---

## Dependencies / integration notes

- **No new pip dependencies.** Uses stdlib + existing pydantic/openai.
- **Spec 01 (Pareto frontier):** "entered/left the Pareto frontier" is a natural, high-value
  diff event. Coordinate the field name: 01 should add a boolean like `on_pareto_frontier` (or a
  `pareto_rank`) to `MergedModel`. If present, `differ.py` adds a **`frontier_changes`** list to
  `Changelog` (same `ChangeEvent` shape, `kind="frontier"`, `direction="entered"|"left"`) by
  diffing that flag across days — this is cleaner than the current `size_tier=="frontier"` rank
  proxy and should be *preferred* for the headline once 01 lands. Keep the rank-move logic as a
  fallback until then. Sequencing: land 03 first with rank moves; add `frontier_changes` in a
  follow-up once 01 exposes the flag.
- **Spec 05 (mechanics fixes):** 05 may change `composite_deal_score` weights / normalisation or
  the merger's slug logic. Two interactions: (a) a one-time re-weighting will cause a large
  **rank-move day** when it ships — gate it by treating the first post-05 run as `first_run`-like
  for rank events, or note it in insights, to avoid a flood of false rank moves; (b) if 05
  changes slug construction, expect a one-day spike of paired add/remove events (renames) — the
  global sanity cap already protects against a catastrophic version, but a moderate slug change
  could produce ~dozens of fake add/remove pairs; consider running 03's diff *after* 05 settles,
  or add a rename-detection pass (match removed↔new by `normalise_name`) as a later enhancement.
- **Ordering in `main.py`:** differ runs after scoring, before writing; insights after differ.
  `generated_at` must be computed before the diff so `today_str` is available.
```
