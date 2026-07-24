# 01 — Pareto-Frontier Deal Ranking

Status: proposed
Owner: (unassigned)
Related: `plans/05-mechanics-fixes.md` (stable score scaling, size-tier fixes, uptime gating) — see "Integration notes".

---

## 1. Problem

The core metric is `value_score = intelligence_score / blended_price`
(`src/models.py:73`). It is degenerate: as `blended_price → 0`,
`value_score → ∞`, so the cheapest model wins regardless of how weak it is.

`composite_deal_score` (`src/scorer.py:97-133`) then min-max normalizes
`value_score` across the daily pool. One near-free outlier stretches the range
so far that every real model collapses toward 0.

**Confirmed in today's data (`docs/data/latest.json`, 2026-07-24):**

- Hero = `inclusionAI: Ling-2.6-flash`: intelligence **14/60**, price
  **$0.0167/1M**, `value_score` **840.0**, `composite_deal_score` **0.8231**.
- Runner-up `DeepSeek: DeepSeek V4 Flash`: intelligence **40/60**, price
  **$0.13/1M** — a far better model — yet ranks below Ling because 40/0.13 = 308
  is ~2.7x smaller than 14/0.0167 = 840.

The ranking rewards "cheap and dumb" over "cheap and smart". We replace the
ratio-driven hero selection with a **Pareto-frontier** model: a model is a
"deal" only if nothing cheaper is at least as smart. The hero is then the
**knee** of that frontier (best intelligence-per-log-dollar), not the corner.

---

## 2. Data findings (measured from `docs/data/latest.json`, 2026-07-24)

All numbers below were measured directly; use them to justify thresholds and to
build test fixtures.

| Metric | Value |
|---|---|
| Total models in pool | **523** |
| Have `intelligence_score` | **115** |
| Have an effective blended price (any source) | **357** |
| Have **both** intelligence + price (AA frontier candidates) | **51** |
| Have `coding_index` | **0** (never populated by any source) |
| Have `livecodebench_score` | **0** |
| Have `arena_elo` (text) | **0** |
| Have `arena_coding_elo` | **100** |
| Have `arena_coding_elo` **and** price (coding frontier candidates) | **93** |
| `composite_deal_score` non-null | **51** |
| `arena_composite_score` non-null | **93** |
| Models with `size_tier == "frontier"` | **305** (39 with a composite) |
| Effective price range | **$0.0167 – $300.00 /1M** |
| Price percentiles | p5 $0.083, p25 $0.37, p50 $1.00, p75 $4.00, p95 $16.67 |
| Models with price < $0.05/1M | **4** |
| Models with price < $0.02/1M | **1** (Ling-2.6-flash) |
| Intelligence range | **4 – 60**, median 30 |
| Models with zero/negative price | **0** |

**Critical finding — coding data:** `coding_index` and `livecodebench_score`
are **null for every model** in the current pool, so `coding_value` is null for
all 523 models and the "Coding Value" methodology line is dead. Any coding
frontier **must** be built from `arena_coding_elo` (93 usable points), not
`coding_index`. The AA-side coding frontier is therefore effectively the
Arena-coding frontier; we treat "coding" as an Arena-fed axis and do not block
on `coding_index` ever appearing (design still reads it first if it returns).

**Measured intelligence-vs-price Pareto frontier (12 of 51):**

```
$0.0167  intel 14  Ling-2.6-flash        <- current degenerate hero, frontier endpoint
$0.0660  intel 15  gpt-oss-20b
$0.0813  intel 24  gpt-oss-120b
$0.1307  intel 40  DeepSeek V4 Flash     <- proposed new hero (knee)
$0.2867  intel 41  Tencent Hy3
$0.5767  intel 44  DeepSeek V4 Pro
$2.2500  intel 51  Muse Spark 1.1
$3.3333  intel 54  Grok 4.5 (high)
$6.6667  intel 55  GPT-5.6 Terra
$7.0000  intel 57  Kimi K3
$13.333  intel 59  GPT-5.6 Sol
$23.333  intel 60  Claude Fable 5        <- most-expensive/smartest endpoint
```

**Measured coding (arena_coding_elo)-vs-price frontier (8 of 93):** spans IBM
Granite 4.1 8B ($0.067, elo 1200) → Kimi K3 ($7.00, elo 1678).

---

## 3. Design

### 3.1 What "on the frontier" means

For an axis pair (price `p`, quality `q`), model A **dominates** model B iff
`p_A <= p_B AND q_A >= q_B` with at least one strict. A model is **on the
frontier** iff no other candidate dominates it. Candidates are only models that
have **both** a price and that axis's quality value.

We build **two independent frontiers**, mirroring the existing two-view UI:

1. **Intelligence frontier ("aa" view):** `q = intelligence_score`,
   `p = effective blended price`. 51 candidates today, 12 on frontier.
2. **Coding/Arena frontier ("arena" view):** `q = arena_quality`,
   `p = effective blended price`, where
   `arena_quality = arena_elo if not None else arena_coding_elo`
   (today always `arena_coding_elo`; `coding_index` is read first only if it
   ever becomes non-null — it is not today). 93 candidates, 8 on frontier.

Rationale: two frontiers keep the existing AA/Arena toggle meaningful and let
each view rank on its own quality signal. We do **not** attempt a 3-axis
frontier (intel + coding + speed) — speed is a tiebreaker, not a deal axis, and
adding axes inflates the frontier toward "everything is Pareto-optimal".

### 3.2 Effective price

Reuse the existing `MergedModel._effective_blended_price()` logic verbatim
(`src/models.py:89-103`). It already mirrors the frontend `effPrice()`
(`templates/index.html:400-407`). Do not change it here (size-tier / price
mechanics are owned by spec 05).

### 3.3 Missing data handling

- A model missing price **or** the view's quality axis is **not a frontier
  candidate**: `is_frontier_<view> = false`, `frontier_rank_<view> = null`,
  `frontier_score_<view> = null`. It still appears in the full table (the table
  already filters per-view on `intelligence_score` / arena fields).
- Price `<= 0` is treated as missing (defensive; 0 such models today, but guards
  against a bad scrape producing a `$0` corner that would break `log10`).

### 3.4 Near-duplicate frontier points

Distinct model *variants* frequently sit at nearly identical (price, quality).
Two frontier members are **near-duplicates** when
`abs(log10(p_A) - log10(p_B)) < 0.02` (~4.7% price) **and**
`abs(q_A - q_B) < QDUP` where `QDUP = 0.5` for intelligence (0–60 scale) and
`QDUP = 5` for Arena Elo. Among a near-duplicate cluster keep exactly one
representative: lowest price, then highest quality, then highest
`composite_deal_score` (view-appropriate), then name (stable). Dropped members
get `is_frontier_<view> = false` but retain their other fields. This prevents
three "mimo-v2-flash (thinking/non-thinking/…)" variants from occupying three
frontier slots and crowding the hero + both runners-up.

### 3.5 Hero selection — the knee, not the corner

All frontier points are Pareto-optimal, so "best deal" needs a tiebreak that is
**not** the ratio (which always picks the cheap corner). Use the **knee** of the
frontier via a log-price-normalized elbow score:

For the frontier set of a view, let `x = log10(price)`, `y = quality`. Min-max
normalize both across the frontier members to `x', y' ∈ [0,1]`. Define

```
frontier_score = y' - x'
```

Higher = more quality per order-of-magnitude of dollars. The hero is
`argmax(frontier_score)`; runners-up are the next-highest `frontier_score`
members (see §3.6). Both endpoints self-cancel (`x'=0,y'≈0` for the cheap-dumb
corner; `x'=1,y'=1` for the expensive-smart corner → both ≈ 0), so the knee — a
genuinely balanced model — wins.

**Validated on today's data** (intelligence view): this picks
**DeepSeek V4 Flash** (intel 40, $0.13, `frontier_score` +0.281) as hero, with
Tencent Hy3 (+0.194) and DeepSeek V4 Pro (+0.163) as runners-up. The degenerate
Ling-2.6-flash drops to `frontier_score` ≈ 0.000. This is the intended fix.

Rationale for **log** price: prices span $0.0167–$300 (4+ orders of magnitude);
linear normalization would put 95% of models in the bottom 6% of the x-axis and
make the knee meaningless. Log makes "10x cheaper" a constant step, matching how
buyers reason about cost.

`frontier_score` is min-max over the *frontier only* (≤ ~15 points), so it is
stable day-to-day and immune to the full-pool outlier squashing that breaks
`composite_deal_score`. Note this is a *ranking* value within a day, not a
cross-day-comparable score (spec 05 owns any cross-day scaling).

### 3.6 Runners-up and frontier ordering

- `hero` = frontier member with max `frontier_score`.
- `runners_up` = next 3 frontier members by descending `frontier_score`,
  **after** near-duplicate collapse, excluding the hero.
- `frontier_rank_<view>` = 1-based rank by descending `frontier_score` (hero=1).
- The frontier is *also* serialized in natural (price-ascending) order for the
  chart (see §7) — ordering for the chart is by price; ordering for hero/runners
  is by `frontier_score`.

### 3.7 Keep `composite_deal_score`?

**Keep it**, demoted to a secondary role:

- It remains the **table's "Deal Score" column** and the table's default sort
  (the frontend table sort is independent of hero selection).
- It is the **final tiebreak** inside a near-duplicate cluster (§3.4).
- It is **no longer** what selects the hero/runners-up.

We keep it because ripping it out would churn the table, the Arena view, and 60
archives' worth of schema. Spec 05 is expected to fix its normalization/scaling;
this spec only stops *hero selection* from depending on it. The two specs touch
`rank_models`/`compute_top_models` — coordinate (see §9).

### 3.8 Size-tier interaction

`compute_top_models` currently restricts the hero pool to
`size_tier == "frontier"` (`src/scorer.py:175`) — i.e. proprietary/closed
models. Under the new design we **drop that restriction for frontier
membership**: the Pareto frontier is computed over *all* candidates (a cheap
open-weight model absolutely can be the best deal — that's the point). The
`size_tier` value is retained on each model for the table's size filter, but it
no longer gates who can be hero. (Spec 05 fixes size-tier misclassification;
until then, mixing tiers on the frontier is desired, not a bug.)

---

## 4. Exact changes per file

### 4.1 `src/models.py`

Add frontier fields to `MergedModel` (all default `None`/`False`, so old
archives that lack them deserialize fine and new archives stay
backward-superset):

```python
class MergedModel(RawModelRecord):
    # ... existing fields ...
    # --- Pareto frontier (intelligence-vs-price, "aa" view) ---
    is_frontier_intel: bool = False
    frontier_rank_intel: Optional[int] = None      # 1 = hero
    frontier_score_intel: Optional[float] = None    # knee score, y'-x'
    # --- Pareto frontier (arena/coding-vs-price, "arena" view) ---
    is_frontier_arena: bool = False
    frontier_rank_arena: Optional[int] = None
    frontier_score_arena: Optional[float] = None
```

Keep `value_score`, `coding_value`, `composite_deal_score`,
`arena_composite_score` unchanged (still computed, still used by table). No
change to the `model_validator` blocks or `_effective_blended_price`.

### 4.2 `src/scorer.py`

Add a frontier module-level helper set and rewrite `compute_top_models` to be
frontier-driven. `rank_models` is unchanged except it should call the new
frontier annotator (or `main.py` calls it — see §4.3; recommend calling it from
`main.py` right after `rank_models` for a clean seam).

New functions:

```python
import math

# Near-duplicate thresholds
_DUP_LOG_PRICE = 0.02      # ~4.7% price
_DUP_Q_INTEL = 0.5         # intelligence 0-60 scale
_DUP_Q_ARENA = 5.0         # Arena Elo scale


def _effective_price(m: MergedModel) -> float | None:
    p = m._effective_blended_price()
    return p if (p is not None and p > 0) else None


def _arena_quality(m: MergedModel) -> float | None:
    """Arena view quality: coding_index first (never present today), then
    arena_elo, then arena_coding_elo."""
    if m.coding_index is not None:
        return m.coding_index
    if m.arena_elo is not None:
        return m.arena_elo
    return m.arena_coding_elo


def _pareto_frontier(
    candidates: list[tuple[MergedModel, float, float]]  # (model, price, quality)
) -> list[tuple[MergedModel, float, float]]:
    """Return the subset not dominated by any other (cheaper-or-equal AND
    smarter-or-equal, strict in one)."""
    front = []
    for m, p, q in candidates:
        dominated = any(
            (p2 <= p and q2 >= q) and (p2 < p or q2 > q)
            for m2, p2, q2 in candidates
            if m2 is not m
        )
        if not dominated:
            front.append((m, p, q))
    return front


def _collapse_near_duplicates(front, q_dup, score_attr):
    """Keep one representative per near-duplicate (log-price, quality) cluster.
    Preference: lower price, higher quality, higher composite, then name."""
    front = sorted(front, key=lambda t: t[1])  # by price asc
    kept = []
    for m, p, q in front:
        dup = None
        for km, kp, kq in kept:
            if abs(math.log10(p) - math.log10(kp)) < _DUP_LOG_PRICE and abs(q - kq) < q_dup:
                dup = (km, kp, kq)
                break
        if dup is None:
            kept.append((m, p, q))
        else:
            km, kp, kq = dup
            challenger_better = (
                p < kp or
                (p == kp and q > kq) or
                (p == kp and q == kq and (getattr(m, score_attr) or 0) > (getattr(km, score_attr) or 0))
            )
            if challenger_better:
                kept.remove(dup)
                kept.append((m, p, q))
    return kept


def _knee_scores(front) -> dict[int, float]:
    """frontier_score = y' - x' with x = log10(price) min-maxed, y = quality
    min-maxed, over the frontier set. Returns {id(model): score}."""
    if not front:
        return {}
    xs = [math.log10(p) for _, p, _ in front]
    ys = [q for _, _, q in front]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    out = {}
    for (m, p, q), x in zip(front, xs):
        xn = 0.0 if xmax == xmin else (x - xmin) / (xmax - xmin)
        yn = 0.0 if ymax == ymin else (q - ymin) / (ymax - ymin)
        out[id(m)] = round(yn - xn, 4)
    return out


def annotate_frontier(models: list[MergedModel]) -> None:
    """Compute both Pareto frontiers and set is_frontier_*/frontier_rank_*/
    frontier_score_* on every model (mutates in place)."""
    specs = [
        ("intel", lambda m: m.intelligence_score, _DUP_Q_INTEL, "composite_deal_score"),
        ("arena", _arena_quality, _DUP_Q_ARENA, "arena_composite_score"),
    ]
    for view, qfn, q_dup, score_attr in specs:
        candidates = []
        for m in models:
            p = _effective_price(m)
            q = qfn(m)
            if p is not None and q is not None:
                candidates.append((m, p, q))
        front = _pareto_frontier(candidates)
        front = _collapse_near_duplicates(front, q_dup, score_attr)
        scores = _knee_scores(front)
        ranked = sorted(front, key=lambda t: -scores[id(t[0])])
        for rank, (m, p, q) in enumerate(ranked, start=1):
            setattr(m, f"is_frontier_{view}", True)
            setattr(m, f"frontier_rank_{view}", rank)
            setattr(m, f"frontier_score_{view}", scores[id(m)])


def compute_top_models(models: list[MergedModel], view: str = "aa") -> list[MergedModel]:
    """Return frontier members ordered by frontier_score desc (hero first).

    Falls back to the old composite ordering only if the frontier is empty
    (e.g. a day with no priced+scored models)."""
    v = "arena" if view == "arena" else "intel"
    rank_attr = f"frontier_rank_{v}"
    on_front = [m for m in models if getattr(m, rank_attr) is not None]
    if on_front:
        return sorted(on_front, key=lambda m: getattr(m, rank_attr))
    # Fallback: preserve previous behaviour so the page never goes empty.
    score_field = "arena_composite_score" if view == "arena" else "composite_deal_score"
    pool = [m for m in models if getattr(m, score_field) is not None]
    return sorted(pool, key=lambda m: -(getattr(m, score_field) or 0))
```

Notes:
- Map the caller's `view="aa"` to internal `"intel"`; keep the public `view`
  argument values (`"aa"`/`"arena"`) unchanged so `main.py` needs no edit there.
- `rank_models` keeps computing `composite_deal_score`/`arena_composite_score`
  unchanged (needed by the table + tiebreaks).

### 4.3 `main.py`

Insert one call after `rank_models` (line 72), before `compute_top_models`:

```python
ranked = rank_models(merged)
annotate_frontier(ranked)                 # NEW: sets frontier fields in place

top_models = compute_top_models(ranked, "aa")
arena_top = compute_top_models(ranked, "arena")
```

Update the import on line 16:

```python
from src.scorer import annotate_frontier, annotate_sizes, compute_top_models, rank_models
```

The log line at 76-82 still works (`composite_deal_score` still populated). No
other `main.py` change needed — `write_archives(...)` already receives
`top_models`/`arena_top`, whose `[0]`/`[1:4]` become hero/runners_up.

### 4.4 `src/writer.py`

No signature change. `hero`/`runners_up`/`arena_hero`/`arena_runners_up` are
still `top_models[0]`/`[1:4]` etc. (lines 27-30) — but now those lists are
frontier-ordered by `frontier_score`, so the hero is the knee model and the
runners-up are its frontier neighbors. The new `MergedModel` fields serialize
automatically via `payload.model_dump(mode="json")`.

Optional (recommended): also emit an explicit frontier array for the chart, to
avoid the frontend re-deriving domination client-side. Add to `ArchivePayload`
(§6) and populate in `write_archives`:

```python
frontier_intel = [m for m in models if m.is_frontier_intel]
frontier_arena = [m for m in models if m.is_frontier_arena]
# sort by price asc for the chart step-line
frontier_intel.sort(key=lambda m: m._effective_blended_price() or 0)
frontier_arena.sort(key=lambda m: m._effective_blended_price() or 0)
```

### 4.5 `src/insights.py`

No signature change (`generate_insights(top_models[:10], api_key)`), but the
top-10 it receives are now frontier knee-ordered, which is strictly better input
for the LLM. Optionally update the prompt to say "these are Pareto-frontier best
deals" — not required for correctness.

---

## 5. Archive schema changes & backward compatibility

### 5.1 New per-model fields (§4.1)

`is_frontier_intel`, `frontier_rank_intel`, `frontier_score_intel`,
`is_frontier_arena`, `frontier_rank_arena`, `frontier_score_arena`.

### 5.2 New top-level fields (optional, for chart) — see §6.

### 5.3 Backward compatibility with existing archives

Measured: **62 archive files** in `docs/archive/` (2026-05-24 → today). The
oldest (`2026-05-24.json`) has top-level keys
`[date, generated_at, insights, total_models, models, hero, runners_up]` — **no**
`arena_hero`, `arena_runners_up`, and its model objects lack `model_id`,
`arena_*`, `openrouter_*`, and every field this spec adds.

Compatibility is one-directional and must hold on **read**:

- **Old archives are never rewritten.** They will not contain the new fields.
  The frontend must treat all new fields as optional: `is_frontier_* == true`
  only when present; `frontier_score_* == null/undefined` → fall back to the
  old client-side "top-3 visible rows" hero derivation (which still works — see
  §7). The new chart section must **hide itself** when no frontier data is
  present in the loaded archive.
- **Pydantic read side:** every new field has a default, so
  `MergedModel(**old_json)` and `ArchivePayload(**old_json)` still validate
  (extra-missing = default). No migration/backfill script is required or
  desired.
- Do **not** rename or remove any existing field (`value_score`,
  `composite_deal_score`, `hero`, `runners_up`, etc.) — 62 archives and the
  table depend on them.

No backfill: historical heroes stay as-recorded (they were written with the old
logic; rewriting them would be dishonest and pointless).

---

## 6. `ArchivePayload` additions (optional chart payload)

If adopting the explicit frontier arrays (recommended, §4.4/§7):

```python
class ArchivePayload(BaseModel):
    # ... existing fields ...
    frontier_intel: list["MergedModel"] = Field(default_factory=list)
    frontier_arena: list["MergedModel"] = Field(default_factory=list)
```

Both default to `[]`, so old archives (which lack them) load as empty and the
chart hides. The full `models[]` array already carries the `is_frontier_*`
flags, so a frontend could instead derive the frontier by filtering
`models[]` — the explicit arrays are purely a convenience/perf choice. Pick one
and document it in the frontend section; this spec assumes the explicit arrays
exist but degrades to filtering if they are `[]`.

---

## 7. Frontend changes (`templates/index.html`)

### 7.1 How the frontend consumes data today (important)

`renderPage()` (line 556) does **not** use the JSON `hero`/`runners_up` fields.
It derives the scorecard client-side from the **top 3 visible table rows**
(frontier tier + Q3 intel filter + deal-score sort) at lines 596-609. So today
the JSON hero fields are essentially unused by the page.

Two options; **pick option B**:

- **Option A (minimal):** leave the client-side top-3 derivation, but change the
  table's default sort to `frontier_score_<view>` (desc) restricted to
  `is_frontier_<view>` rows. Smaller diff, but couples hero to table filters
  (the Q3 slider could still hide the true knee).
- **Option B (recommended):** make the hero/runners-up read the JSON
  `hero`/`runners_up` (and `arena_hero`/`arena_runners_up`) fields directly when
  present, falling back to the current top-3 derivation only when they are
  absent (old archives). This decouples the hero from table UI state and makes
  the page show exactly the knee the pipeline computed.

```js
// in renderPage(), replacing the IIFE at 599-609:
var heroField    = isArena ? data.arena_hero        : data.hero;
var runnersField = isArena ? data.arena_runners_up  : data.runners_up;
if (heroField) {
  renderHero(heroField, maxIntel, maxElo);
  renderRunnersUp((runnersField || []).slice(0, 3), maxIntel, maxElo);
} else {
  /* legacy fallback: existing top-3-visible-rows derivation */
}
```

`renderHero`/`renderRunnersUp` need **no change** — they already read
`value_score`, `composite_deal_score`, `arena_*`, and `effPrice(model)`.

### 7.2 New frontier chart

Add a section between the runners-up (line 105) and insights (line 107):

```html
<section id="frontier-section" class="hidden">
  <p class="text-xs uppercase tracking-widest text-indigo-400 mb-3 font-semibold">
    Price vs Intelligence Frontier</p>
  <div id="frontier-chart"></div>   <!-- inline SVG rendered by JS -->
</section>
```

**Chart type:** log-x scatter + step line. X = effective price (log scale,
$0.01–$300 domain covers today's $0.0167–$300). Y = quality
(intelligence 0–60 for aa view; Arena Elo ~1150–1700 for arena view). Plot **all
view-eligible models** as faint dots; overlay the **frontier members** as bright
dots connected by a descending step line (price-ascending); mark the **hero**
(`frontier_rank_* == 1`) with a ring + label.

**Exact data the chart JS needs per point** (all already in `models[]`):

| Field | Source | Use |
|---|---|---|
| `effPrice(m)` | existing helper (line 400) | x (log) |
| `intelligence_score` / `arena_quality` | model | y |
| `is_frontier_intel` / `is_frontier_arena` | new | bright vs faint |
| `frontier_rank_intel` / `frontier_rank_arena` | new | hero ring (==1), label |
| `name`, `creator` | model | tooltip/label |
| `frontier_score_*` | new | tooltip ("deal knee score") |

Frontier line points: either read `data.frontier_intel`/`data.frontier_arena`
(if present) or `models.filter(m => m.is_frontier_<view>)` sorted by
`effPrice`. **Hide `#frontier-section`** when the loaded archive yields zero
frontier points (old archives) — check `getFrontier(view).length > 0`.

View toggle (aa/arena) must re-render the chart with the correct axis + quality
field, same as it re-renders the table.

Keep the implementation as dependency-free inline SVG (the page currently ships
no chart lib; do not add one — matches "don't add unrequested dependencies").
This spec defines the data contract, not the SVG code.

### 7.3 Methodology copy (lines 172-179)

Replace the "Deal Score" / "Coding Value" lines to describe the frontier:

- Add: *"Best Deal = the knee of the price-vs-intelligence Pareto frontier
  (nothing cheaper is smarter). Ranked by intelligence gained per 10x of
  spend."*
- Fix the dead line: `Coding Value = Coding Index / Price` — `coding_index` is
  currently never published; either remove it or note the coding/Arena view uses
  `arena_coding_elo`.

---

## 8. Test plan

No test framework is currently installed (only stdlib deps in
`pyproject.toml`). Add `pytest` as a dev dependency and a `tests/` package.

**Unit — `tests/test_frontier.py` (`src/scorer.py`):**

1. `_pareto_frontier`: hand-built set where B dominates C → C excluded; ties on
   both axes both kept (neither strictly dominates).
2. `annotate_frontier` on a fixture reproducing today's 12-point intel frontier
   → asserts the 12 expected members are flagged `is_frontier_intel`, and
   `frontier_rank_intel == 1` is **DeepSeek V4 Flash** (intel 40, $0.1307),
   **not** Ling-2.6-flash. This is the regression lock for the bug.
3. `_knee_scores`: cheap-dumb and expensive-smart endpoints both get
   `frontier_score ≈ 0`; the knee is strictly positive and maximal.
4. `_collapse_near_duplicates`: two points within `_DUP_LOG_PRICE` and `QDUP`
   collapse to one (the cheaper/higher-quality/higher-composite rep); the three
   `mimo-v2-flash` variants in real arena data collapse appropriately.
5. Missing-data: model with price but no `intelligence_score` → not on intel
   frontier, on arena frontier iff it has `arena_coding_elo`.
6. `compute_top_models("aa")` returns frontier order (rank 1 first);
   `compute_top_models` fallback path fires (and matches old ordering) when no
   model has a frontier rank.
7. Guard: price `<= 0` treated as missing (no `log10` domain error).

**Integration — `tests/test_pipeline.py`:**

8. Load `docs/data/latest.json` as `RawModelRecord`s → `merge` → `annotate_sizes`
   → `rank_models` → `annotate_frontier` → assert exactly the measured counts
   (51 intel candidates → 12 frontier; 93 arena candidates → 8 frontier, pre-
   dedupe) and that hero ≠ Ling-2.6-flash.
9. `ArchivePayload(**old_archive_json)` for `docs/archive/2026-05-24.json`
   validates (backward-compat: new fields default, missing `arena_hero` ok).
10. Round-trip: `write_archives` output re-parses as `ArchivePayload` and every
    model carries the six new fields.

**Frontend (manual / smoke):** open `docs/index.html` against a freshly
generated `latest.json`; verify hero = knee model, runners-up = frontier
neighbors, chart renders with hero ringed; switch to Arena view → chart + hero
update; select an **old** archive date → chart hides, hero falls back to legacy
derivation without error.

---

## 9. Dependencies / integration notes

- **Spec 05 (mechanics-fixes)** overlaps in three places — coordinate:
  1. *Score scaling:* 05 rescales `composite_deal_score` for cross-day
     stability. This spec no longer uses that score for hero selection, so 05 is
     free to change it; but this spec still uses it as the **near-duplicate
     tiebreak** (§3.4) and the table column — 05 must keep it monotonic/non-null
     for scored models or the tiebreak silently no-ops (still safe: falls to
     name order).
  2. *Size-tier fixes:* this spec **removes** the `size_tier == "frontier"` gate
     from `compute_top_models` (§3.8), so 05's size-tier reclassification no
     longer affects who can be hero — it only affects the table's size filter.
     Land order doesn't matter, but re-verify the hero test (#2) after 05 lands.
  3. *Uptime gating:* if 05 excludes low-uptime models from "deals", apply that
     filter to the **frontier candidate set** (before `_pareto_frontier`) so a
     flaky-but-cheap model can't claim the knee. Add the uptime predicate inside
     `annotate_frontier`'s candidate loop when 05 defines the threshold.
- **`coding_index` is dead** in the current pipeline (0/523). The coding/arena
  frontier relies entirely on `arena_coding_elo`. If a future scraper change
  populates `coding_index`, `_arena_quality` already prefers it — no code change
  needed, but re-check dedupe thresholds (`coding_index` is a 0–100-ish scale,
  not Elo).
- **No new runtime dependencies.** `math` is stdlib; chart is inline SVG.
  `pytest` is dev-only.
- **Pydantic:** all new fields have defaults → additive, non-breaking for the 62
  existing archives.
