# 05 — Scoring Mechanics Fixes (foundational)

Status: spec / ready to implement
Owner primitives for: 01-pareto-frontier, 02, 03, 04 (they consume the stable
primitives defined in **§8 Dependencies / integration**).

This spec fixes the five foundational defects in the scoring layer
(`src/models.py`, `src/scorer.py`) so that:

- a model's score means the same thing today, yesterday, and next month
  (archives become usable for trends),
- a single ultra-cheap outlier no longer squashes everyone toward 0,
- the "top deals" list is actually the best deals (not "models whose names
  happen to lack a parameter count"),
- missing data is neither rewarded nor punished,
- uptime data (already collected, never used) gates out models that are down.

All thresholds below are grounded in measurements from `docs/data/latest.json`
(2026-07-24, 523 models) plus archives `2026-06-23.json` (745 models) and
`2026-05-25.json` (223 models). Measurements are quoted inline.

---

## 1. Problem (all five)

### P1 — Unstable normalization
`rank_models()` in `src/scorer.py` min-max normalizes `value_score`,
`coding_value`, `output_speed_tps` (and the arena equivalents) against the
min/max of **whatever models were scraped that day** (`_build_range` +
`_normalise`, lines 78-106). Two consequences:

- **Not comparable across days.** The endpoints move every day, so the same
  raw value maps to a different normalized score on different days. Measured
  `value_score` upper tail by day:

  | archive      | n  | p75   | p90   | max    |
  |--------------|----|-------|-------|--------|
  | 2026-05-25   | 146| 185.7 | 314.3 | 1100.0 |
  | 2026-06-23   | 46 | 120.0 | 211.1 | 666.7  |
  | 2026-07-24   | 51 | 75.0  | 180.0 | 840.0  |

  A model with `value_score = 180` normalizes to ≈0.16 on 2026-05-25 but ≈0.90
  on 2026-07-24. Archived `composite_deal_score` values are therefore not
  trendable — spec 01/02 cannot chart them over time as-is.

- **Outlier squash.** Today `inclusionAI: Ling-2.6-flash` has
  `value_score = 840`; the runner-up is `306`. Min-max puts Ling at 1.0 and
  compresses the rest of the field toward 0. Resulting `composite_deal_score`
  top-10: `[0.823, 0.332, 0.312, 0.259, 0.246, 0.198, 0.177, 0.145, 0.143,
  0.137]` — one model at 0.82, everyone else < 0.34.

### P2 — Degenerate ratio
`MergedModel.compute_raw_scores` (`src/models.py:67-76`) sets
`value_score = intelligence_score / blended_price`. As price → 0 this explodes.
The Ling outlier: `intelligence_score = 14` (a weak model, 14/60), effective
price `= (2·0.01 + 0.03)/3 = 0.0167` → `value_score = 840`, beating
`DeepSeek V4 Flash` (`intelligence 40`, price 0.131 → 306). A near-worthless but
near-free model wins purely on the ratio's tail. Measured effective-price
distribution: `min=0.0167, p5=0.083, p10=0.150, median=1.00, max=300`; only
**4 of 357** priced models are below `$0.05`.

### P3 — Size-tier mislabel
`_size_tier()` (`src/scorer.py:55-71`) returns **`"frontier"` when there is no
size info at all**. `compute_top_models()` (line 172-176) then filters to
`size_tier == "frontier"`. Measured in `latest.json`: 305 models are
`"frontier"` and **0 of them have a parsed `param_billions`** — the tier is
literally "name contains no `NNB` token". The top-deals / hero / insights set is
thus "models whose names lack a parameter count", and 12 genuinely-scored models
(51 with a composite − 39 frontier-with-composite) are silently excluded from
top deals.

### P4 — Weight redistribution bias
In `rank_models()` (lines 118-122, mirrored 148-152) when `coding_value` is
missing the code does:

```python
parts = [(W_VALUE + W_CODING, _normalise(model.value_score, *vs_range))]
```

This **reassigns coding's 0.30 weight to value AND drops the speed leg
entirely** (it overwrites `parts`, discarding the speed tuple appended later
only if it comes after — here speed is appended after, but the branch replaces
`parts` so a model with no coding gets 0.8·value + 0.2·speed while a model
*with* coding is scored on all three). Net effect: a model **missing** coding
data is scored `0.8·value (+0.2·speed)`; a model **with poor** coding data is
dragged down by its low coding leg. Missing data is rewarded, present data is
punished. Note: coding data is currently **100% absent** for the AA view
(`coding_index` n=0 in all three archives), so this branch fires for every
scored model today.

### P5 — Unused uptime data
`uptime_30m` / `uptime_1d` are scraped, merged (`src/merger.py:51-52`), stored,
and displayed, but never affect ranking. A cheap model that is down is not a
deal. Measured `uptime_30m`: `n=248, min=92.65, p10=99.82, median=100`;
**only 1 model < 95%**, 7 < 99%, 186 at exactly 100. `uptime_1d`:
`n=297, min=93.24, 1 model < 95%`. AA-only models have no OpenRouter uptime
(gate must no-op when data is absent).

---

## 2. Design per fix

### D2 — Price floor (fixes P2, feeds P1)
Add a **price floor** `PRICE_FLOOR = 0.05` ($/1M tokens) inside
`_effective_blended_price()`. Rationale:

- Interpretability is preserved — `value_score` stays "intelligence per dollar"
  (a log transform of the *ratio* would make the number unreadable; the log
  belongs in normalization, §D1, not in the stored ratio).
- Only 4 of 357 priced models sit below `$0.05`, so the floor touches almost
  nothing except the degenerate tail.
- Effect on the outlier: Ling's `value_score` drops `840 → 14/0.05 = 280`,
  now **below** DeepSeek V4 Flash (306), which is the correct ordering (a 40/60
  model at $0.13 is a better deal than a 14/60 model at $0.017).

Floor is applied to the *effective* price used for every ratio (AA value, AA
coding, arena value, arena coding) so all ratios share one price definition.

### D1 — Normalization: log transform against pinned reference anchors
**Chosen approach: `clamp01( (log10(x) − log10(REF_LO)) / (log10(REF_HI) −
log10(REF_LO)) )` with `REF_LO`/`REF_HI` **hard-coded constants**, not
recomputed from the daily pool.**

Options evaluated:

| option | day-stable? | outlier-robust? | pool-independent? | notes |
|--------|-------------|-----------------|-------------------|-------|
| current daily min-max | ✗ | ✗ | ✗ | status quo, broken |
| fixed linear min-max (pinned min/max) | ✓ | ✗ | ✓ | value spans 3.5 orders of magnitude; linear crushes the low end |
| **log + pinned anchors + clamp** | ✓ | ✓ | ✓ | **chosen** |
| percentile vs pinned reference distribution | ✓ | ✓ | ✓ | equivalent stability but needs a maintained reference-distribution snapshot file; more moving parts |
| winsorize + daily min-max | ✓-ish | ✓ | ✗ | endpoints still drift with pool |

Rationale for log + pinned anchors:
- **Day-stable & pool-independent**: the anchors are constants in code, so a
  given raw value always maps to the same normalized score regardless of what
  else was scraped. This is the hard requirement.
- **Outlier-robust**: with the §D2 floor the value ratio is bounded at
  `intelligence_max(60)/0.05 = 1200`; observed floored max ≈ 306. `log10`
  compresses the tail, and `clamp01` caps anything beyond `REF_HI`. A future
  outlier at 5000 and one at 500 both map to 1.0 instead of squashing the field.
- **Transparent**: no external reference file to maintain (the reason percentile
  scaling was rejected — it needs a pinned distribution snapshot that itself
  drifts and must be versioned).

**Pinned anchors** (chosen from the measured distributions; see sanity table):

| metric | constant name | REF_LO | REF_HI | basis |
|--------|---------------|--------|--------|-------|
| AA value_score | `VS_REF` | 5 | 500 | p10≈8, median≈37, p90≈180, floored max≈306 |
| AA coding_value | `CV_REF` | 5 | 500 | same units as value (index/price); coverage currently 0, symmetric anchors |
| output_speed_tps | `SP_REF` | 20 | 1000 | min 14, p10 45, median 104, p90 251, max 958 |
| arena_value | `AV_REF` | 50 | 10000 | elo(~1000)/price; scale ~1000× the AA ratios |
| arena_coding_value | `AC_REF` | 50 | 10000 | median 704, p90 4251, max 18000 |

Sanity (VS_REF = (5, 500), floored value): `37→0.435`, `75→0.588`,
`180→0.778`, `280(Ling floored)→0.874`, `306(DeepSeek)→0.893`, `500→1.0`,
`840→1.0`. DeepSeek now edges Ling on the value leg (0.893 vs 0.874), matching
the intuition that a real 40/60 model is the better deal.

Anchors are **the exported primitive** other specs pin to (see §8). Changing an
anchor is a scoring-version bump (§6).

### D3 — `"unknown"` tier + deal-candidate selection (fixes P3)
- Rename the no-size-info fallback in `_size_tier()` from `"frontier"` to
  `"unknown"`. Update `MergedModel.size_tier` default (`src/models.py:65`) to
  `"unknown"`. The tier set becomes `nano | small | medium | large | unknown`.
  There is no reliable proprietary/closed flag in the source data, so
  "no parseable size" is honestly labeled "unknown", not "frontier".
- `compute_top_models()` **stops filtering on tier**. A "top deal" is any model
  that (a) has a non-None composite score for the requested view **and**
  (b) passes the uptime gate (§D5). This surfaces the 12 currently-excluded
  scored models and removes the accidental "names without a number" filter.

### D4 — Fair missing-data treatment: fixed weights + neutral imputation (fixes P4)
Remove both weight-redistribution `elif` branches. Replace with **fixed weights
and neutral (0.5) imputation** for any missing normalized component:

- Weights stay `W_VALUE=0.50, W_CODING=0.30, W_SPEED=0.20` (arena mirror
  unchanged).
- Compute a composite only if **at least one quality signal is present**
  (`value_score` or `coding_value` for AA; `arena_value` or
  `arena_coding_value` for arena) — unchanged gate, speed alone is not a deal.
- For each of the three legs: if the raw value is present, normalize it (§D1);
  if absent, substitute the **neutral normalized value `0.5`** (the midpoint /
  "average" model). Then `composite = 0.5·v + 0.3·c + 0.2·s` with no
  renormalization.

Rationale: a missing leg contributes exactly the field-average, so missing data
neither inflates a sibling leg (the old bug) nor penalizes the model. A model
that *reports* a poor coding value is scored on it, same scale as everyone —
no longer worse off than a model that simply omitted it. When a leg is 100%
absent across the pool (today's AA coding), imputing 0.5 for all adds a constant
and does not perturb ranking.

### D5 — Uptime gate (fixes P5)
`UPTIME_GATE_PCT = 95.0`. A model **fails** the gate if it has uptime data and
`best(uptime_30m, uptime_1d) < 95.0`. Models with no uptime data (AA-only) pass
(gate no-ops). Rationale from data: the 95% line excludes the single genuinely
degraded model (`uptime_30m` min 92.65) while retaining the 99.9%+ mass (only 1
model < 95% on either metric).

The gate is applied at **selection time** (`compute_top_models`, hero,
insights), **not** baked into `composite_deal_score`. Keeping the composite a
pure value/coding/speed number preserves its cross-day comparability (a stored
score doesn't silently change if uptime blips). Export `passes_uptime_gate(m)`
so the frontend and other specs can reuse the same rule.

---

## 3. Exact changes per file

### 3.1 `src/models.py`

Add the price floor and apply it everywhere the effective price is used.

```python
# module-level
PRICE_FLOOR = 0.05  # $/1M tokens; below this the intel/price ratio is degenerate

class MergedModel(RawModelRecord):
    ...
    size_tier: str = "unknown"   # was "frontier"  (nano|small|medium|large|unknown)

    def _effective_blended_price(self) -> Optional[float]:
        raw = self._raw_blended_price()
        if raw is None:
            return None
        return max(raw, PRICE_FLOOR)          # <-- apply floor

    def _raw_blended_price(self) -> Optional[float]:
        # (body identical to the current _effective_blended_price:
        #  price_blended -> 2:1 input/output -> price_input -> OR fallbacks)
        ...
```

`compute_raw_scores` / `compute_arena_raw_scores` are unchanged in shape — they
already call `_effective_blended_price()`, which now returns the floored price,
so `value_score`, `coding_value`, `arena_value`, `arena_coding_value` are all
computed against the floor automatically.

Optionally add a schema/version marker to `ArchivePayload` (see §6):

```python
class ArchivePayload(BaseModel):
    scoring_version: int = 2
    ...
```

### 3.2 `src/scorer.py`

**(a) Constants** — replace the arena-weight block region with the pinned
anchors and gate, keep the weights:

```python
W_VALUE, W_CODING, W_SPEED = 0.50, 0.30, 0.20
W_ARENA_VALUE, W_ARENA_CODING, W_ARENA_SPEED = 0.50, 0.30, 0.20

# Pinned log-normalization anchors (REF_LO, REF_HI). Day-stable, pool-independent.
# Changing any of these is a SCORING_VERSION bump.
VS_REF = (5.0, 500.0)     # AA value_score
CV_REF = (5.0, 500.0)     # AA coding_value
SP_REF = (20.0, 1000.0)   # output_speed_tps
AV_REF = (50.0, 10000.0)  # arena_value
AC_REF = (50.0, 10000.0)  # arena_coding_value

NEUTRAL = 0.5             # imputed normalized value for a missing leg
UPTIME_GATE_PCT = 95.0
SCORING_VERSION = 2
```

**(b) Normalization helper** — replace `_normalise` / `_build_range` /
`_safe_values` usage in the composite loop with a pinned log-normalizer:

```python
import math

def normalize_log(x: float | None, ref: tuple[float, float]) -> float:
    """Map x to [0,1] via log10 against pinned (lo, hi). Missing -> NEUTRAL."""
    if x is None:
        return NEUTRAL
    lo, hi = ref
    x = max(x, lo)                       # clamp low (also guards log10(0))
    v = (math.log10(x) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    return min(1.0, max(0.0, v))         # winsorize/clamp the high tail
```

`_build_range`, `_safe_values`, `_normalise` are no longer needed for the
composite (may be deleted).

**(c) `_size_tier`** — one-line change:

```python
    ...
    m = _SIZE_HINT_RE.search(clean)
    if m:
        return _HINT_TO_TIER[m.group(1).lower()]
    return "unknown"   # was "frontier": genuinely no size info
```

**(d) `rank_models`** — replace the per-model AA and arena composite blocks:

```python
def rank_models(models: list[MergedModel]) -> list[MergedModel]:
    if not models:
        return []

    for m in models:
        # AA composite: require >=1 quality signal (value or coding)
        if m.value_score is None and m.coding_value is None:
            m.composite_deal_score = None
        else:
            v = normalize_log(m.value_score,  VS_REF)   # imputes NEUTRAL if None
            c = normalize_log(m.coding_value, CV_REF)
            s = normalize_log(m.output_speed_tps, SP_REF)
            composite = W_VALUE * v + W_CODING * c + W_SPEED * s
            m.composite_deal_score = round(composite, 4)

        # Arena composite: require >=1 arena quality signal
        if m.arena_value is None and m.arena_coding_value is None:
            m.arena_composite_score = None
        else:
            av = normalize_log(m.arena_value,        AV_REF)
            ac = normalize_log(m.arena_coding_value, AC_REF)
            s  = normalize_log(m.output_speed_tps,   SP_REF)
            arena = W_ARENA_VALUE * av + W_ARENA_CODING * ac + W_ARENA_SPEED * s
            m.arena_composite_score = round(arena, 4)

    return sorted(
        models,
        key=lambda m: (
            m.composite_deal_score is None,
            -(m.composite_deal_score or 0),
            -(m.intelligence_score or 0),
        ),
    )
```

Note: neutral imputation means a leg that is 100% absent across the pool adds a
constant `weight·0.5` to every model (e.g. today's AA coding adds 0.15 to all)
and does not change ordering.

**(e) Uptime gate + `compute_top_models`**:

```python
def passes_uptime_gate(m: MergedModel) -> bool:
    """True unless the model has uptime data below the gate."""
    vals = [u for u in (m.uptime_30m, m.uptime_1d) if u is not None]
    if not vals:
        return True                      # no data (AA-only) -> do not gate
    return max(vals) >= UPTIME_GATE_PCT

def is_deal_candidate(m: MergedModel, view: str = "aa") -> bool:
    field = "arena_composite_score" if view == "arena" else "composite_deal_score"
    return getattr(m, field) is not None and passes_uptime_gate(m)

def compute_top_models(models: list[MergedModel], view: str = "aa") -> list[MergedModel]:
    """Top deals = scored + up, sorted by the view's composite (no tier filter)."""
    field = "arena_composite_score" if view == "arena" else "composite_deal_score"
    candidates = [m for m in models if is_deal_candidate(m, view)]
    return sorted(candidates, key=lambda m: -(getattr(m, field) or 0))
```

### 3.3 `main.py`
No structural change. `compute_top_models(ranked, "aa"/"arena")` and the
`write_archives(...)` call are unchanged. (If `scoring_version` is added to
`ArchivePayload`, no call-site change is needed — it defaults.)

### 3.4 `src/writer.py`
No change required. If `scoring_version` is added to `ArchivePayload` it is
serialized automatically by `model_dump`.

---

## 4. Archive schema impact

Stored score fields **change meaning** (not shape):

- `value_score`, `coding_value`, `arena_value`, `arena_coding_value` — now
  computed against the **floored** price. Only the ≤4 sub-$0.05 models per day
  change; everything else is identical. Units unchanged (intel/$, elo/$).
- `composite_deal_score`, `arena_composite_score` — now **log-normalized against
  pinned anchors** instead of daily min-max. Still `[0,1]`, but now
  **comparable across days** and no longer outlier-squashed.
- `size_tier` — new value `"unknown"` replaces the misused `"frontier"`.

**Migration / annotation strategy:**
1. Add `scoring_version: int = 2` to `ArchivePayload` (v1 = legacy daily min-max
   / "frontier" fallback; v2 = this spec). Consumers (frontend trend charts,
   specs 01/02) must branch on it and only trend v2+ archives.
2. Old archives keep their v1 numbers (no in-place mutation on read). Provide a
   one-off **backfill script** `scripts/rescore_archives.py` that, for each
   `docs/archive/*.json`, reconstructs `MergedModel`s from the stored raw fields
   (raw `intelligence_score`, prices, `arena_*_elo`, `output_speed_tps`,
   `uptime_*` are all persisted), re-runs `annotate_sizes` + `rank_models`, and
   rewrites the file with `scoring_version = 2`. This is safe and idempotent
   because the raw inputs are stored; run it once to make the full history
   trendable, or leave history as v1 and only trend forward — either is
   acceptable, document the choice in the manifest.

---

## 5. Frontend impact (`templates/index.html`)

Score fields keep the same names/shape, so the fetch/render plumbing is
unaffected. Required edits:

1. **Tier chip rename** (P3). The "Frontier" chip uses `data-tier="frontier"`
   and is the **default active** filter; after the rename it would match nothing.
   Change the chip (lines ~128-132) and the default `activeTier`
   (line ~677 `var activeTier = 'frontier';`) and the two JS fallbacks
   `r.dataset.tier || 'frontier'` (lines ~470, ~714, ~729) and
   `m.size_tier || 'frontier'` (line ~470) from `'frontier'` to `'unknown'`.
   Relabel the chip text/tooltip from "Frontier / proprietary" to
   "Unknown size" (or "Other"). *(If a parallel spec reworks the tier UI, it
   owns this; otherwise do it here so the default filter isn't empty.)*

2. **Cosmetic thresholds** — no scale-breaking change, but note:
   - `dealClass` buckets (`> 0.7` indigo, `> 0.4` gray, lines ~453-454 /
     ~451): with stable log-normalization more mid-field models legitimately
     exceed 0.4 (the neutral-imputation constant alone puts a value-only model
     near `0.5·v + 0.15 + 0.1`). This is correct — the old squash was the bug —
     but expect more colored rows. No code change required; optionally retune.
   - `valueClass` / `avClass` (`> 100`/`> 30`, lines ~502-503 / ~518-519) read
     the raw `value_score` (units unchanged), so they still work; the only
     shift is ultra-cheap models drop out of the `>100` bucket (Ling 840→280,
     still >100 anyway).

3. **Methodology copy** (lines ~175-178) — update the "Deal Score" sentence to
   describe the new mechanics, e.g.: *"Deal Score = weighted composite (50%
   value, 30% coding, 20% speed), each leg log-normalized against fixed
   reference ranges so scores are comparable across days; value uses a $0.05/1M
   price floor; models below 95% uptime are excluded from top deals."*

The hero/runners-up are already re-derived client-side from the visible table
rows (`renderPage` IIFE, lines ~599-609), so once the tier default is fixed they
follow automatically; the payload `hero`/`runners_up` (now uptime-gated,
tier-unfiltered) feed only the insights step and any non-JS consumer.

---

## 6. Versioning

- `SCORING_VERSION = 2` in `src/scorer.py`; mirror into
  `ArchivePayload.scoring_version` (default 2).
- Any change to a pinned anchor (`VS_REF`, `SP_REF`, …), a weight, `PRICE_FLOOR`,
  `NEUTRAL`, or `UPTIME_GATE_PCT` **must** bump `SCORING_VERSION` — that is the
  contract that keeps archives interpretable.

---

## 7. Test plan

Unit (`tests/test_scorer.py`, add):

1. `normalize_log`: `normalize_log(5, (5,500)) == 0.0`;
   `normalize_log(500, (5,500)) == 1.0`; `normalize_log(37,(5,500)) ≈ 0.435`
   (±0.005); `normalize_log(5000,(5,500)) == 1.0` (clamp);
   `normalize_log(1,(5,500)) == 0.0` (low clamp); `normalize_log(None,ref)==0.5`.
2. **Day-stability**: build two model pools with different outlier sets but one
   shared model at a fixed `value_score`; assert its `composite_deal_score` is
   identical in both. (This is the regression that P1 must fix.)
3. **Outlier robustness**: pool with one model at `value_score=5000` and one at
   `500`; assert both value legs clamp to 1.0 and the rest of the field is not
   compressed (a `value_score=180` model stays ≈0.78, not ≈0.03).
4. **Price floor**: model `intelligence=14, or_in=0.01, or_out=0.03` →
   `value_score == round(14/0.05,4) == 280.0`.
5. **Neutral imputation**: model with `value_score` present, `coding_value` and
   speed None → `composite == round(0.5·v + 0.3·0.5 + 0.2·0.5, 4)`; assert a
   coding-missing model is not boosted above an otherwise-identical model that
   reports coding at the neutral point.
6. **Tier**: `_size_tier(None, "GPT-5.5")` == `"unknown"`;
   `_size_tier(None, "Qwen 7B")` via `_extract_param_billions` path == `"small"`.
7. **Uptime gate**: `passes_uptime_gate` True when both None; True at 95.0;
   False at 92.65; True when `uptime_1d=99, uptime_30m=None`.
8. **compute_top_models**: a scored model with `size_tier="large"` now appears;
   a scored model at `uptime_30m=90` is excluded.

Regression on real archives (add `tests/test_rescore_regression.py` or a manual
`scripts/rescore_archives.py --check`):

9. Re-run `annotate_sizes` + `rank_models` + `compute_top_models` on
   `docs/archive/2026-07-24.json` (reconstructed from raw fields) and assert:
   - top-1 AA deal is `DeepSeek V4 Flash`, **not** `inclusionAI: Ling-2.6-flash`
     (the floor + log-norm correction — DeepSeek value leg 0.893 > Ling 0.874);
   - the top-10 `composite_deal_score` spread is no longer one-model-dominated
     (2nd/1st ratio > 0.6, versus the current 0.332/0.823 ≈ 0.40);
   - every returned top model passes `passes_uptime_gate`.
10. Cross-day comparability check: pick a model present in both
    `2026-06-23.json` and `2026-07-24.json` with an unchanged raw
    `value_score`; assert its recomputed `composite_deal_score` matches within
    rounding across the two days.

Sanity: run `python main.py` offline path is not required — the scorer is pure;
tests operate on reconstructed `MergedModel`s.

---

## 8. Dependencies / integration — stable primitives exported to specs 01–04

These are the frozen contracts other specs build on. Do not change without a
`SCORING_VERSION` bump.

- **`PRICE_FLOOR = 0.05`** and **`MergedModel._effective_blended_price()`** — the
  single floored price definition. Spec 01 (Pareto frontier) MUST use this same
  effective price for its x-axis so its frontier and this composite agree.
- **`value_score` / `coding_value` / `arena_value` / `arena_coding_value`** —
  floored intel/elo-per-dollar ratios (units: score/$). Interpretable,
  unnormalized. These are the raw axes for Pareto (spec 01) and any custom
  weighting (spec 02/03).
- **`normalize_log(x, ref)`** and the **pinned anchors** `VS_REF, CV_REF,
  SP_REF, AV_REF, AC_REF` — the day-stable, pool-independent [0,1] mapping. Any
  spec that needs a comparable-across-days normalized value uses this function
  and these constants (do not re-derive daily min-max anywhere).
- **`composite_deal_score` / `arena_composite_score`** — `[0,1]`, day-stable,
  outlier-robust. Safe to trend across archives at `scoring_version >= 2`.
- **`size_tier ∈ {nano, small, medium, large, unknown}`** — honest taxonomy;
  `"unknown"` = no parseable size. No hidden "frontier==proprietary" semantics.
- **`passes_uptime_gate(m)`**, **`UPTIME_GATE_PCT = 95.0`**,
  **`is_deal_candidate(m, view)`** — the reliability gate. Selection/UI specs
  reuse these rather than re-implementing thresholds.
- **`SCORING_VERSION = 2`** (also on `ArchivePayload.scoring_version`) — the
  interpretability marker consumers branch on.

Specs 01–04 should treat everything above as read-only inputs; they add
new views/columns/frontier logic on top of these numbers, they do not re-open
the normalization or price decisions made here.
