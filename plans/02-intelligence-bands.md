# Spec 02 — Intelligence-Band Leaderboards

## Problem

The site crowns exactly one global "Best Deal Today" across all 523 tracked models
using a single intelligence-per-dollar ratio (`value_score = intelligence / blended_price`,
folded into `composite_deal_score`). That ratio is monotonically dominated by
near-free, low-intelligence models: today's hero is **inclusionAI Ling-2.6-flash**
with an intelligence score of **14/60** and a `value_score` of 840 purely because it
is nearly free. This is a useless recommendation. Nobody asks "which model has the
highest intelligence-per-dollar?" — they ask **"what is the cheapest model that is
good enough for my task?"**

The fix is to segment the leaderboard by intelligence capability band, so that
"best value" is computed *within* a band of comparable capability, and to add a
direct **"cheapest model at or above intelligence X"** lookup. Segmentation alone
removes the cross-band pathology: within a band every model is already "good enough,"
so ranking by value is meaningful again.

### Confirmed data reality (inspected in `docs/data/latest.json`, 2026-07-24)

Only a minority of the 523 models carry the signals bands need:

| field | present / 523 |
|---|---|
| `intelligence_score` | 115 |
| `value_score` (needs intel + price) | 51 |
| `composite_deal_score` | 51 |
| `coding_index` | **0** |
| `coding_value` | **0** |
| `livecodebench_score` | **0** |
| `arena_elo` | **0** |
| `arena_coding_elo` | 100 |
| `arena_composite_score` | 93 |
| `price_blended` | 0 (all prices come via `price_input` fallback) |

Consequences that shape this design:
- **Bands are an AA-intelligence-view feature only.** The eligible pool is the
  51 models that have a `composite_deal_score` (all 51 also have `intelligence_score`).
- **No AA coding bands are possible right now** — `coding_index` is 0/523. See
  "Do bands apply to the coding view?" below.
- **The Arena view keeps its existing single-hero behaviour** — Arena has no
  0–60 intelligence analog (Elo is ~1000–1500), and `arena_elo` is currently 0/523.

## Design

### Band cutoffs, grounded in the real distribution

Intelligence-score distribution across the **115** models that have a score
(histogram, bin width 5; `intelligence_score` ranges 4–60, this is the AA
Intelligence Index):

```
  0-4  :  1  #
  5-9  : 10  ##########
 10-14 : 11  ###########
 15-19 : 13  #############
 20-24 : 12  ############
 25-29 :  9  #########
 30-34 : 12  ############
 35-39 :  9  #########
 40-44 : 15  ###############
 45-49 :  7  #######
 50-54 :  9  #########
 55-59 :  6  ######
 60-64 :  1  #
percentiles: p10=10  p25=17  p50=30  p75=42  p90=51  p95=56
```

Distribution across the **51 band-eligible** models (those with a
`composite_deal_score`, which is the pool bands actually rank):

```
 10-14 :  4
 15-19 :  3
 20-24 :  5
 25-29 :  4
 30-34 :  7
 35-39 :  4
 40-44 :  9
 45-49 :  2
 50-54 :  7
 55-59 :  5
 60-64 :  1
```

**Chosen cutoffs: `< 25`, `25–<40`, `≥ 40`.**

Rationale:
- **40** sits at the top-quartile boundary of the full population (p75 ≈ 42) and at
  the base of the visually dense 40–60 cluster (15+7+9+6+1 = 38 of 115 models score
  ≥40). Everything ≥40 is genuinely capable — GPT/Claude/Gemini-class general
  reasoning. This is the "is it smart enough for serious work" line.
- **25** sits just below the population median (p50 = 30) and cleanly separates the
  small/quantized/older tier (the 4–24 mass) from mid-capability models. Below 25 the
  models are useful mainly for cheap bulk/classification work, not open-ended reasoning.
- On the eligible pool these cutoffs yield a **balanced 24 / 15 / 12** split
  (flagship / workhorse / budget), so no band is empty or trivially small. This
  balance is a design goal — bands with 0–1 members produce a useless leaderboard.

Cutoffs are defined once as a constant (see `scorer.py` changes) so a future
re-tune is a one-line edit. Boundaries are **left-closed / right-open**:
`budget = [−∞, 25)`, `workhorse = [25, 40)`, `flagship = [40, +∞)`.

### Band naming

Capability-communicating labels, with **stable internal ids that deliberately do
NOT reuse the string `"frontier"`** (that token is already `size_tier == "frontier"`,
the "no explicit size info" bucket — reusing it would be a footgun, especially since
spec 05 is changing what `size_tier == "frontier"` means):

| id (JSON, stable) | display label | range | intent |
|---|---|---|---|
| `flagship` | **Frontier-class** · Intelligence ≥ 40 | `[40, ∞)` | Serious reasoning / agentic work |
| `workhorse` | **Workhorse** · 25–39 | `[25, 40)` | Everyday tasks, good value |
| `budget` | **Budget** · < 25 | `[−∞, 25)` | Cheap bulk / classification |

The UI label for `flagship` reads "Frontier-class" (per the task brief); the JSON
`id` is `flagship` to keep it orthogonal to `size_tier`.

### What "best deal within a band" means

Each band exposes **two** answers, because users ask two different questions:

1. **`leader` — best value within the band.** The band member with the highest
   `composite_deal_score`. Because all members are within a narrow intelligence range,
   the value component is no longer dominated by cross-band price gaps; the winner is a
   genuinely good "most bang per buck at this capability level" pick. The composite
   also folds in output speed (20% weight), so `leader` ≠ merely cheapest.
2. **`cheapest` — the literal cheapest member.** Lowest effective price among band
   members. Directly answers "the cheapest thing in this capability class."

**We do NOT re-normalize the composite per band.** The existing global
`composite_deal_score` is reused as-is. Rationale: (a) segmentation already removes
the cross-band pathology, so per-band renormalization buys little; (b) it keeps the
band scores consistent with the "All Models" table's `Deal Score` column (which the
user can cross-reference); (c) it is strictly additive to the pipeline and needs no
change to `rank_models`. Per-band renormalization is a rejected alternative recorded
here so nobody re-derives it.

### "Cheapest model at or above threshold X" — cumulative

Separately from the per-band `cheapest` (which is band-bounded), expose a
**cumulative** lookup: the single cheapest model whose intelligence is `≥ threshold`,
where `threshold` ∈ {25, 40} (the band floors). This is the truest answer to
"cheapest good-enough," because a higher-band model can be cheaper than a
same-band one. Verified against today's data:

```
intel ≥ 40 : cheapest = Tencent Hy3   (intel 41, eff $0.287)   [24 candidates]
intel ≥ 25 : cheapest = MiMo-V2.5      (intel 37, eff $0.187)   [39 candidates]
```

(Note `cheapest_above[25]` legitimately surfaces a 37-intelligence model — cumulative,
not band-bounded — which is exactly the useful behaviour.)

### Do bands apply to the coding view?

**Not now, and by design not as a separate axis.** `coding_index` is 0/523 in the
current data, so an AA coding-band leaderboard is impossible. More importantly, even
once `coding_index` populates, we should **not** create a parallel set of "coding
bands." A model's *capability class* is its general intelligence; coding is a
*use-case sort within* that class. The recommended future extension is: keep the
same three intelligence bands, and add a per-band sort/secondary metric on
`coding_value`. This spec does not implement coding sorting; it only reserves the
decision. The Arena view is out of scope for bands entirely (see Problem).

## Exact changes per file

### `src/models.py`

Add two lightweight archive-only models and extend `ArchivePayload`. Band entries
store **`model_id` references**, not full `MergedModel` copies, to avoid bloating the
already-large payload (523 full models). The frontend resolves ids against
`data.models` (it already builds a name→model map; it will build an id→model map).

```python
class BandDeal(BaseModel):
    """One intelligence band's leaderboard summary (references models by model_id)."""
    id: str                      # "flagship" | "workhorse" | "budget"
    label: str                   # display label, e.g. "Frontier-class"
    min_intelligence: float      # inclusive floor
    max_intelligence: Optional[float] = None  # exclusive ceiling; None = open-ended
    count: int = 0               # number of eligible members in this band
    leader_id: Optional[str] = None    # best composite_deal_score in band
    cheapest_id: Optional[str] = None  # lowest effective price in band
    member_ids: list[str] = Field(default_factory=list)  # top-N by composite, incl. leader


class CheapestAbove(BaseModel):
    """Cheapest model with intelligence >= threshold (cumulative)."""
    threshold: float
    model_id: Optional[str] = None
    effective_price: Optional[float] = None


class ArchivePayload(BaseModel):
    ...
    # NEW — all optional / defaulted for backward compat (see Archive schema section)
    bands: list["BandDeal"] = Field(default_factory=list)
    cheapest_above: list["CheapestAbove"] = Field(default_factory=list)
```

Also **promote the price helper to a public method** so `scorer.py` (and spec 01)
can share it instead of re-implementing the fallback ladder. Rename
`_effective_blended_price` → keep as-is but add a thin public alias:

```python
def effective_price(self) -> Optional[float]:
    """Public alias for the best-available blended price estimate."""
    return self._effective_blended_price()
```

(Keeping the private method avoids touching its two existing `model_validator`
call sites.)

### `src/scorer.py`

Add band constants and a `compute_bands` function. Do **not** touch `rank_models`
or the existing `compute_top_models` (the overall hero stays for backward compat).
Critically, **band membership is based on `intelligence_score`, never `size_tier`.**

```python
# Intelligence-band definitions. Left-closed / right-open on intelligence_score.
# (id, display_label, min_inclusive, max_exclusive_or_None)
BANDS: list[tuple[str, str, float, float | None]] = [
    ("flagship",  "Frontier-class", 40.0, None),
    ("workhorse", "Workhorse",      25.0, 40.0),
    ("budget",    "Budget",          0.0, 25.0),
]

# Thresholds for the cumulative "cheapest at or above X" lookup (band floors > 0).
CHEAPEST_THRESHOLDS: list[float] = [40.0, 25.0]

BAND_MEMBER_LIMIT = 8  # top-N members surfaced per band for the UI


def _band_of(intel: float) -> str | None:
    for band_id, _label, lo, hi in BANDS:
        if intel >= lo and (hi is None or intel < hi):
            return band_id
    return None


def compute_bands(models: list[MergedModel]) -> list["BandDeal"]:
    """Build per-band leaderboards from the AA-intelligence pool.

    Eligible = has intelligence_score AND composite_deal_score. Returns bands in
    flagship→workhorse→budget order. Empty bands are still returned (count=0) so
    the frontend can render a stable set of sections.
    """
    from src.models import BandDeal  # local import avoids a cycle

    eligible = [
        m for m in models
        if m.intelligence_score is not None and m.composite_deal_score is not None
    ]

    result: list[BandDeal] = []
    for band_id, label, lo, hi in BANDS:
        members = [
            m for m in eligible
            if m.intelligence_score >= lo and (hi is None or m.intelligence_score < hi)
        ]
        by_composite = sorted(members, key=lambda m: -(m.composite_deal_score or 0))
        by_price = sorted(
            members,
            key=lambda m: (m.effective_price() is None, m.effective_price() or float("inf")),
        )
        result.append(BandDeal(
            id=band_id,
            label=label,
            min_intelligence=lo,
            max_intelligence=hi,
            count=len(members),
            leader_id=by_composite[0].model_id if by_composite else None,
            cheapest_id=by_price[0].model_id if by_price else None,
            member_ids=[m.model_id for m in by_composite[:BAND_MEMBER_LIMIT]],
        ))
    return result


def compute_cheapest_above(models: list[MergedModel]) -> list["CheapestAbove"]:
    """For each threshold, the single cheapest model with intelligence >= threshold."""
    from src.models import CheapestAbove

    pool = [
        m for m in models
        if m.intelligence_score is not None and m.composite_deal_score is not None
    ]
    out: list[CheapestAbove] = []
    for t in CHEAPEST_THRESHOLDS:
        cands = [m for m in pool if m.intelligence_score >= t and m.effective_price() is not None]
        if cands:
            best = min(cands, key=lambda m: m.effective_price())
            out.append(CheapestAbove(threshold=t, model_id=best.model_id,
                                     effective_price=round(best.effective_price(), 4)))
        else:
            out.append(CheapestAbove(threshold=t))
    return out
```

Note on `compute_top_models` (unchanged here but flagged): it filters
`size_tier == "frontier"`. That filter is the mislabel spec 05 fixes and is
**irrelevant to bands** — `compute_bands` ignores `size_tier`. Leaving
`compute_top_models` as-is preserves the legacy overall hero for backward compat.

### `main.py`

After ranking, compute bands and thread them into `write_archives`. Also feed the
**band leaders** (not the near-free top-10) into insights so the AI commentary
describes real recommendations.

```python
from src.scorer import annotate_sizes, compute_bands, compute_cheapest_above, compute_top_models, rank_models
...
    ranked = rank_models(merged)

    top_models = compute_top_models(ranked, "aa")      # legacy overall hero (kept)
    arena_top = compute_top_models(ranked, "arena")
    bands = compute_bands(ranked)                        # NEW
    cheapest_above = compute_cheapest_above(ranked)      # NEW
...
    # Insights: prefer band leaders (deduped, capped) over the near-free top-10.
    id_to_model = {m.model_id: m for m in ranked}
    insight_seed = []
    for b in bands:
        if b.leader_id and id_to_model.get(b.leader_id):
            insight_seed.append(id_to_model[b.leader_id])
    if api_key:
        insights = generate_insights(insight_seed or top_models[:10], api_key)
...
    write_archives(
        ranked, insights, generated_at,
        top_models=top_models, arena_top_models=arena_top,
        bands=bands, cheapest_above=cheapest_above,   # NEW
    )
```

(`generate_insights(top_models: list[MergedModel], api_key)` signature is unchanged —
we just pass a better list.)

### `src/writer.py`

Thread the new optional args through `write_archives` into the payload:

```python
def write_archives(
    models: list[MergedModel],
    insights: str,
    generated_at: datetime,
    top_models: list[MergedModel] | None = None,
    arena_top_models: list[MergedModel] | None = None,
    bands: list["BandDeal"] | None = None,            # NEW
    cheapest_above: list["CheapestAbove"] | None = None,  # NEW
) -> None:
    ...
    payload = ArchivePayload(
        ...,
        bands=bands or [],
        cheapest_above=cheapest_above or [],
    )
```

No change to `_write_json` or `_update_manifest`.

### `src/renderer.py`

No change — it renders `templates/index.html` verbatim. All new UI is client-side.

## Archive schema changes & backward compatibility

New top-level keys in each archive JSON (`docs/data/latest.json` and
`docs/archive/YYYY-MM-DD.json`):

```jsonc
{
  // ... existing keys unchanged: date, generated_at, insights, total_models,
  //     models, hero, runners_up, arena_hero, arena_runners_up ...
  "bands": [
    {
      "id": "flagship",
      "label": "Frontier-class",
      "min_intelligence": 40.0,
      "max_intelligence": null,
      "count": 24,
      "leader_id": "deepseek-v4-flash",
      "cheapest_id": "tencent-hy3",
      "member_ids": ["deepseek-v4-flash", "tencent-hy3", "..."]
    },
    { "id": "workhorse", "min_intelligence": 25.0, "max_intelligence": 40.0, ... },
    { "id": "budget",    "min_intelligence": 0.0,  "max_intelligence": 25.0, ... }
  ],
  "cheapest_above": [
    { "threshold": 40.0, "model_id": "tencent-hy3", "effective_price": 0.287 },
    { "threshold": 25.0, "model_id": "mimo-v2.5",   "effective_price": 0.187 }
  ]
}
```

**Backward compatibility (62 existing archives in `docs/archive/`):**
- **Reading old archives with new code:** `bands` and `cheapest_above` are
  `Field(default_factory=list)` — Pydantic supplies `[]` when the keys are absent.
  No migration, no re-generation of historical archives required.
- **Frontend reading old archives:** the band UI checks `data.bands && data.bands.length`
  and, when empty, **falls back to the existing single-hero derivation** (the current
  top-3-visible-rows logic in `renderPage`). So historical dates keep rendering exactly
  as they do today.
- **Model references resolve via `data.models`**, which every archive already contains,
  so `leader_id` / `member_ids` never dangle within the same archive.
- No existing field is renamed, removed, or has its meaning changed. `hero` /
  `runners_up` remain written (legacy overall deal) for compatibility.

## Frontend changes (`templates/index.html`)

### Data contract consumed by the frontend

From the loaded archive:
- `data.models: MergedModel[]` — full list (already used). Build an id map:
  `byId = {}; data.models.forEach(m => byId[m.model_id] = m);`
- `data.bands: BandDeal[]` — ordered flagship→workhorse→budget. Each has
  `id, label, min_intelligence, max_intelligence, count, leader_id, cheapest_id, member_ids`.
- `data.cheapest_above: CheapestAbove[]` — `{threshold, model_id, effective_price}`.
- Resolve any `*_id` to a model via `byId[id]`; skip if missing.
- If `!data.bands || data.bands.length === 0` → **legacy path**: render the existing
  single hero + runners-up exactly as today.

### UI: stacked band sections (not tabs)

Replace the single "Best Deal Today" hero region with **three stacked sections**,
one per band, rendered only in the AA view (`currentView !== 'arena'`). Stacked (not
tabbed) so a visitor scanning "what's the cheapest good-enough model" sees all three
capability tiers at a glance without clicking. In the Arena view, hide the band
sections and keep the existing single Arena hero/runners-up behaviour.

Each band section contains:
1. **Header row:** display label + range badge ("Intelligence ≥ 40" / "25–39" / "< 25")
   + member `count`.
2. **Leader card** (reuse the existing `hero-card` visual style): the model resolved
   from `leader_id`, showing name, creator, `composite_deal_score` ("Deal Score"),
   intelligence, blended price (`effPrice`), value score, speed — identical metric
   layout to today's AA hero card.
3. **"Cheapest in band" chip:** small inline line resolved from `cheapest_id` —
   e.g. `Cheapest here: Tencent Hy3 · $0.29/1M · intel 41`.
4. **Runners row:** `member_ids` minus the leader, up to ~4, rendered as the existing
   runner-up card style.

Above the three sections, a compact **"Cheapest that's good enough"** strip built
from `data.cheapest_above`:
`≥40 → Tencent Hy3 $0.29/1M   ·   ≥25 → MiMo-V2.5 $0.19/1M`.

The **"All Models" table below is unchanged** (it already has the intelligence slider
and size-tier chips). Bands augment, they don't replace, the table.

### Rendering notes (spec, not full JS)

- Add `function renderBands(data, byId)` that early-returns to the legacy hero path
  when `data.bands` is empty or `currentView === 'arena'`.
- Reuse existing helpers verbatim: `effPrice`, `fmtPrice`, `fmtScore`, `fmtSpeed`,
  `escapeHtml`. Do not duplicate the price ladder — `effPrice` already mirrors
  `MergedModel.effective_price()`.
- The three section containers are new static markup in `<main>` (ids
  `band-flagship`, `band-workhorse`, `band-budget`, plus a `cheapest-strip`), all
  `class="hidden"` by default and shown by `renderBands`.
- When falling back to legacy, `renderBands` shows the existing `#hero-section` /
  `#runners-section` and hides the band containers; when bands are present it does the
  inverse. This keeps historical archives pixel-compatible.

## Test plan

Unit (`tests/` — pytest, matching repo style):
1. **Band assignment boundaries** — `_band_of(40.0) == "flagship"`,
   `_band_of(39.9) == "workhorse"`, `_band_of(25.0) == "workhorse"`,
   `_band_of(24.9) == "budget"`, `_band_of(0.0) == "budget"`.
2. **`compute_bands` membership** — construct ~6 `MergedModel`s spanning the ranges;
   assert each lands in the right band and counts sum to the eligible total. Models
   lacking `intelligence_score` OR `composite_deal_score` are excluded.
3. **`leader_id` = max composite in band**, **`cheapest_id` = min effective price**;
   verify they can differ (a model with high composite but not cheapest).
4. **Empty band** — no models in a range → band still returned with `count == 0`,
   `leader_id is None`, `member_ids == []`.
5. **`compute_cheapest_above` cumulative** — a ≥40 model that is pricier than a
   37-intel model must NOT be returned for `threshold=25`; assert the cheaper
   lower-intel model wins at 25 while the ≥40 threshold returns the cheapest ≥40.
6. **`member_ids` capped at `BAND_MEMBER_LIMIT`** and ordered by descending composite.
7. **Backward-compat load** — `ArchivePayload.model_validate(json.loads(old))` on a
   real file from `docs/archive/` (which lacks `bands`) yields `bands == []` and
   `cheapest_above == []` without error.
8. **Round-trip** — build a payload with bands, `model_dump(mode="json")`, reload,
   assert equality.

Integration / manual:
9. Run `python main.py` (or the writer directly on ranked fixtures) and confirm
   `docs/data/latest.json` gains non-empty `bands` (3 entries) and `cheapest_above`.
10. Open `docs/index.html` against latest.json — three stacked band sections render,
    flagship leader is a ≥40-intelligence model (not the 14-intel near-free hero),
    the cheapest strip shows both thresholds.
11. Switch the date selector to a pre-existing archive date — legacy single hero
    renders, no console errors (validates the fallback path).
12. Switch to the Arena view — band sections hidden, existing Arena hero intact.

## Dependencies & integration notes (parallel specs)

**Spec 05 — mechanics-fixes (`size_tier` "frontier" mislabel):**
- Bands are computed **entirely from `intelligence_score`** and never read
  `size_tier`, so 05's changes cannot break band membership. This is deliberate — the
  task brief calls it out.
- 05 changes what `compute_top_models` returns (it filters `size_tier == "frontier"`),
  which affects only the **legacy overall hero** kept for backward compat. If 05 lands
  first, that legacy hero simply improves; no coordination needed. Keep
  `compute_top_models` intact so 05 owns it cleanly.
- Both specs touch `src/scorer.py`; keep changes in separate functions/constant blocks
  to minimize merge friction (bands add `BANDS`, `compute_bands`, `compute_cheapest_above`;
  05 edits `_size_tier` / `annotate_sizes`).

**Spec 01 — pareto-frontier:**
- Both need an "effective price" and both are alternative "best deal" framings.
  **Share one price helper** — this spec promotes `MergedModel.effective_price()` to a
  public method; spec 01 should consume it rather than re-implementing the fallback
  ladder (also mirrored in the frontend's `effPrice`). Coordinate so only one such
  helper exists in Python and one in JS.
- If 01 adds a per-model pareto flag (e.g. `on_pareto_frontier: bool`), bands can
  cheaply surface it: mark band members that are also pareto-optimal (a "★ frontier"
  badge). This is an optional enhancement, not a dependency.
- **Avoid a field-name collision:** if 01 also introduces an archive field for
  effective price, agree on one name/semantics. This spec stores price only inside
  `cheapest_above[].effective_price`; it does not add a per-model price field.
- Both add top-level archive keys — no key names overlap (`bands`, `cheapest_above`
  here). Both must use `Field(default_factory=...)` for old-archive compatibility.
```
