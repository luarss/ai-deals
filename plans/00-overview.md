# Implementation overview

Five specs, each self-contained but sharing primitives. Total ~2,800 lines, all
thresholds grounded in measurements of the real archives (61 days in `docs/archive/`).

## Recommended implementation order

1. **05-mechanics-fixes** — foundational scoring primitives (`normalize_log` with
   pinned anchors, `PRICE_FLOOR`, `"unknown"` size tier, neutral imputation, uptime
   gate, `scoring_version=2`). Everything else consumes these.
2. **01-pareto-frontier** + **02-intelligence-bands** — both consume 05's primitives;
   independent of each other, can be done in either order or parallel.
3. **04-trends** — depends on shared `src/bands.py` and `src/pricing.py` (see below).
4. **03-daily-diff** — last, so diffs are computed on stable v2 scores. Its spec warns
   that landing it before/alongside 05 causes a one-day flood of fake rank-change
   events; it includes a `scoring_version`-change suppression for exactly this.

## Cross-spec reconciliations (decisions the specs defer to integration time)

### 1. `src/pricing.py` is the single source of truth for effective price
Specs 01, 03, 04, 05 each need the price fallback ladder (blended → input/output →
OpenRouter). Adopt spec 04's `src/pricing.py::effective_price(dict | MergedModel)`:
- `effective_price()` returns the **raw** effective price — used by diff (03), trends
  (04), and frontier x-axis (01). Real prices, not floored.
- Spec 05's `PRICE_FLOOR=$0.05` is applied **only inside scoring** (value_score /
  coding_value computation), not inside `effective_price()` — flooring the displayed
  or trended price would falsify data.
- Refactor `MergedModel._effective_blended_price` to delegate to it (parity test in
  spec 04 §Test plan).

### 2. `src/bands.py` is the single source of truth for intelligence thresholds
Spec 02 defines bands `<25 / 25–39 / ≥40` and `CHEAPEST_THRESHOLDS=[40, 25]`; spec 04
uses cumulative `[20, 30, 40, 50]`. Unify on **`[25, 40, 50]`**: 25/40 are spec 02's
distribution-grounded band edges (p50≈30, p75≈42), 50 added for the trend chart's
"frontier-class" line. Drop 04's 20 (below every band edge, answers no user question).
The site's "cheapest ≥40" number must match the trend line "cheapest ≥40" — same
constant, same price helper.

### 3. Hero selection ownership
Spec 01's frontier knee (`frontier_score`) picks the hero; spec 05's fixed composite
remains the table sort and tiebreak. Both independently converge on DeepSeek V4 Flash
over Ling-2.6-flash on today's data — a good regression anchor (locked in both specs'
test plans).

### 4. `compute_top_models`
Spec 05 owns the rewrite (drop the `size_tier=="frontier"` filter → has-composite +
uptime gate). Specs 01/02 read the result but do not modify the function.

## Live bugs found during spec work (fix regardless of the plans)

- `price_blended` is null/0 for **all 523 models** in the latest archive — AA scraper
  regression, currently masked by the OpenRouter fallback (spec 04 §Design).
- `coding_index` / `livecodebench_score` are null for all models — the composite's
  coding leg is dead code and the buggy weight-redistribution branch fires for every
  model (specs 01, 05).
- The frontend ignores `data.hero` and re-derives the hero from the top-3 table rows
  client-side (spec 02 §Frontend).
- `arena-text` source outages on 07-18 and 07-21 silently dropped 300+ models from
  the archives (spec 03 §Outage defense).
