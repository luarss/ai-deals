# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Developers and builders choosing which LLM API to use. They arrive with a budget and a capability bar and want the model that gives the most intelligence per dollar, plus any free tiers or credits that lower the cost further.

## Product Purpose

AI Deal Tracker publishes a daily leaderboard of the best-value AI models. Success means a developer can land on the page, see today's best-value pick and the alternatives at their capability level, and leave with a model choice they trust, without having to cross-reference several leaderboards themselves.

## Positioning

- **Value, not raw power:** ranks intelligence per dollar, using a Pareto frontier (price vs. intelligence) and a single hero pick chosen at the frontier knee, instead of ranking on benchmark scores alone.
- **Cross-source merge:** joins Artificial Analysis benchmarks and prices, Arena Elo (text and code), and OpenRouter pricing and uptime into one view per model.
- **Daily change log:** shows what changed since yesterday (price moves, new models, rank shifts) and long-run price trends built from the daily archive.
- **Free credits alongside paid value:** a Free AI Credits tab (sourced from freebie-radar) sits next to the paid rankings.

## Operating Context

- Regenerated daily by a GitHub Actions workflow (`.github/workflows/daily.yml`) and published to GitHub Pages at `luarss.github.io/ai-deals`.
- Pipeline: `main.py` → fetchers (`src/scraper.py`, `src/arena_fetcher.py`, `src/openrouter_fetcher.py`, `src/freebie_fetcher.py`) → `src/merger.py` → scoring, frontier, bands, diff, timeseries → `src/renderer.py` renders `templates/index.html` into `docs/index.html`.
- DeepSeek-generated "AI Analysis" commentary when `DEEPSEEK_API_KEY` is set.
- Data artifacts: `docs/data/latest.json`, `docs/data/timeseries.json`, `docs/data/freebies.json`, and daily snapshots in `docs/archive/`.
- Health gates (`src/health.py`) guard against source outages that would silently drop models.

## Capabilities and Constraints

- Views: Artificial Analysis and Arena AI toggles (OpenRouter feeds price fallback and uptime, not a separate view), plus a Free AI Credits tab. Also tier chips, a minimum-intelligence slider (no minimum by default, with presets at the band edges), a sortable All Models table, intelligence bands (`<25 / 25–39 / ≥40`, with 50 as the frontier-class trend line), a frontier chart, trend charts, and the changelog headline.
- Scoring constants live in one place: `src/bands.py` holds the thresholds, and `src/pricing.py::effective_price` returns the raw price. `PRICE_FLOOR` applies only inside scoring, never to displayed or trended prices.
- **Stay static and free:** the output is a single generated HTML page on GitHub Pages, with no backend and no paid hosting.
- **No data fabrication:** show only sourced numbers. Missing data stays visibly missing on the page. Neutral imputation is allowed inside scoring, but never in display.
- **Mobile matters:** a meaningful share of visits come from phones, so the leaderboard, filters and charts must work at phone width.
- Python 3.12+, managed with `uv`. Tests run with `pytest`.

## Brand Commitments

- Name: "AI Deal Tracker". Tagline in current copy: "Intelligence per dollar · updated daily".
- The current copy attributes data to artificialanalysis.ai, arena.ai and openrouter.ai and states the site is not affiliated with any data source.

## Evidence on Hand

- Real daily data: `docs/data/latest.json`, `docs/data/timeseries.json`, `docs/data/freebies.json`, and the archive in `docs/archive/` (from 2026-05-24 onward).
- Design and scoring specs: `plans/00-overview.md` through `plans/05-mechanics-fixes.md`.
- No testimonials, user counts, press, or usage analytics exist. Future work must not invent them.

## Product Principles

1. **Answer the decision first.** Today's best-value pick and the alternatives at the user's capability bar come before raw tables.
2. **Truth over completeness.** Show a gap rather than fill one; every number traces to a named source.
3. **One source of truth per concept.** The same threshold and price helper drive every number that claims the same thing.
4. **Change is the reason to come back.** What moved since yesterday is first-class content, not a footnote.
5. **Zero-cost to run, zero-friction to read.** Static, fast, and usable on a phone.
