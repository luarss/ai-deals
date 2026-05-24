# AI Deal Tracker

> Daily leaderboard of the best-value AI models — intelligence per dollar, coding performance, and speed — served via GitHub Pages.

**Live site:** `https://<your-username>.github.io/<repo-name>/`

---

## What it does

1. **Scrapes** [artificialanalysis.ai](https://artificialanalysis.ai/leaderboards/models) for model benchmarks (Intelligence Index, price, speed, coding index) by parsing embedded JSON-LD datasets.
2. **Calculates** value scores: `intelligence / blended_price`, `coding_index / blended_price`, and a composite deal score.
3. **Asks DeepSeek** to write a concise 3-paragraph market analysis of the best deals.
4. **Renders** a self-contained dark-theme HTML page with a hero card, runner-up cards, sortable table, and AI insights section.
5. **Commits** `docs/index.html` back to the repo via GitHub Actions — served automatically by GitHub Pages.

Runs automatically every day at 06:00 UTC.

---

## Quick start

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
- Python 3.12+
- A [DeepSeek API key](https://platform.deepseek.com/) (optional — insights section skipped if absent)

### Local run

```bash
# Clone and install
git clone https://github.com/<you>/<repo>.git
cd <repo>
uv sync

# Run without AI insights
uv run python main.py

# Run with AI insights
DEEPSEEK_API_KEY=sk-... uv run python main.py

# Open the output
open docs/index.html
```

---

## GitHub Pages setup

1. Push this repo to GitHub.
2. Go to **Settings → Pages → Source**: branch `main`, folder `/docs`. Save.
3. Add your DeepSeek API key: **Settings → Secrets and variables → Actions → New repository secret** named `DEEPSEEK_API_KEY`.
4. Trigger the first run manually: **Actions → Daily AI Deals Update → Run workflow**.

The page will be live at `https://<username>.github.io/<repo>/` within a minute of the first successful run.

---

## Project structure

```
ai-deals/
├── .github/workflows/daily.yml   # Daily cron (06:00 UTC) + manual trigger
├── src/
│   ├── models.py                 # Pydantic data models
│   ├── scraper.py                # HTTP fetch + JSON-LD parse + HTML table fallback
│   ├── merger.py                 # Deduplication across pages, field coalescion
│   ├── scorer.py                 # Value metrics + normalised composite score
│   ├── insights.py               # DeepSeek API integration
│   └── renderer.py               # Jinja2 → HTML
├── templates/index.html.j2       # Dark-theme page template
├── docs/index.html               # Generated output (committed by CI)
├── main.py                       # Pipeline orchestrator
└── pyproject.toml                # Dependencies (managed with uv)
```

---

## Value score methodology

| Metric | Formula |
|--------|---------|
| **Value Score** | `intelligence_score / blended_price` |
| **Coding Value** | `coding_index / blended_price` |
| **Deal Score** | `0.5 × norm(value) + 0.3 × norm(coding_value) + 0.2 × norm(speed)` |

Blended price uses Artificial Analysis's 7:2:1 cache/input/output ratio where available; falls back to 2:1 input/output.

---

## Resilience

- Single-page fetch failures log a warning and are skipped; remaining pages still contribute data.
- JSON-LD yields 0 records → automatic fallback to HTML table parsing.
- DeepSeek API errors → static fallback message; page still renders in full.
- If 0 models are scraped, the pipeline exits with code 1 and the Actions job fails visibly.

---

*Data: [artificialanalysis.ai](https://artificialanalysis.ai) · Analysis: DeepSeek AI · Not affiliated with Artificial Analysis.*
