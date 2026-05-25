# AI Deal Tracker

Daily leaderboard of the best-value AI models — intelligence per dollar, served via GitHub Pages.

**Live site:** [`luarss.github.io/ai-deals`](https://luarss.github.io/ai-deals/)

---

## How it works

1. Scrapes [artificialanalysis.ai](https://artificialanalysis.ai/leaderboards/models) for model benchmarks.
2. Scores models by value: `intelligence / blended_price`, `coding_index / blended_price`, and a composite deal score.
3. Generates AI market analysis via DeepSeek (optional).
4. Renders a self-contained HTML page to `docs/index.html`, auto-deployed via GitHub Pages.

Runs daily at 06:00 UTC via GitHub Actions.

---

## Quick start

```bash
git clone https://github.com/luarss/ai-deals.git
cd ai-deals
uv sync

# Run (AI insights optional)
uv run python main.py
open docs/index.html
```

---

## GitHub Pages setup

1. Go to **Settings → Pages → Source**: branch `main`, folder `/docs`.
2. Add `DEEPSEEK_API_KEY` as a repository secret (**Settings → Secrets → Actions**).
3. Trigger: **Actions → Daily AI Deals Update → Run workflow**.

---

## Project structure

```
ai-deals/
├── .github/workflows/daily.yml
├── src/
│   ├── models.py          # Pydantic data models
│   ├── scraper.py         # HTTP fetch + JSON-LD parse
│   ├── merger.py          # Deduplication across pages
│   ├── scorer.py          # Value metrics + composite score
│   ├── insights.py        # DeepSeek API integration
│   └── renderer.py        # Jinja2 → HTML
├── templates/index.html.j2
├── docs/index.html         # Generated output (committed by CI)
├── main.py                 # Pipeline orchestrator
└── pyproject.toml
```

---

## Value score

| Metric | Formula |
|--------|---------|
| **Value Score** | `intelligence_score / blended_price` |
| **Coding Value** | `coding_index / blended_price` |
| **Deal Score** | `0.5 × norm(value) + 0.3 × norm(coding_value) + 0.2 × norm(speed)` |

Blended price uses a 7:2:1 cache/input/output ratio where available; falls back to 2:1 input/output.

---

*Data: [artificialanalysis.ai](https://artificialanalysis.ai) · Not affiliated with Artificial Analysis.*
