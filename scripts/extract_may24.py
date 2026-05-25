"""One-off: extract May 24 model data from the rendered HTML committed at a4719b7.

Usage: python scripts/extract_may24.py
Output: docs/archive/2026-05-24.json, docs/archive/manifest.json
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from bs4 import BeautifulSoup

REPO_DIR = Path(__file__).parent.parent
DOCS_DIR = REPO_DIR / "docs"

# ── helpers ──────────────────────────────────────────────────────────────

def _float_or_none(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _int_or_none(s: str) -> int | None:
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    # Get the HTML from the May 24 commit
    result = subprocess.run(
        ["git", "show", "a4719b7:docs/index.html"],
        capture_output=True, text=True, cwd=REPO_DIR, check=True,
    )
    soup = BeautifulSoup(result.stdout, "html.parser")

    # --- Extract total_models ---
    models_tracked_el = soup.find("span", string=re.compile(r"models tracked"))
    total_models = 0
    if models_tracked_el:
        m = re.search(r"(\d+)", models_tracked_el.text)
        if m:
            total_models = int(m.group(1))

    # --- Extract date from header ---
    date_str = "2026-05-24"
    header_spans = soup.select("header span.bg-gray-800")
    for span in header_spans:
        text = span.text.strip()
        if re.match(r"^\w+ \d{1,2}, \d{4}$", text):
            date_str_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
            # The header uses "May 24, 2026" format — parse it
            try:
                from datetime import datetime
                dt = datetime.strptime(text, "%B %d, %Y")
                date_str = dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

    # --- Extract generated_at ---
    generated_at = "2026-05-24 10:26 UTC"
    gen_el = soup.find(string=re.compile(r"Generated \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC"))
    if gen_el:
        m = re.search(r"Generated (\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC)", str(gen_el))
        if m:
            generated_at = m.group(1)

    # --- Extract insights ---
    insights_parts = []
    insights_section = soup.select_one("section:has(h2:contains('AI Analysis'))")
    if insights_section:
        for p in insights_section.select("div.text-gray-300 p"):
            text = p.get_text(strip=True)
            if text:
                insights_parts.append(text)
    insights = "\n\n".join(insights_parts)

    # --- Extract model data from table rows ---
    models = []
    for tr in soup.select("table#model-table tbody tr"):
        name = tr.get("data-name", "")
        if not name:
            continue

        # Extract creator from the name cell
        creator = None
        creator_span = tr.select_one("td:first-child span.text-gray-500")
        if creator_span:
            t = creator_span.text.strip()
            # Remove leading "· " if present
            if t.startswith("· "):
                t = t[2:]
            creator = t if t else None

        model = {
            "name": name,
            "creator": creator,
            "intelligence_score": _float_or_none(tr.get("data-intel", "")),
            "price_input": None,
            "price_output": None,
            "price_blended": _float_or_none(tr.get("data-price", "")),
            "output_speed_tps": _float_or_none(tr.get("data-speed", "")),
            "context_window_k": _int_or_none(tr.get("data-ctx", "")),
            "coding_index": _float_or_none(tr.get("data-coding", "")),
            "livecodebench_score": None,
            "source_pages": [],
            "value_score": _float_or_none(tr.get("data-value", "")),
            "coding_value": None,
            "composite_deal_score": _float_or_none(tr.get("data-deal", "")),
            "param_billions": None,
            "size_tier": tr.get("data-tier", "frontier"),
        }
        models.append(model)

    if not models:
        print("ERROR: No models extracted from HTML", file=sys.stderr)
        sys.exit(1)

    # --- Compute hero & runners-up (frontier models, best deal score first) ---
    frontier = [
        m for m in models
        if m["size_tier"] == "frontier" and m["composite_deal_score"] is not None
    ]
    frontier.sort(key=lambda m: m["composite_deal_score"], reverse=True)
    hero = frontier[0] if frontier else None
    runners_up = frontier[1:4]

    # --- Write archive ---
    payload = {
        "date": date_str,
        "generated_at": generated_at,
        "insights": insights,
        "total_models": total_models,
        "models": models,
        "hero": hero,
        "runners_up": runners_up,
    }

    archive_dir = DOCS_DIR / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    out_path = archive_dir / "2026-05-24.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(models)} models to {out_path}")

    # --- Write manifest ---
    manifest_path = archive_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dates = set(manifest.get("dates", []))
    else:
        dates = set()
    dates.add("2026-05-24")
    manifest = {"dates": sorted(dates, reverse=True)}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
