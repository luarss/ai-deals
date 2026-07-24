"""Generate an AI-written daily market brief via the DeepSeek API (OpenAI-compatible).

Spec 03 rewired this from "restate the leaderboard" to "summarize the diff":
the model is handed the structured :class:`~src.models.Changelog` plus the current
top value models for context, and writes a short brief that leads with the biggest
change. On an empty changelog it says the market was quiet rather than inventing news.
The DEEPSEEK_API_KEY-absent / error path still returns FALLBACK so the page renders.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from src.models import Changelog, MergedModel

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
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
- 120-200 words, 2-3 short paragraphs, plain prose, no headers, no bullet points, no markdown.
- Direct and factual. No hype, no marketing language."""

FALLBACK = (
    "AI insights are unavailable right now. "
    "See the table below for the full value rankings."
)


def _fmt_price(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    if v < 1:
        return f"${v:.3f}"
    return f"${v:.2f}"


def build_user_prompt(
    top_models: list[MergedModel], changelog: Optional[Changelog], as_of: str
) -> str:
    """Build the diff-summary user message sent to DeepSeek.

    Lists each changelog section (or "none"), then the current top value models
    for reference only. If every section is "none" the model is told to say the
    market was quiet.
    """
    cl = changelog or Changelog(first_run=True)
    compared = cl.compared_to or "N/A"
    dq = cl.data_quality or "ok"
    dq_line = f"Data quality: {dq}"
    if cl.degraded_sources:
        dq_line += f", unavailable sources: {', '.join(cl.degraded_sources)}"

    lines = [
        f"Date: {as_of}. Changes since {compared}.",
        dq_line + ".",
    ]
    if cl.first_run:
        lines.append("(First snapshot — no previous day to compare.)")
    if cl.scoring_changed:
        lines.append(
            "(Scoring methodology changed vs the previous day; rank/frontier "
            "moves are suppressed to avoid false alarms.)"
        )

    def section(title: str, items: list[str]) -> None:
        lines.append("")
        lines.append(f"{title}:")
        if items:
            lines.extend(f"  - {it}" for it in items)
        else:
            lines.append("  - none")

    section(
        "PRICE CHANGES",
        [
            f"{e.name}: {_fmt_price(e.old)} -> {_fmt_price(e.new)} "
            f"({(e.pct or 0):+.0f}%, {e.direction})"
            for e in cl.price_changes
        ],
    )
    section(
        "NEW MODELS",
        [
            f"{e.name}: intelligence {e.intelligence_score}, price {_fmt_price(e.price)}, "
            f"deal {e.composite_deal_score}"
            for e in cl.new_models
        ],
    )
    section(
        "INTELLIGENCE CHANGES",
        [
            f"{e.name}: {e.old} -> {e.new} ({(e.delta or 0):+.1f} pts, {e.direction})"
            for e in cl.intelligence_changes
        ],
    )
    rank_items = [
        f"{e.name}: "
        + (f"#{e.old_rank} -> #{e.new_rank}" if e.new_rank else f"#{e.old_rank} -> out")
        + " ("
        + (
            "entered top 10"
            if e.entered_top
            else "left top 10"
            if e.left_top
            else (e.direction or "")
        )
        + ")"
        for e in cl.rank_changes
    ]
    rank_items += [
        f"{e.name}: {e.direction} the intelligence frontier" for e in cl.frontier_changes
    ]
    section("RANK / FRONTIER MOVES", rank_items)
    section(
        "REMOVED MODELS",
        [
            f"{e.name} (was intelligence {e.intelligence_score}, {_fmt_price(e.price)})"
            for e in cl.removed_models
        ],
    )
    section(
        "SPEED CHANGES (secondary, top 5 only)",
        [
            f"{e.name}: {e.old} -> {e.new} t/s ({(e.pct or 0):+.0f}%)"
            for e in cl.speed_changes[:5]
        ],
    )

    lines.append("")
    lines.append("For reference only (do not just restate this list), current top value models:")
    for i, m in enumerate(top_models[:5], 1):
        parts = [f"  {i}. {m.name}"]
        if m.intelligence_score is not None:
            parts.append(f"intelligence {m.intelligence_score:.1f}")
        price = m.effective_price()
        if price is not None:
            parts.append(f"{_fmt_price(price)}/1M")
        if m.composite_deal_score is not None:
            parts.append(f"deal {m.composite_deal_score:.2f}")
        lines.append(", ".join(parts))

    lines.append("")
    lines.append("If every section above is \"none\", say the market was quiet today.")
    lines.append("Write the brief now.")
    return "\n".join(lines)


def generate_insights(
    top_models: list[MergedModel], changelog: Optional[Changelog], api_key: str
) -> str:
    """Call DeepSeek and return a prose daily brief summarizing the changelog.

    Falls back to FALLBACK constant on any error so the page always renders.
    """
    try:
        from openai import OpenAI  # lazy import — optional at module load
    except ImportError:
        log.warning("openai package not installed; skipping insights")
        return FALLBACK

    user_prompt = build_user_prompt(top_models, changelog, date.today().isoformat())

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=500,
            temperature=0.4,
        )
        text = response.choices[0].message.content or ""
        return text.strip()
    except Exception as exc:
        log.warning("DeepSeek API error (%s): %s", type(exc).__name__, exc)
        return FALLBACK
