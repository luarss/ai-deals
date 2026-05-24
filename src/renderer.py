"""Render the final HTML page from a Jinja2 template."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.models import MergedModel

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def _fmt_price(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 0.01:
        return f"${value:.4f}"
    if value < 1:
        return f"${value:.3f}"
    return f"${value:.2f}"


def _fmt_score(value: float | None, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}" if value is not None else "—"


def _fmt_speed(value: float | None) -> str:
    return f"{value:.0f}" if value is not None else "—"


def _fmt_context(value: int | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value // 1000}M"
    return f"{value}K"


def render(
    top_models: list[MergedModel],
    all_models: list[MergedModel],
    insights_text: str,
    generated_at: datetime | None = None,
) -> str:
    """Render the full HTML page and return it as a string."""
    if generated_at is None:
        generated_at = datetime.now(timezone.utc)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    # Register formatting helpers as template filters
    env.filters["fmt_price"] = _fmt_price
    env.filters["fmt_score"] = _fmt_score
    env.filters["fmt_speed"] = _fmt_speed
    env.filters["fmt_context"] = _fmt_context

    template = env.get_template("index.html.j2")
    return template.render(
        hero_model=top_models[0] if top_models else None,
        runner_up_cards=top_models[1:4],
        all_models=all_models,
        insights=insights_text,
        generated_at=generated_at.strftime("%Y-%m-%d %H:%M UTC"),
        last_updated=generated_at.strftime("%B %d, %Y"),
        total_models=len(all_models),
    )
