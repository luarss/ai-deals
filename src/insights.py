"""Generate AI-written market insights via the DeepSeek API (OpenAI-compatible)."""

from __future__ import annotations

import logging
from datetime import date

from src.models import MergedModel

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a concise technical analyst specialising in AI model economics.
Produce a 280–320 word summary of the best-value AI models based on the \
benchmark data provided.
Write in clear, direct prose aimed at developers choosing a model for a \
production application. Avoid hype and marketing language.
Mention specific model names and concrete numbers.
Structure your response as exactly 3 paragraphs:
  1. The single best deal: which model, why, with its key metrics.
  2. Two or three notable runners-up and their specific trade-offs.
  3. One actionable recommendation for the most common use case \
(cost-sensitive API use).
Do not add headers, bullet points, or markdown formatting. \
Plain prose only."""

FALLBACK = (
    "AI insights are unavailable right now. "
    "See the table below for the full value rankings."
)


def build_user_prompt(top_models: list[MergedModel], as_of: str) -> str:
    """Build the data-rich user message sent to DeepSeek."""
    lines = [f"Data as of {as_of}. Top models by composite deal score:\n"]
    for i, m in enumerate(top_models[:10], 1):
        parts = [f"{i}. {m.name}"]
        if m.intelligence_score is not None:
            parts.append(f"intelligence={m.intelligence_score:.1f}/60")
        if m.price_blended is not None:
            parts.append(f"blended_price=${m.price_blended:.4f}/1M")
        elif m.price_input is not None:
            parts.append(f"input_price=${m.price_input:.4f}/1M")
        if m.value_score is not None:
            parts.append(f"value_score={m.value_score:.1f}")
        if m.coding_index is not None:
            parts.append(f"coding_index={m.coding_index:.1f}")
        if m.coding_value is not None:
            parts.append(f"coding_value={m.coding_value:.1f}")
        if m.output_speed_tps is not None:
            parts.append(f"speed={m.output_speed_tps:.0f}t/s")
        if m.context_window_k is not None:
            parts.append(f"context={m.context_window_k}K")
        if m.arena_elo is not None:
            parts.append(f"arena_elo={m.arena_elo:.0f}")
        if m.arena_coding_elo is not None:
            parts.append(f"arena_code_elo={m.arena_coding_elo:.0f}")
        lines.append(", ".join(parts))
    lines.append("\nWrite the 3-paragraph summary now.")
    return "\n".join(lines)


def generate_insights(top_models: list[MergedModel], api_key: str) -> str:
    """Call DeepSeek and return a prose insights string.

    Falls back to FALLBACK constant on any error so the page always renders.
    """
    try:
        from openai import OpenAI  # lazy import — optional at module load
    except ImportError:
        log.warning("openai package not installed; skipping insights")
        return FALLBACK

    user_prompt = build_user_prompt(top_models, date.today().isoformat())

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1200,
            temperature=0.4,
        )
        text = response.choices[0].message.content or ""
        return text.strip()
    except Exception as exc:
        log.warning("DeepSeek API error (%s): %s", type(exc).__name__, exc)
        return FALLBACK
