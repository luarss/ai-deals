"""Render the final HTML page from a static template."""

from __future__ import annotations

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def render_app_shell() -> str:
    """Return the static HTML app shell."""
    return (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
