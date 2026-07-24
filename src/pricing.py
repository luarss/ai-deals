"""Single source of truth for effective (blended) model price.

Per 00-overview §1 this is the one place the price fallback ladder lives. It
returns the **raw** (unfloored) effective price in $/1M tokens — the real number
used for display, the frontier x-axis (spec 01), the daily diff (spec 03), and
the longitudinal trends (spec 04). The scoring-only PRICE_FLOOR lives in
``src/models.py`` and is applied *after* calling this helper, never here — flooring
a displayed or trended price would falsify the data.

The helper accepts either a plain ``dict`` (archive JSON rows, which is what the
backfill reads — the 4 oldest archives lack ``model_id`` and other fields) or a
``MergedModel`` instance, so every stage can share one ladder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid a runtime import cycle (models.py imports this module)
    from src.models import MergedModel


def _field(rec: "dict[str, Any] | MergedModel", key: str) -> Any:
    """Read ``key`` from a dict (``.get``) or a model instance (``getattr``)."""
    if isinstance(rec, dict):
        return rec.get(key)
    return getattr(rec, key, None)


def effective_price(rec: "dict[str, Any] | MergedModel") -> float | None:
    """Best-available blended $/1M-token price, raw (no floor applied).

    Fallback ladder (matches ``templates/index.html::effPrice`` and the legacy
    ``MergedModel._raw_blended_price``):

        price_blended (if > 0)
          -> (2*price_input + price_output) / 3
          -> price_input
          -> (2*openrouter_price_input + openrouter_price_output) / 3
          -> openrouter_price_input
          -> None
    """
    pb = _field(rec, "price_blended")
    if pb is not None and pb > 0:
        return float(pb)

    pi = _field(rec, "price_input")
    po = _field(rec, "price_output")
    if pi is not None and po is not None:
        return (2 * pi + po) / 3
    if pi is not None:
        return float(pi)

    ori = _field(rec, "openrouter_price_input")
    oro = _field(rec, "openrouter_price_output")
    if ori is not None and oro is not None:
        return (2 * ori + oro) / 3
    if ori is not None:
        return float(ori)

    return None
