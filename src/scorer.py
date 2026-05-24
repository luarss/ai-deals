"""Compute normalised value metrics and rank models.

Normalisation: (x - min) / (max - min) across all models with a non-None value.
Composite deal score: weighted sum of normalised components.
"""

from __future__ import annotations

import logging

from src.models import MergedModel

log = logging.getLogger(__name__)

# Weights for composite_deal_score (must sum to 1.0)
W_VALUE = 0.50
W_CODING = 0.30
W_SPEED = 0.20


def _safe_values(models: list[MergedModel], attr: str) -> list[float]:
    return [v for m in models if (v := getattr(m, attr)) is not None]


def _normalise(value: float, vmin: float, vmax: float) -> float:
    """Map value into [0, 1]; returns 1.0 if min == max (single-model edge case)."""
    if vmax == vmin:
        return 1.0
    return (value - vmin) / (vmax - vmin)


def _build_range(values: list[float]) -> tuple[float, float]:
    return (min(values), max(values)) if values else (0.0, 1.0)


def rank_models(models: list[MergedModel]) -> list[MergedModel]:
    """Add composite_deal_score to every model and return sorted list (highest first)."""
    if not models:
        return []

    # Compute normalisation ranges from non-None values
    vs_range = _build_range(_safe_values(models, "value_score"))
    cv_range = _build_range(_safe_values(models, "coding_value"))
    sp_range = _build_range(_safe_values(models, "output_speed_tps"))

    for model in models:
        parts: list[tuple[float, float]] = []  # (weight, normalised_score)

        if model.value_score is not None:
            parts.append((W_VALUE, _normalise(model.value_score, *vs_range)))

        if model.coding_value is not None:
            parts.append((W_CODING, _normalise(model.coding_value, *cv_range)))
        elif model.value_score is not None:
            # Redistribute coding weight onto value when coding data is absent
            parts = [(W_VALUE + W_CODING, _normalise(model.value_score, *vs_range))]

        if model.output_speed_tps is not None:
            parts.append((W_SPEED, _normalise(model.output_speed_tps, *sp_range)))

        if parts:
            total_weight = sum(w for w, _ in parts)
            composite = sum(w * s for w, s in parts) / total_weight
            model.composite_deal_score = round(composite, 4)
        else:
            model.composite_deal_score = None

    # Sort: models with a composite score first (descending), then by intelligence
    return sorted(
        models,
        key=lambda m: (
            m.composite_deal_score is None,  # None last
            -(m.composite_deal_score or 0),
            -(m.intelligence_score or 0),
        ),
    )
