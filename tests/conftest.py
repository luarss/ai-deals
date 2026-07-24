"""Shared test fixtures and archive-reconstruction helper."""

from __future__ import annotations

import json
from pathlib import Path

from src.models import MergedModel, RawModelRecord

REPO_ROOT = Path(__file__).resolve().parent.parent
_RAW_FIELDS = set(RawModelRecord.model_fields.keys())


def reconstruct_models(archive_path: Path) -> list[MergedModel]:
    """Rebuild MergedModel objects from a stored archive's raw fields.

    Only the persisted *raw* inputs are used; MergedModel's validators recompute
    value_score / coding_value / arena_* from scratch, so this exercises the
    current scoring code against real data.
    """
    data = json.loads(archive_path.read_text(encoding="utf-8"))
    models: list[MergedModel] = []
    for record in data["models"]:
        raw = {k: v for k, v in record.items() if k in _RAW_FIELDS}
        models.append(MergedModel(**raw))
    return models
