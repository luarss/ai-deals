"""Pydantic data models shared across all pipeline stages."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, model_validator


class RawModelRecord(BaseModel):
    """Populated from a single JSON-LD dataset or HTML table row."""

    model_id: str  # slug from href, e.g. "gpt-4o-mini"
    name: str
    creator: Optional[str] = None
    intelligence_score: Optional[float] = None  # 0–60 AA Intelligence Index
    price_input: Optional[float] = None  # $/1M input tokens
    price_output: Optional[float] = None  # $/1M output tokens
    price_blended: Optional[float] = None  # preferred; AA 7:2:1 cache/in/out
    output_speed_tps: Optional[float] = None  # tokens per second
    context_window_k: Optional[int] = None  # context window in thousands of tokens
    coding_index: Optional[float] = None  # AA Coding Index score
    livecodebench_score: Optional[float] = None  # LiveCodeBench %
    source_pages: list[str] = Field(default_factory=list)


class MergedModel(RawModelRecord):
    """Post-deduplication fully merged record with computed value metrics."""

    value_score: Optional[float] = None  # intelligence / blended_price
    coding_value: Optional[float] = None  # coding_index / blended_price
    composite_deal_score: Optional[float] = None  # normalised weighted composite

    @model_validator(mode="after")
    def compute_raw_scores(self) -> "MergedModel":
        """Compute pre-normalisation ratio scores (normalisation happens in scorer.py)."""
        blended = self._effective_blended_price()
        if blended and blended > 0:
            if self.intelligence_score is not None:
                self.value_score = round(self.intelligence_score / blended, 4)
            if self.coding_index is not None:
                self.coding_value = round(self.coding_index / blended, 4)
        return self

    def _effective_blended_price(self) -> Optional[float]:
        """Return the best available blended price estimate."""
        if self.price_blended is not None and self.price_blended > 0:
            return self.price_blended
        if self.price_input is not None and self.price_output is not None:
            # Approximate 2:1 input/output ratio when cache price unavailable
            return (2 * self.price_input + self.price_output) / 3
        if self.price_input is not None:
            return self.price_input
        return None
