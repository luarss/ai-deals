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
    # Arena AI (LMSYS Chatbot Arena)
    arena_elo: Optional[float] = None
    arena_ci: Optional[float] = None
    arena_votes: Optional[int] = None
    arena_coding_elo: Optional[float] = None
    # OpenRouter pricing (per-1M-tokens, converted from per-token API response)
    openrouter_price_input: Optional[float] = None
    openrouter_price_output: Optional[float] = None


class ArchivePayload(BaseModel):
    """Top-level wrapper for JSON archive serialization."""

    date: str
    generated_at: str
    insights: str
    total_models: int
    models: list["MergedModel"]
    hero: Optional["MergedModel"] = None
    runners_up: list["MergedModel"] = Field(default_factory=list)
    arena_hero: Optional["MergedModel"] = None
    arena_runners_up: list["MergedModel"] = Field(default_factory=list)


class MergedModel(RawModelRecord):
    """Post-deduplication fully merged record with computed value metrics."""

    value_score: Optional[float] = None  # intelligence / blended_price
    coding_value: Optional[float] = None  # coding_index / blended_price
    composite_deal_score: Optional[float] = None  # normalised weighted composite (AA view)
    arena_value: Optional[float] = None  # arena_elo / effective_price
    arena_coding_value: Optional[float] = None  # arena_coding_elo / effective_price
    arena_composite_score: Optional[float] = None  # normalised weighted composite (Arena view)
    param_billions: Optional[float] = None  # parameter count parsed from name
    size_tier: str = "frontier"  # nano | small | medium | large | frontier

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

    @model_validator(mode="after")
    def compute_arena_raw_scores(self) -> "MergedModel":
        """Compute Arena-based pre-normalisation ratio scores."""
        blended = self._effective_blended_price()
        if blended and blended > 0:
            if self.arena_elo is not None:
                self.arena_value = round(self.arena_elo / blended, 4)
            if self.arena_coding_elo is not None:
                self.arena_coding_value = round(self.arena_coding_elo / blended, 4)
        return self

    def _effective_blended_price(self) -> Optional[float]:
        """Return the best available blended price estimate (AA first, then OpenRouter)."""
        if self.price_blended is not None and self.price_blended > 0:
            return self.price_blended
        if self.price_input is not None and self.price_output is not None:
            # Approximate 2:1 input/output ratio when cache price unavailable
            return (2 * self.price_input + self.price_output) / 3
        if self.price_input is not None:
            return self.price_input
        # OpenRouter fallbacks
        if self.openrouter_price_input is not None and self.openrouter_price_output is not None:
            return (2 * self.openrouter_price_input + self.openrouter_price_output) / 3
        if self.openrouter_price_input is not None:
            return self.openrouter_price_input
        return None
