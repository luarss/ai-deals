"""Pydantic data models shared across all pipeline stages."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, model_validator

# $/1M tokens. Below this the intelligence/price ratio is degenerate (a
# near-free but near-worthless model wins purely on the ratio's tail).
# Applied ONLY inside scoring ratios via _effective_blended_price(); the raw
# price (_raw_blended_price) is never floored so displayed/trended prices stay
# truthful.
PRICE_FLOOR = 0.05


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
    openrouter_model_id: Optional[str] = None  # raw OR id, e.g. "openai/gpt-4o"
    # Modality (from OpenRouter architecture field)
    input_modalities: list[str] = Field(default_factory=list)
    output_modalities: list[str] = Field(default_factory=list)
    # Uptime (best across all OpenRouter endpoints, last 30 min / 1 day)
    uptime_30m: Optional[float] = None
    uptime_1d: Optional[float] = None


class BandDeal(BaseModel):
    """One intelligence band's leaderboard summary (references models by model_id).

    Band membership is keyed on intelligence_score only (never size_tier). See
    src/bands.py for the threshold definitions (the single source of truth).
    """

    id: str  # "flagship" | "workhorse" | "budget"
    label: str  # display label, e.g. "Frontier-class"
    min_intelligence: float  # inclusive floor
    max_intelligence: Optional[float] = None  # exclusive ceiling; None = open-ended
    count: int = 0  # number of eligible members in this band
    leader_id: Optional[str] = None  # best composite_deal_score in band (uptime-gated)
    cheapest_id: Optional[str] = None  # lowest raw effective price in band
    member_ids: list[str] = Field(default_factory=list)  # top-N by composite, incl. leader


class CheapestAbove(BaseModel):
    """Cheapest model with intelligence >= threshold (cumulative, not band-bounded)."""

    threshold: float
    model_id: Optional[str] = None
    effective_price: Optional[float] = None  # raw (unfloored) price


class ChangeEvent(BaseModel):
    """One entry in the daily changelog (spec 03).

    A single flat shape serves every event kind; only the fields relevant to a
    given `kind` are populated, the rest stay None so old archives and mixed
    event lists round-trip cleanly.
    """

    model_id: str
    name: str
    kind: str  # new | removed | price | intelligence | speed | rank | frontier
    # populated per-kind; all optional so one class serves every event type
    intelligence_score: Optional[float] = None
    price: Optional[float] = None
    composite_deal_score: Optional[float] = None
    old: Optional[float] = None
    new: Optional[float] = None
    delta: Optional[float] = None
    pct: Optional[float] = None
    direction: Optional[str] = None  # cut/hike, up/down, entered/left
    old_rank: Optional[int] = None
    new_rank: Optional[int] = None
    entered_top: bool = False
    left_top: bool = False


class Changelog(BaseModel):
    """Structured day-over-day diff embedded in each archive (spec 03)."""

    compared_to: Optional[str] = None  # previous archive date, or None
    first_run: bool = False
    # True when the previous archive's scoring_version differs from today's, so
    # score-derived events (rank/frontier) are suppressed to avoid a fake flood.
    scoring_changed: bool = False
    data_quality: Optional[str] = None  # None | "partial" | "degraded"
    degraded_sources: list[str] = Field(default_factory=list)
    total_events: int = 0
    new_models: list["ChangeEvent"] = Field(default_factory=list)
    removed_models: list["ChangeEvent"] = Field(default_factory=list)
    price_changes: list["ChangeEvent"] = Field(default_factory=list)
    intelligence_changes: list["ChangeEvent"] = Field(default_factory=list)
    speed_changes: list["ChangeEvent"] = Field(default_factory=list)
    rank_changes: list["ChangeEvent"] = Field(default_factory=list)
    frontier_changes: list["ChangeEvent"] = Field(default_factory=list)


class ArchivePayload(BaseModel):
    """Top-level wrapper for JSON archive serialization."""

    scoring_version: int = 2
    date: str
    generated_at: str
    insights: str
    total_models: int
    models: list["MergedModel"]
    hero: Optional["MergedModel"] = None
    runners_up: list["MergedModel"] = Field(default_factory=list)
    arena_hero: Optional["MergedModel"] = None
    arena_runners_up: list["MergedModel"] = Field(default_factory=list)
    # Explicit frontier arrays for the chart (price-ascending). Optional: old
    # archives lack them and load as [], so the frontend hides the chart.
    frontier_intel: list["MergedModel"] = Field(default_factory=list)
    frontier_arena: list["MergedModel"] = Field(default_factory=list)
    # Intelligence-band leaderboards (spec 02). Optional: old archives lack them
    # and load as [], so the frontend falls back to the legacy single hero.
    bands: list["BandDeal"] = Field(default_factory=list)
    cheapest_above: list["CheapestAbove"] = Field(default_factory=list)
    # Day-over-day changelog (spec 03). Optional: the 62 pre-03 archives lack the
    # key and load as None, so the frontend hides the changelog headline for them.
    changelog: Optional["Changelog"] = None


class MergedModel(RawModelRecord):
    """Post-deduplication fully merged record with computed value metrics."""

    value_score: Optional[float] = None  # intelligence / blended_price
    coding_value: Optional[float] = None  # coding_index / blended_price
    composite_deal_score: Optional[float] = None  # normalised weighted composite (AA view)
    arena_value: Optional[float] = None  # arena_elo / effective_price
    arena_coding_value: Optional[float] = None  # arena_coding_elo / effective_price
    arena_composite_score: Optional[float] = None  # normalised weighted composite (Arena view)
    param_billions: Optional[float] = None  # parameter count parsed from name
    size_tier: str = "unknown"  # nano | small | medium | large | unknown
    # --- Pareto frontier (intelligence-vs-price, "aa" view) ---
    # All default None/False so pre-frontier archives deserialize unchanged.
    is_frontier_intel: bool = False
    frontier_rank_intel: Optional[int] = None      # 1 = hero (knee), by frontier_score desc
    frontier_score_intel: Optional[float] = None    # knee score y'-x' over the frontier set
    # --- Pareto frontier (arena/coding-vs-price, "arena" view) ---
    is_frontier_arena: bool = False
    frontier_rank_arena: Optional[int] = None
    frontier_score_arena: Optional[float] = None

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

    def effective_price(self) -> Optional[float]:
        """Public alias for the raw (unfloored) best-available blended price.

        Mirrors the frontend's effPrice() and is the price used for display and
        for band/cheapest comparisons (spec 02). Scoring ratios use the floored
        _effective_blended_price() instead; see 00-overview §1.
        """
        return self._raw_blended_price()

    def _effective_blended_price(self) -> Optional[float]:
        """Return the floored blended price used for every scoring ratio.

        Applies PRICE_FLOOR so the intel/price ratio does not explode for
        near-free models. Spec 04 will extract this into src/pricing.py; until
        then this is the single floored-price definition for scoring.
        """
        raw = self._raw_blended_price()
        if raw is None:
            return None
        return max(raw, PRICE_FLOOR)

    def _raw_blended_price(self) -> Optional[float]:
        """Return the best available blended price estimate (AA first, then OpenRouter).

        This is the *raw* effective price with no floor applied — safe for
        display and trending. Scoring must use _effective_blended_price().

        Delegates to src.pricing.effective_price (00-overview §1) so the ladder
        is defined in exactly one place; a parity test locks the two together.
        """
        from src.pricing import effective_price

        return effective_price(self)
