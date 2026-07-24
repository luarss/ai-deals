"""Compute normalised value metrics and rank models.

Normalisation: log10 against pinned reference anchors, clamped to [0, 1]. The
anchors are hard-coded constants (not recomputed from the daily pool), so a
given raw value always maps to the same normalized score regardless of what
else was scraped — scores are comparable across days and outlier-robust.

Composite deal score: weighted sum of normalised components, with a neutral
(0.5) imputation for any missing leg so missing data is neither rewarded nor
punished.
"""

from __future__ import annotations

import logging
import math
import re

from src.models import MergedModel

log = logging.getLogger(__name__)

# Weights for composite_deal_score (must sum to 1.0)
W_VALUE = 0.50
W_CODING = 0.30
W_SPEED = 0.20

# Arena view weights
W_ARENA_VALUE = 0.50
W_ARENA_CODING = 0.30
W_ARENA_SPEED = 0.20

# Pinned log-normalization anchors (REF_LO, REF_HI). Day-stable, pool-independent.
# Changing any of these is a SCORING_VERSION bump (see §6/§8 of spec 05).
VS_REF = (5.0, 500.0)     # AA value_score
CV_REF = (5.0, 500.0)     # AA coding_value
SP_REF = (20.0, 1000.0)   # output_speed_tps
AV_REF = (50.0, 10000.0)  # arena_value
AC_REF = (50.0, 10000.0)  # arena_coding_value

NEUTRAL = 0.5             # imputed normalized value for a missing leg
UPTIME_GATE_PCT = 95.0    # models below this (when data present) are gated out
SCORING_VERSION = 2       # interpretability marker; mirrored in ArchivePayload

# Regex: match "70B", "0.8B", "397B A17B" (MoE — use total params), "27b" etc.
_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[Bb](?:\b|$)")

# Size-hint keywords for models that omit explicit parameter counts.
# Parenthetical effort suffixes like "(medium)" and "(low)" are stripped first
# so they don't pollute tier detection (e.g. "Gemini Flash (medium)" → unknown).
_PAREN_RE = re.compile(r"\([^)]*\)")
_SIZE_HINT_RE = re.compile(
    r"\b(nano|tiny|micro|mini|xs|small|medium|large|xl|xxl)\b",
    re.IGNORECASE,
)
_HINT_TO_TIER: dict[str, str] = {
    "nano": "nano", "tiny": "nano",
    "micro": "small", "mini": "small", "xs": "small",
    "small": "small",
    "medium": "medium",
    "large": "large", "xl": "large", "xxl": "large",
}


def _extract_param_billions(name: str) -> float | None:
    """Parse the first (largest) parameter count from a model name."""
    matches = _PARAM_RE.findall(name)
    if not matches:
        return None
    # Take the largest number found (handles "397B A17B" → 397)
    return max(float(m) for m in matches)


def _size_tier(param_b: float | None, name: str = "") -> str:
    """Map parameter count (or name keywords) to a display tier string."""
    if param_b is not None:
        if param_b < 3:
            return "nano"    # <3B
        if param_b < 15:
            return "small"   # 3–15B
        if param_b < 70:
            return "medium"  # 15–70B
        return "large"       # ≥70B
    # No explicit param count — try size-hint keywords in the model name.
    # Strip parenthetical suffixes first (e.g. "(medium)" = reasoning effort, not size).
    clean = _PAREN_RE.sub("", name)
    m = _SIZE_HINT_RE.search(clean)
    if m:
        return _HINT_TO_TIER[m.group(1).lower()]
    return "unknown"  # genuinely no parseable size info


def normalize_log(x: float | None, ref: tuple[float, float]) -> float:
    """Map x to [0, 1] via log10 against pinned (lo, hi). Missing -> NEUTRAL.

    Day-stable and pool-independent: the anchors are constants, so a given raw
    value always maps to the same normalized score. The low clamp guards
    log10(0); the high clamp winsorizes the outlier tail.
    """
    if x is None:
        return NEUTRAL
    lo, hi = ref
    x = max(x, lo)  # clamp low (also guards log10(0) / log10(negative))
    v = (math.log10(x) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    return min(1.0, max(0.0, v))  # winsorize/clamp the high tail


def annotate_sizes(models: list[MergedModel]) -> None:
    """Set param_billions and size_tier on every model (mutates in place)."""
    for model in models:
        pb = _extract_param_billions(model.name)
        model.param_billions = pb
        model.size_tier = _size_tier(pb, model.name)


def rank_models(models: list[MergedModel]) -> list[MergedModel]:
    """Add composite scores to every model and return sorted list (highest first).

    Each leg is log-normalized against pinned anchors (day-stable), and a
    missing leg is imputed at NEUTRAL (0.5) with fixed weights — no
    redistribution. A composite is computed only if at least one quality signal
    is present (speed alone is not a "deal").
    """
    if not models:
        return []

    for model in models:
        # AA composite: require >=1 quality signal (value or coding)
        if model.value_score is None and model.coding_value is None:
            model.composite_deal_score = None
        else:
            v = normalize_log(model.value_score, VS_REF)      # imputes NEUTRAL if None
            c = normalize_log(model.coding_value, CV_REF)
            s = normalize_log(model.output_speed_tps, SP_REF)
            composite = W_VALUE * v + W_CODING * c + W_SPEED * s
            model.composite_deal_score = round(composite, 4)

        # Arena composite: require >=1 arena quality signal
        if model.arena_value is None and model.arena_coding_value is None:
            model.arena_composite_score = None
        else:
            av = normalize_log(model.arena_value, AV_REF)
            ac = normalize_log(model.arena_coding_value, AC_REF)
            s = normalize_log(model.output_speed_tps, SP_REF)
            arena = W_ARENA_VALUE * av + W_ARENA_CODING * ac + W_ARENA_SPEED * s
            model.arena_composite_score = round(arena, 4)

    # Sort: models with a composite score first (descending), then by intelligence
    return sorted(
        models,
        key=lambda m: (
            m.composite_deal_score is None,  # None last
            -(m.composite_deal_score or 0),
            -(m.intelligence_score or 0),
        ),
    )


def passes_uptime_gate(m: MergedModel) -> bool:
    """True unless the model has uptime data below the gate.

    Models with no uptime data (AA-only) pass — the gate no-ops when data is
    absent so we never punish a source that simply doesn't report uptime.
    """
    vals = [u for u in (m.uptime_30m, m.uptime_1d) if u is not None]
    if not vals:
        return True
    return max(vals) >= UPTIME_GATE_PCT


def is_deal_candidate(m: MergedModel, view: str = "aa") -> bool:
    """A model is a deal candidate if it is scored for the view and passes uptime."""
    field = "arena_composite_score" if view == "arena" else "composite_deal_score"
    return getattr(m, field) is not None and passes_uptime_gate(m)


# ── Pareto frontier ──────────────────────────────────────────────────────────
# Near-duplicate collapse thresholds (spec 01 §3.4).
_DUP_LOG_PRICE = 0.02      # ~4.7% price on a log10 axis
_DUP_Q_INTEL = 0.5         # intelligence 0-60 scale
_DUP_Q_ARENA = 5.0         # Arena Elo scale


def _frontier_price(m: MergedModel) -> float | None:
    """Raw (unfloored) effective price for the frontier's price axis.

    Per 00-overview §1 the frontier plots *real* prices — the PRICE_FLOOR is a
    scoring-only device and flooring the axis would falsify the chart and squash
    the cheap corner. <= 0 is treated as missing (guards the log10 domain).
    """
    p = m._raw_blended_price()
    return p if (p is not None and p > 0) else None


def _arena_quality(m: MergedModel) -> float | None:
    """Arena view quality: coding_index first (never present today), then
    arena_elo, then arena_coding_elo."""
    if m.coding_index is not None:
        return m.coding_index
    if m.arena_elo is not None:
        return m.arena_elo
    return m.arena_coding_elo


def _pareto_frontier(
    candidates: list[tuple[MergedModel, float, float]],  # (model, price, quality)
) -> list[tuple[MergedModel, float, float]]:
    """Return the subset not dominated by any other (cheaper-or-equal AND
    higher-or-equal quality, strict in at least one)."""
    front = []
    for m, p, q in candidates:
        dominated = any(
            (p2 <= p and q2 >= q) and (p2 < p or q2 > q)
            for m2, p2, q2 in candidates
            if m2 is not m
        )
        if not dominated:
            front.append((m, p, q))
    return front


def _collapse_near_duplicates(front, q_dup, score_attr):
    """Keep one representative per near-duplicate (log-price, quality) cluster.
    Preference: lower price, higher quality, higher composite, then name."""
    front = sorted(front, key=lambda t: (t[1], -t[2], t[0].name))  # price asc, q desc, name
    kept: list[tuple[MergedModel, float, float]] = []
    for m, p, q in front:
        dup = None
        for km, kp, kq in kept:
            if abs(math.log10(p) - math.log10(kp)) < _DUP_LOG_PRICE and abs(q - kq) < q_dup:
                dup = (km, kp, kq)
                break
        if dup is None:
            kept.append((m, p, q))
        else:
            km, kp, kq = dup
            challenger_better = (
                p < kp
                or (p == kp and q > kq)
                or (
                    p == kp
                    and q == kq
                    and (getattr(m, score_attr) or 0) > (getattr(km, score_attr) or 0)
                )
            )
            if challenger_better:
                kept.remove(dup)
                kept.append((m, p, q))
    return kept


def _knee_scores(front) -> dict[int, float]:
    """frontier_score = y' - x' with x = log10(price) and y = quality both
    min-maxed over the frontier set. Returns {id(model): score}."""
    if not front:
        return {}
    xs = [math.log10(p) for _, p, _ in front]
    ys = [q for _, _, q in front]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    out = {}
    for (m, _, q), x in zip(front, xs):
        xn = 0.0 if xmax == xmin else (x - xmin) / (xmax - xmin)
        yn = 0.0 if ymax == ymin else (q - ymin) / (ymax - ymin)
        out[id(m)] = round(yn - xn, 4)
    return out


def annotate_frontier(models: list[MergedModel]) -> None:
    """Compute both Pareto frontiers and set is_frontier_*/frontier_rank_*/
    frontier_score_* on every model (mutates in place).

    Candidates need both a positive price and the view's quality axis, and must
    pass the uptime gate (spec 05 primitive; see 00-overview §4/§9.3) so a
    flaky-but-cheap model cannot claim the knee.
    """
    specs = [
        ("intel", lambda m: m.intelligence_score, _DUP_Q_INTEL, "composite_deal_score"),
        ("arena", _arena_quality, _DUP_Q_ARENA, "arena_composite_score"),
    ]
    for view, qfn, q_dup, score_attr in specs:
        # Reset flags so repeated calls / cross-view reuse stay consistent.
        for m in models:
            setattr(m, f"is_frontier_{view}", False)
            setattr(m, f"frontier_rank_{view}", None)
            setattr(m, f"frontier_score_{view}", None)

        candidates = []
        for m in models:
            if not passes_uptime_gate(m):
                continue
            p = _frontier_price(m)
            q = qfn(m)
            if p is not None and q is not None:
                candidates.append((m, p, q))

        front = _pareto_frontier(candidates)
        front = _collapse_near_duplicates(front, q_dup, score_attr)
        scores = _knee_scores(front)
        # Rank by frontier_score desc; break ties by name for determinism.
        ranked = sorted(front, key=lambda t: (-scores[id(t[0])], t[0].name))
        for rank, (m, p, q) in enumerate(ranked, start=1):
            setattr(m, f"is_frontier_{view}", True)
            setattr(m, f"frontier_rank_{view}", rank)
            setattr(m, f"frontier_score_{view}", scores[id(m)])


def compute_top_models(models: list[MergedModel], view: str = "aa") -> list[MergedModel]:
    """Return frontier members ordered by frontier_score desc (hero first).

    Falls back to the previous uptime-gated composite ordering only if the
    frontier is empty (e.g. a day with no priced+scored models, or when
    annotate_frontier has not been run) so the page never goes empty.
    """
    v = "arena" if view == "arena" else "intel"
    rank_attr = f"frontier_rank_{v}"
    on_front = [m for m in models if getattr(m, rank_attr) is not None]
    if on_front:
        return sorted(on_front, key=lambda m: getattr(m, rank_attr))
    # Fallback: previous behaviour (uptime-gated composite order, no tier filter).
    field = "arena_composite_score" if view == "arena" else "composite_deal_score"
    candidates = [m for m in models if is_deal_candidate(m, view)]
    return sorted(candidates, key=lambda m: -(getattr(m, field) or 0))
