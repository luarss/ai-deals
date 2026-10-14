from __future__ import annotations

import logging
import os

from pydantic import BaseModel, Field

from src.insights import FALLBACK as INSIGHTS_FALLBACK
from src.models import BandDeal, Changelog, MergedModel
from src.pricing import effective_price

log = logging.getLogger(__name__)

SKIP_ENV_VAR = "AI_DEALS_SKIP_HEALTH_GATE"

MIN_AA_RECORDS = 150
MIN_OPENROUTER_RECORDS = 200
MIN_ARENA_RECORDS = 50
MIN_SCORED_MODELS = 30
MIN_MODELS_WITH_INTELLIGENCE = 50
MIN_MODELS_WITH_AA_PRICE = 50
MIN_FRONTIER_MODELS = 1
MIN_BAND_MEMBERS = 1


class HealthCheck(BaseModel):
    name: str
    value: int
    floor: int
    severity: str
    passed: bool


class HealthReport(BaseModel):
    source_counts: dict[str, int]
    merged_count: int
    with_intelligence: int
    with_effective_price: int
    with_aa_price: int
    scored_count: int
    frontier_count: int
    band_counts: dict[str, int]
    insights_length: int
    insights_is_fallback: bool
    hero_present: bool
    checks: list[HealthCheck] = Field(default_factory=list)

    @property
    def hard_failures(self) -> list[HealthCheck]:
        return [check for check in self.checks if check.severity == "hard" and not check.passed]

    @property
    def soft_failures(self) -> list[HealthCheck]:
        return [check for check in self.checks if check.severity == "soft" and not check.passed]


def build_health_report(
    aa_count: int,
    arena_count: int,
    openrouter_count: int,
    ranked: list[MergedModel],
    bands: list[BandDeal],
    insights: str,
    hero: MergedModel | None,
) -> HealthReport:
    report = HealthReport(
        source_counts={"aa": aa_count, "arena": arena_count, "openrouter": openrouter_count},
        merged_count=len(ranked),
        with_intelligence=sum(1 for model in ranked if model.intelligence_score is not None),
        with_effective_price=sum(1 for model in ranked if effective_price(model) is not None),
        with_aa_price=sum(1 for model in ranked if model.price_blended is not None),
        scored_count=sum(1 for model in ranked if model.composite_deal_score is not None),
        frontier_count=sum(1 for model in ranked if model.is_frontier_intel),
        band_counts={band.id: band.count for band in bands},
        insights_length=len(insights.strip()),
        insights_is_fallback=insights.strip() == INSIGHTS_FALLBACK.strip(),
        hero_present=hero is not None,
    )
    report.checks = _evaluate_checks(report)
    return report


def _check(name: str, value: int, floor: int, severity: str) -> HealthCheck:
    return HealthCheck(name=name, value=value, floor=floor, severity=severity, passed=value >= floor)


def _evaluate_checks(report: HealthReport) -> list[HealthCheck]:
    usable_insights = 0 if report.insights_is_fallback else report.insights_length
    checks = [
        _check("aa", report.source_counts["aa"], MIN_AA_RECORDS, "hard"),
        _check("openrouter", report.source_counts["openrouter"], MIN_OPENROUTER_RECORDS, "hard"),
        _check("scored_nonzero", report.scored_count, 1, "hard"),
        _check("hero", int(report.hero_present), 1, "hard"),
        _check("arena", report.source_counts["arena"], MIN_ARENA_RECORDS, "soft"),
        _check("scored", report.scored_count, MIN_SCORED_MODELS, "soft"),
        _check("intelligence", report.with_intelligence, MIN_MODELS_WITH_INTELLIGENCE, "soft"),
        _check("aa_prices", report.with_aa_price, MIN_MODELS_WITH_AA_PRICE, "soft"),
        _check("frontier", report.frontier_count, MIN_FRONTIER_MODELS, "soft"),
        _check("insights", usable_insights, 1, "soft"),
    ]
    if not report.band_counts:
        checks.append(_check("bands", 0, 1, "soft"))
    for band_id, count in report.band_counts.items():
        checks.append(_check(f"band:{band_id}", count, MIN_BAND_MEMBERS, "soft"))
    return checks


def apply_to_changelog(report: HealthReport, changelog: Changelog) -> None:
    changelog.health = report.model_dump(mode="json")
    soft_names = [check.name for check in report.soft_failures]
    if not soft_names:
        return
    if changelog.data_quality is None:
        changelog.data_quality = "partial"
    changelog.degraded_sources = sorted(set(changelog.degraded_sources) | set(soft_names))


def log_report(report: HealthReport) -> None:
    log.info("Health report: %s", report.model_dump_json())
    for check in report.checks:
        status = "ok" if check.passed else "FAIL"
        log.info(
            "  [%s] %-18s %s value=%d floor=%d",
            check.severity,
            check.name,
            status,
            check.value,
            check.floor,
        )
    for check in report.soft_failures:
        print(
            f"::warning title=Data health::{check.name} below floor "
            f"(value={check.value}, floor={check.floor})",
            flush=True,
        )
    for check in report.hard_failures:
        print(
            f"::error title=Data health::{check.name} below hard floor "
            f"(value={check.value}, floor={check.floor})",
            flush=True,
        )


def gate_is_skipped() -> bool:
    return os.environ.get(SKIP_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}


def enforce(report: HealthReport) -> None:
    if not report.hard_failures:
        return
    names = ", ".join(check.name for check in report.hard_failures)
    if gate_is_skipped():
        log.warning("Hard health checks failed (%s) but %s is set; continuing", names, SKIP_ENV_VAR)
        return
    log.error("Hard health checks failed (%s); aborting before writing archives", names)
    raise SystemExit(1)
