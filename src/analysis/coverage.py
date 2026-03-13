"""Coverage scoring and flagging for SemanticAnalysis extractions.

Computes dimension coverage scores and generates flags based on how many
of the 40 question dimensions received non-None answers. Research Core
dimensions (q01-q05) are tracked separately for core gap detection.

Coverage tiers:
    Full:     >= 85% (34+/40)  - Normal
    Partial:  60-84% (24-33/40) - Expected for non-research docs
    Sparse:   30-59% (12-23/40) - Review recommended
    Critical: < 30% (< 12/40)  - Likely extraction failure
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.analysis.schemas import SemanticAnalysis

logger = logging.getLogger(__name__)

# All 40 question field names in order
QUESTION_FIELDS: tuple[str, ...] = tuple(f"q{i:02d}" for i in range(1, 41))

# Question field names with their attribute suffixes on SemanticAnalysis
QUESTION_ATTRS: tuple[str, ...] = (
    "q01_research_question",
    "q02_thesis",
    "q03_key_claims",
    "q04_evidence",
    "q05_limitations",
    "q06_paradigm",
    "q07_methods",
    "q08_data",
    "q09_reproducibility",
    "q10_framework",
    "q11_traditions",
    "q12_key_citations",
    "q13_assumptions",
    "q14_counterarguments",
    "q15_novelty",
    "q16_stance",
    "q17_field",
    "q18_audience",
    "q19_implications",
    "q20_future_work",
    "q21_quality",
    "q22_contribution",
    "q23_source_type",
    "q24_other",
    "q25_institutional_context",
    "q26_historical_timing",
    "q27_paradigm_influence",
    "q28_disciplines_bridged",
    "q29_cross_domain_insights",
    "q30_cultural_scope",
    "q31_philosophical_assumptions",
    "q32_deployment_gap",
    "q33_infrastructure_contribution",
    "q34_power_dynamics",
    "q35_gaps_and_omissions",
    "q36_dual_use_concerns",
    "q37_emergence_claims",
    "q38_remaining_other",
    "q39_network_properties",
    "q40_policy_recommendations",
)

TOTAL_DIMENSIONS = len(QUESTION_ATTRS)  # 40

# Research Core dimensions (q01-q05) for core gap detection
CORE_ATTRS: tuple[str, ...] = QUESTION_ATTRS[:5]

# Coverage tier thresholds (as fractions)
TIER_FULL = 0.85
TIER_PARTIAL = 0.60
TIER_SPARSE = 0.30

# Flag constants
FLAG_PARTIAL_COVERAGE = "PARTIAL_COVERAGE"
FLAG_SPARSE_COVERAGE = "SPARSE_COVERAGE"
FLAG_CRITICAL_GAPS = "CRITICAL_GAPS"
FLAG_CORE_GAPS = "CORE_GAPS"


@dataclass(frozen=True)
class CoverageResult:
    """Result of coverage scoring for a single SemanticAnalysis."""

    paper_id: str
    answered: int
    total: int
    coverage: float
    tier: str
    flags: list[str]
    missing_core: list[str]

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "answered": self.answered,
            "total": self.total,
            "coverage": round(self.coverage, 4),
            "tier": self.tier,
            "flags": self.flags,
            "missing_core": self.missing_core,
        }


def score_coverage(analysis: SemanticAnalysis) -> CoverageResult:
    """Compute coverage score and flags for a SemanticAnalysis.

    Args:
        analysis: A SemanticAnalysis instance to score.

    Returns:
        CoverageResult with coverage fraction, tier, and flags.
    """
    answered = 0
    for attr in QUESTION_ATTRS:
        if getattr(analysis, attr, None) is not None:
            answered += 1

    coverage = answered / TOTAL_DIMENSIONS

    # Determine tier
    if coverage >= TIER_FULL:
        tier = "full"
    elif coverage >= TIER_PARTIAL:
        tier = "partial"
    elif coverage >= TIER_SPARSE:
        tier = "sparse"
    else:
        tier = "critical"

    # Build flags
    flags: list[str] = []
    if tier == "partial":
        flags.append(FLAG_PARTIAL_COVERAGE)
    elif tier == "sparse":
        flags.append(FLAG_SPARSE_COVERAGE)
    elif tier == "critical":
        flags.append(FLAG_CRITICAL_GAPS)

    # Check core dimensions (q01-q05)
    missing_core: list[str] = []
    for attr in CORE_ATTRS:
        if getattr(analysis, attr, None) is None:
            missing_core.append(attr)

    if missing_core:
        flags.append(FLAG_CORE_GAPS)

    return CoverageResult(
        paper_id=analysis.paper_id,
        answered=answered,
        total=TOTAL_DIMENSIONS,
        coverage=coverage,
        tier=tier,
        flags=flags,
        missing_core=missing_core,
    )


def apply_coverage(analysis: SemanticAnalysis) -> SemanticAnalysis:
    """Score coverage and set dimension_coverage and coverage_flags on the analysis.

    Mutates the analysis in place and returns it for chaining.

    Args:
        analysis: A SemanticAnalysis instance to update.

    Returns:
        The same analysis instance with coverage fields populated.
    """
    result = score_coverage(analysis)
    analysis.dimension_coverage = result.coverage
    analysis.coverage_flags = list(result.flags)
    return analysis


def generate_coverage_report(
    analyses: list[SemanticAnalysis],
    output_path: Path | str | None = None,
) -> dict:
    """Generate a coverage report for a batch of SemanticAnalysis results.

    Args:
        analyses: List of SemanticAnalysis instances to report on.
        output_path: If provided, write the report as JSON to this path.
            Defaults to data/logs/extraction_review.json if None.

    Returns:
        Report dictionary with summary statistics and per-paper details.
    """
    results = [score_coverage(a) for a in analyses]

    # Tier counts
    tier_counts = {"full": 0, "partial": 0, "sparse": 0, "critical": 0}
    for r in results:
        tier_counts[r.tier] += 1

    # Papers with core gaps
    core_gap_papers = [r.paper_id for r in results if r.missing_core]

    # Average coverage
    avg_coverage = sum(r.coverage for r in results) / len(results) if results else 0.0

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_papers": len(results),
        "average_coverage": round(avg_coverage, 4),
        "tier_distribution": tier_counts,
        "core_gap_count": len(core_gap_papers),
        "core_gap_papers": core_gap_papers,
        "papers": [r.to_dict() for r in results],
    }

    if output_path is None:
        output_path = Path("data/logs/extraction_review.json")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    logger.info(
        "Coverage report written to %s (%d papers, avg %.1f%%)",
        output_path,
        len(results),
        avg_coverage * 100,
    )

    return report
