"""LLM Council for multi-provider consensus extraction.

This module implements a council approach where multiple LLM providers
are queried in parallel and their responses are aggregated into a
consensus extraction.

The approach is inspired by karpathy/llm-council but simplified for
structured extraction tasks:

1. Fan out: Query all configured providers simultaneously
2. Collect: Gather successful responses with error handling
3. Aggregate: Build consensus using field-specific strategies

Reference: https://github.com/karpathy/llm-council
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.analysis.schemas import (
    KeyClaim,
    KeyFinding,
    Methodology,
    PaperExtraction,
)

logger = logging.getLogger(__name__)


def _normalize_discipline_tags(tags: list[str] | None) -> list[str]:
    """Normalize discipline tags to lowercase and deduplicate.

    Args:
        tags: Raw discipline tags from union merge.

    Returns:
        Normalized, deduplicated list of discipline tags.
    """
    if not tags:
        return []

    normalized = []
    seen = set()
    for tag in tags:
        if not isinstance(tag, str):
            continue
        clean = tag.lower().strip()
        if clean and clean not in seen:
            normalized.append(clean)
            seen.add(clean)
    return normalized


class ConsensusStrategy(str, Enum):
    """Strategy for reaching consensus on a field."""

    MAJORITY_VOTE = "majority_vote"  # Most common value wins
    UNION = "union"  # Combine all unique values
    INTERSECTION = "intersection"  # Keep only values all providers agree on
    LONGEST = "longest"  # Take the longest/most detailed response
    FIRST_VALID = "first_valid"  # Take first non-empty response
    AVERAGE = "average"  # Average numeric values
    WEIGHTED = "weighted"  # Weight by provider reliability


@dataclass
class ProviderConfig:
    """Configuration for a council provider."""

    name: str
    weight: float = 1.0  # Reliability weight for weighted consensus
    timeout: int = 120  # Per-provider timeout in seconds
    max_cost: float | None = None  # Max cost per extraction (USD)
    enabled: bool = True


@dataclass
class CouncilConfig:
    """Configuration for the LLM council."""

    providers: list[ProviderConfig] = field(default_factory=list)
    min_responses: int = 2  # Minimum responses needed for consensus
    fallback_to_single: bool = True  # Use single response if min not met
    parallel: bool = True  # Run providers in parallel
    timeout: int = 180  # Overall timeout for council
    consensus_threshold: float = 0.5  # Threshold for majority consensus


@dataclass
class ProviderResponse:
    """Response from a single provider."""

    provider: str
    extraction: PaperExtraction | None
    success: bool
    error: str | None = None
    duration_seconds: float = 0.0
    cost: float = 0.0


@dataclass
class CouncilResult:
    """Result of council consensus extraction."""

    paper_id: str
    consensus: PaperExtraction | None
    provider_responses: list[ProviderResponse]
    success: bool
    consensus_confidence: float = 0.0
    total_duration_seconds: float = 0.0
    total_cost: float = 0.0
    errors: list[str] = field(default_factory=list)


# Field-specific consensus strategies
FIELD_STRATEGIES: dict[str, ConsensusStrategy] = {
    # String fields - take longest/most detailed
    "thesis_statement": ConsensusStrategy.LONGEST,
    "theoretical_framework": ConsensusStrategy.LONGEST,
    "conclusions": ConsensusStrategy.LONGEST,
    "contribution_summary": ConsensusStrategy.LONGEST,
    "extraction_notes": ConsensusStrategy.LONGEST,
    # List fields - union for coverage
    "research_questions": ConsensusStrategy.UNION,
    "limitations": ConsensusStrategy.UNION,
    "future_directions": ConsensusStrategy.UNION,
    "keywords": ConsensusStrategy.UNION,
    "discipline_tags": ConsensusStrategy.UNION,
    # Nested objects handled specially
    "methodology": ConsensusStrategy.MAJORITY_VOTE,
    "key_findings": ConsensusStrategy.UNION,
    "key_claims": ConsensusStrategy.UNION,
    # Numeric - average
    "extraction_confidence": ConsensusStrategy.AVERAGE,
}


def _majority_vote_string(values: list[str | None]) -> str | None:
    """Return the most common non-None string value."""
    valid = [v for v in values if v]
    if not valid:
        return None
    from collections import Counter

    counts = Counter(valid)
    return counts.most_common(1)[0][0]


def _longest_string(values: list[str | None]) -> str | None:
    """Return the longest non-None string."""
    valid = [v for v in values if v]
    if not valid:
        return None
    return max(valid, key=len)


def _union_lists(lists: list[list[str]]) -> list[str]:
    """Return union of all lists, preserving order of first occurrence."""
    seen = set()
    result = []
    for lst in lists:
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
    return result


def _merge_methodology(methodologies: list[Methodology]) -> Methodology:
    """Merge methodology objects from multiple providers."""
    if not methodologies:
        return Methodology()

    # Collect all values for each field
    approaches = [m.approach for m in methodologies if m.approach]
    designs = [m.design for m in methodologies if m.design]
    data_sources = [m.data_sources for m in methodologies]
    analysis_methods = [m.analysis_methods for m in methodologies]
    sample_sizes = [m.sample_size for m in methodologies if m.sample_size]
    time_periods = [m.time_period for m in methodologies if m.time_period]

    return Methodology(
        approach=_majority_vote_string(approaches),
        design=_majority_vote_string(designs),
        data_sources=_union_lists(data_sources),
        analysis_methods=_union_lists(analysis_methods),
        sample_size=_longest_string(sample_sizes),
        time_period=_longest_string(time_periods),
    )


def _merge_key_findings(findings_lists: list[list[KeyFinding]]) -> list[KeyFinding]:
    """Merge key findings, deduplicating by similarity."""
    all_findings = []
    for findings in findings_lists:
        all_findings.extend(findings)

    if not all_findings:
        return []

    # Deduplicate by finding text similarity
    unique_findings = []
    seen_findings = set()

    for finding in all_findings:
        # Normalize for comparison
        normalized = finding.finding.lower().strip()
        if normalized not in seen_findings:
            seen_findings.add(normalized)
            unique_findings.append(finding)

    return unique_findings


def _merge_key_claims(claims_lists: list[list[KeyClaim]]) -> list[KeyClaim]:
    """Merge key claims, deduplicating by similarity."""
    all_claims = []
    for claims in claims_lists:
        all_claims.extend(claims)

    if not all_claims:
        return []

    # Deduplicate by claim text similarity
    unique_claims = []
    seen_claims = set()

    for claim in all_claims:
        normalized = claim.claim.lower().strip()
        if normalized not in seen_claims:
            seen_claims.add(normalized)
            unique_claims.append(claim)

    return unique_claims


def aggregate_extractions(
    extractions: list[PaperExtraction],
    weights: list[float] | None = None,
) -> PaperExtraction:
    """Aggregate multiple extractions into consensus.

    Args:
        extractions: List of extractions from different providers.
        weights: Optional weights for each extraction (provider reliability).

    Returns:
        Consensus PaperExtraction.
    """
    if not extractions:
        return PaperExtraction()

    if len(extractions) == 1:
        return extractions[0]

    # Apply default equal weights if not provided
    if weights is None:
        weights = [1.0] * len(extractions)

    # Build consensus for each field
    return PaperExtraction(
        # String fields - longest
        thesis_statement=_longest_string([e.thesis_statement for e in extractions]),
        theoretical_framework=_longest_string(
            [e.theoretical_framework for e in extractions]
        ),
        conclusions=_longest_string([e.conclusions for e in extractions]),
        contribution_summary=_longest_string(
            [e.contribution_summary for e in extractions]
        ),
        extraction_notes=_longest_string([e.extraction_notes for e in extractions]),
        # List fields - union
        research_questions=_union_lists([e.research_questions for e in extractions]),
        limitations=_union_lists([e.limitations for e in extractions]),
        future_directions=_union_lists([e.future_directions for e in extractions]),
        keywords=_union_lists([e.keywords for e in extractions]),
        discipline_tags=_normalize_discipline_tags(_union_lists([e.discipline_tags for e in extractions])),
        # Nested objects
        methodology=_merge_methodology([e.methodology for e in extractions]),
        key_findings=_merge_key_findings([e.key_findings for e in extractions]),
        key_claims=_merge_key_claims([e.key_claims for e in extractions]),
        # Numeric - weighted average
        extraction_confidence=sum(
            e.extraction_confidence * w for e, w in zip(extractions, weights, strict=True)
        )
        / (sum(weights) if sum(weights) > 0 else 1),
    )


def calculate_consensus_confidence(
    extractions: list[PaperExtraction],
    config: CouncilConfig,
) -> float:
    """Calculate confidence score for the consensus.

    Higher confidence when:
    - More providers responded successfully
    - Providers agree on key fields

    Args:
        extractions: List of successful extractions.
        config: Council configuration.

    Returns:
        Confidence score between 0 and 1.
    """
    if not extractions:
        return 0.0

    n_providers = len(config.providers)
    n_responses = len(extractions)

    # Base confidence from response rate
    response_rate = n_responses / n_providers if n_providers > 0 else 0

    # Agreement bonus for key fields
    agreement_scores = []

    # Check thesis agreement
    theses = [e.thesis_statement for e in extractions if e.thesis_statement]
    if len(theses) > 1:
        # Simple check: do they start similarly?
        first_words = [t.split()[:5] for t in theses]
        agreement = len({tuple(w) for w in first_words}) / len(first_words)
        agreement_scores.append(1 - agreement)  # Higher when more agreement

    # Check keyword overlap
    keyword_sets = [set(e.keywords) for e in extractions if e.keywords]
    if len(keyword_sets) > 1:
        intersection = set.intersection(*keyword_sets) if keyword_sets else set()
        union = set.union(*keyword_sets) if keyword_sets else set()
        jaccard = len(intersection) / len(union) if union else 0
        agreement_scores.append(jaccard)

    avg_agreement = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.5

    # Combine response rate and agreement
    confidence = 0.6 * response_rate + 0.4 * avg_agreement
    return min(confidence, 1.0)


class LLMCouncil:
    """LLM Council for consensus-based extraction.

    Example usage:

        from src.analysis.llm_council import LLMCouncil, CouncilConfig, ProviderConfig
        from src.analysis.llm_factory import create_llm_client

        config = CouncilConfig(
            providers=[
                ProviderConfig(name="anthropic", weight=1.2),
                ProviderConfig(name="openai", weight=1.0),
                ProviderConfig(name="google", weight=0.8),
            ],
            min_responses=2,
        )

        council = LLMCouncil(config)
        result = council.extract(paper_id, title, authors, year, item_type, text)

        if result.success:
            consensus = result.consensus
            print(f"Confidence: {result.consensus_confidence}")
    """

    def __init__(self, config: CouncilConfig):
        """Initialize the council.

        Args:
            config: Council configuration.
        """
        self.config = config
        self._clients: dict[str, Any] = {}

    def _get_client(self, provider_name: str) -> Any:
        """Get or create LLM client for provider."""
        if provider_name not in self._clients:
            from src.analysis.llm_factory import create_llm_client

            self._clients[provider_name] = create_llm_client(provider=provider_name)
        return self._clients[provider_name]

    def _extract_single(
        self,
        provider: ProviderConfig,
        paper_id: str,
        title: str,
        authors: str,
        year: int | str | None,
        item_type: str,
        text: str,
    ) -> ProviderResponse:
        """Extract using a single provider.

        Args:
            provider: Provider configuration.
            paper_id: Paper identifier.
            title: Paper title.
            authors: Author string.
            year: Publication year.
            item_type: Document type.
            text: Paper text content.

        Returns:
            ProviderResponse with extraction or error.
        """
        import time

        start = time.time()

        try:
            client = self._get_client(provider.name)
            result = client.extract(
                paper_id=paper_id,
                title=title,
                authors=authors,
                year=year,
                item_type=item_type,
                text=text,
            )

            duration = time.time() - start

            if result.success and result.extraction:
                return ProviderResponse(
                    provider=provider.name,
                    extraction=result.extraction,
                    success=True,
                    duration_seconds=duration,
                )
            else:
                return ProviderResponse(
                    provider=provider.name,
                    extraction=None,
                    success=False,
                    error=result.error or "Unknown error",
                    duration_seconds=duration,
                )

        except Exception as e:
            duration = time.time() - start
            logger.error(f"Provider {provider.name} failed: {e}")
            return ProviderResponse(
                provider=provider.name,
                extraction=None,
                success=False,
                error=str(e),
                duration_seconds=duration,
            )

    def extract(
        self,
        paper_id: str,
        title: str,
        authors: str,
        year: int | str | None,
        item_type: str,
        text: str,
    ) -> CouncilResult:
        """Extract using all providers and build consensus.

        Args:
            paper_id: Paper identifier.
            title: Paper title.
            authors: Author string.
            year: Publication year.
            item_type: Document type.
            text: Paper text content.

        Returns:
            CouncilResult with consensus extraction.
        """
        import time

        start = time.time()

        enabled_providers = [p for p in self.config.providers if p.enabled]

        if not enabled_providers:
            return CouncilResult(
                paper_id=paper_id,
                consensus=None,
                provider_responses=[],
                success=False,
                errors=["No providers enabled"],
            )

        # Execute extractions
        responses: list[ProviderResponse] = []

        if self.config.parallel and len(enabled_providers) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=len(enabled_providers)) as executor:
                futures = {
                    executor.submit(
                        self._extract_single,
                        provider,
                        paper_id,
                        title,
                        authors,
                        year,
                        item_type,
                        text,
                    ): provider
                    for provider in enabled_providers
                }

                for future in as_completed(futures, timeout=self.config.timeout):
                    try:
                        response = future.result()
                        responses.append(response)
                    except Exception as e:
                        provider = futures[future]
                        responses.append(
                            ProviderResponse(
                                provider=provider.name,
                                extraction=None,
                                success=False,
                                error=str(e),
                            )
                        )
        else:
            # Sequential execution
            for provider in enabled_providers:
                response = self._extract_single(
                    provider, paper_id, title, authors, year, item_type, text
                )
                responses.append(response)

        # Collect successful extractions
        successful = [r for r in responses if r.success and r.extraction]
        errors = [f"{r.provider}: {r.error}" for r in responses if not r.success]

        total_duration = time.time() - start
        total_cost = sum(r.cost for r in responses)

        # Check if we have enough responses
        if len(successful) < self.config.min_responses:
            if self.config.fallback_to_single and successful:
                # Use the single successful response
                consensus = successful[0].extraction
                confidence = 0.5  # Lower confidence for single response
            else:
                return CouncilResult(
                    paper_id=paper_id,
                    consensus=None,
                    provider_responses=responses,
                    success=False,
                    total_duration_seconds=total_duration,
                    total_cost=total_cost,
                    errors=errors
                    + [
                        f"Only {len(successful)} responses, need {self.config.min_responses}"
                    ],
                )
        else:
            # Build consensus
            extractions = [r.extraction for r in successful if r.extraction]
            weights = [
                next(
                    (p.weight for p in enabled_providers if p.name == r.provider), 1.0
                )
                for r in successful
            ]
            consensus = aggregate_extractions(extractions, weights)
            confidence = calculate_consensus_confidence(extractions, self.config)

        return CouncilResult(
            paper_id=paper_id,
            consensus=consensus,
            provider_responses=responses,
            success=True,
            consensus_confidence=confidence,
            total_duration_seconds=total_duration,
            total_cost=total_cost,
            errors=errors,
        )
