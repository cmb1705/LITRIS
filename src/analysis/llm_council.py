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
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.analysis.schemas import SemanticAnalysis

logger = logging.getLogger(__name__)


class ConsensusStrategy(str, Enum):
    """Strategy for reaching consensus on a field."""

    LONGEST = "longest"  # Take the longest/most detailed response


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


@dataclass
class ProviderResponse:
    """Response from a single provider."""

    provider: str
    extraction: SemanticAnalysis | None
    success: bool
    error: str | None = None
    duration_seconds: float = 0.0
    cost: float = 0.0


@dataclass
class CouncilResult:
    """Result of council consensus extraction."""

    paper_id: str
    consensus: SemanticAnalysis | None
    provider_responses: list[ProviderResponse]
    success: bool
    consensus_confidence: float = 0.0
    total_duration_seconds: float = 0.0
    total_cost: float = 0.0
    errors: list[str] = field(default_factory=list)


def _longest_string(values: list[str | None]) -> str | None:
    """Return the longest non-None string."""
    valid = [v for v in values if v]
    if not valid:
        return None
    return max(valid, key=len)


# All q-fields use LONGEST strategy since they are all str|None
_Q_FIELD_NAMES = [
    f"q{i:02d}_{name}" for i, name in [
        (1, "research_question"), (2, "thesis"), (3, "key_claims"), (4, "evidence"),
        (5, "limitations"), (6, "paradigm"), (7, "methods"), (8, "data"),
        (9, "reproducibility"), (10, "framework"), (11, "traditions"),
        (12, "key_citations"), (13, "assumptions"), (14, "counterarguments"),
        (15, "novelty"), (16, "stance"), (17, "field"), (18, "audience"),
        (19, "implications"), (20, "future_work"), (21, "quality"),
        (22, "contribution"), (23, "source_type"), (24, "other"),
        (25, "institutional_context"), (26, "historical_timing"),
        (27, "paradigm_influence"), (28, "disciplines_bridged"),
        (29, "cross_domain_insights"), (30, "cultural_scope"),
        (31, "philosophical_assumptions"), (32, "deployment_gap"),
        (33, "infrastructure_contribution"), (34, "power_dynamics"),
        (35, "gaps_and_omissions"), (36, "dual_use_concerns"),
        (37, "emergence_claims"), (38, "remaining_other"),
        (39, "network_properties"), (40, "policy_recommendations"),
    ]
]


def aggregate_analyses(
    analyses: list[SemanticAnalysis],
    weights: list[float] | None = None,
) -> SemanticAnalysis:
    """Aggregate multiple SemanticAnalysis results into consensus.

    Uses LONGEST strategy for all q-fields since they are all str|None.

    Args:
        analyses: List of analyses from different providers.
        weights: Optional weights (unused, kept for API compatibility).

    Returns:
        Consensus SemanticAnalysis.
    """
    if not analyses:
        raise ValueError("Cannot aggregate empty list of analyses")

    if len(analyses) == 1:
        return analyses[0]

    # Use first analysis for metadata fields
    base = analyses[0]

    # Build consensus: LONGEST for every q-field
    q_values: dict[str, str | None] = {}
    for field_name in _Q_FIELD_NAMES:
        values = [getattr(a, field_name, None) for a in analyses]
        q_values[field_name] = _longest_string(values)

    return SemanticAnalysis(
        paper_id=base.paper_id,
        prompt_version=base.prompt_version,
        extraction_model=base.extraction_model,
        extracted_at=base.extracted_at,
        **q_values,
    )


def calculate_consensus_confidence(
    analyses: list[SemanticAnalysis],
    config: CouncilConfig,
) -> float:
    """Calculate confidence score for the consensus.

    Higher confidence when:
    - More providers responded successfully
    - Providers agree on key fields

    Args:
        analyses: List of successful analyses.
        config: Council configuration.

    Returns:
        Confidence score between 0 and 1.
    """
    if not analyses:
        return 0.0

    n_providers = len([p for p in config.providers if p.enabled])
    n_responses = len(analyses)

    # Base confidence from response rate
    response_rate = n_responses / n_providers if n_providers > 0 else 0

    # Agreement bonus for key fields
    agreement_scores = []

    # Check thesis agreement
    theses = [a.q02_thesis for a in analyses if a.q02_thesis]
    if len(theses) > 1:
        # Simple check: do they start similarly?
        first_words = [t.split()[:5] for t in theses]
        agreement = len({tuple(w) for w in first_words}) / len(first_words)
        agreement_scores.append(1 - agreement)  # Higher when more agreement

    # Check field/discipline agreement
    fields = [a.q17_field for a in analyses if a.q17_field]
    if len(fields) > 1:
        # Token overlap across field descriptions
        field_token_sets = [set(f.lower().split()) for f in fields]
        intersection = set.intersection(*field_token_sets)
        union = set.union(*field_token_sets)
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
        """Extract using a single provider with timeout enforcement.

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

            # Enforce per-provider timeout
            if duration > provider.timeout:
                logger.warning(
                    f"Provider {provider.name} exceeded timeout "
                    f"({duration:.1f}s > {provider.timeout}s)"
                )

            if result.success and result.extraction:
                cost = getattr(result, "cost", 0.0) or 0.0
                # Check cost limit
                if provider.max_cost is not None and cost > provider.max_cost:
                    logger.warning(
                        f"Provider {provider.name} exceeded cost limit "
                        f"(${cost:.4f} > ${provider.max_cost:.4f})"
                    )
                    return ProviderResponse(
                        provider=provider.name,
                        extraction=None,
                        success=False,
                        error=f"Cost ${cost:.4f} exceeded limit ${provider.max_cost:.4f}",
                        duration_seconds=duration,
                        cost=cost,
                    )
                return ProviderResponse(
                    provider=provider.name,
                    extraction=result.extraction,
                    success=True,
                    duration_seconds=duration,
                    cost=cost,
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
            # Parallel execution with per-provider timeout enforcement
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

                try:
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
                except TimeoutError:
                    # Overall council timeout exceeded - record remaining as timed out
                    for future, provider in futures.items():
                        if not future.done():
                            future.cancel()
                            logger.warning(
                                f"Provider {provider.name} timed out "
                                f"(council timeout {self.config.timeout}s)"
                            )
                            responses.append(
                                ProviderResponse(
                                    provider=provider.name,
                                    extraction=None,
                                    success=False,
                                    error=f"Council timeout ({self.config.timeout}s) exceeded",
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
            analyses = [r.extraction for r in successful if r.extraction]
            weights = [
                next(
                    (p.weight for p in enabled_providers if p.name == r.provider), 1.0
                )
                for r in successful
            ]
            consensus = aggregate_analyses(analyses, weights)
            confidence = calculate_consensus_confidence(analyses, self.config)

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
