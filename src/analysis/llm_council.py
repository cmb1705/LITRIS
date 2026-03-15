"""LLM Council for multi-provider consensus extraction.

This module implements a council approach where multiple LLM providers
are queried in parallel and their responses are aggregated into a
consensus extraction.

The approach is inspired by karpathy/llm-council but simplified for
structured extraction tasks:

1. Fan out: Query all configured providers simultaneously
2. Collect: Gather successful responses with error handling
3. Aggregate: Build consensus using field-specific strategies
4. (Optional) Synthesize: A judge LLM reviews all outputs and produces
   a merged best answer

Reference: https://github.com/karpathy/llm-council
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.analysis.schemas import SemanticAnalysis

if TYPE_CHECKING:
    from src.analysis.base_llm import ExtractionMode

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a council provider."""

    name: str
    weight: float = 1.0  # Reliability weight for weighted consensus
    timeout: int = 120  # Per-provider timeout in seconds
    max_cost: float | None = None  # Max cost per extraction (USD)
    enabled: bool = True
    mode: ExtractionMode = "cli"  # cli default to avoid API costs
    model: str | None = None  # Provider-specific model override


@dataclass
class SynthesisConfig:
    """Configuration for synthesis round (judge LLM)."""

    enabled: bool = False
    judge_provider: str = "anthropic"
    judge_mode: ExtractionMode = "api"
    judge_model: str | None = None


@dataclass
class CouncilConfig:
    """Configuration for the LLM council."""

    providers: list[ProviderConfig] = field(default_factory=list)
    min_responses: int = 2  # Minimum responses needed for consensus
    fallback_to_single: bool = True  # Use single response if min not met
    parallel: bool = True  # Run providers in parallel
    timeout: int = 180  # Overall timeout for council
    aggregation_strategy: str = "longest"  # "longest", "quality_weighted"
    field_strategies: dict[str, str] = field(default_factory=dict)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)


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


@dataclass
class QueryResponse:
    """Response from a single provider for a generic query."""

    provider: str
    response: str | None
    success: bool
    error: str | None = None


@dataclass
class QueryResult:
    """Result of a generic council query."""

    query_id: str
    consensus_response: str | None
    provider_responses: list[QueryResponse]
    success: bool


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def _compute_quality_score(text: str) -> float:
    """Score text quality for aggregation weighting.

    Heuristic scoring based on:
    - Sentence count (2-5 ideal): up to +0.3
    - Named entities / numbers: +0.05 each, capped at +0.2
    - Citation patterns ``(Author, Year)``: +0.1 each, capped at +0.3
    - Short penalty (<50 chars): 0.5x multiplier

    Returns:
        Score between 0 and 1.
    """
    if not text:
        return 0.0

    score = 0.0

    # Sentence count (split on sentence-ending punctuation)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    n_sentences = len(sentences)
    if 2 <= n_sentences <= 5:
        score += 0.3
    elif n_sentences == 1:
        score += 0.15
    elif n_sentences > 5:
        score += 0.2  # Slightly less ideal but still rich

    # Named entities / numbers (simple heuristic: capitalized multi-word or numbers)
    numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
    score += min(len(numbers) * 0.05, 0.2)

    # Citation patterns: (Author, Year) or (Author Year) or (Author et al., Year)
    citations = re.findall(r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\)', text)
    score += min(len(citations) * 0.1, 0.3)

    # Short text penalty
    if len(text) < 50:
        score *= 0.5

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Aggregation strategy functions
# ---------------------------------------------------------------------------

# Type for aggregation strategy: takes list of (value, weight) pairs
AggregationFn = Callable[[list[tuple[str | None, float]]], str | None]


def strategy_longest(pairs: list[tuple[str | None, float]]) -> str | None:
    """Select the longest non-None string. Ignores weights."""
    valid = [v for v, _w in pairs if v]
    if not valid:
        return None
    return max(valid, key=len)


def strategy_quality_weighted(pairs: list[tuple[str | None, float]]) -> str | None:
    """Select the string with highest (quality_score * weight).

    Breaks ties with string length.
    """
    scored: list[tuple[str, float]] = []
    for value, weight in pairs:
        if not value:
            continue
        quality = _compute_quality_score(value)
        scored.append((value, quality * weight))
    if not scored:
        return None
    # Sort by composite score descending, then length descending as tiebreaker
    scored.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    return scored[0][0]


def strategy_union_merge(pairs: list[tuple[str | None, float]]) -> str | None:
    """Merge unique sentences across providers, deduplicating by word overlap.

    Splits each provider's text by sentence boundaries. Deduplicates by
    60% word-overlap threshold (normalized lowercase). Higher-weight
    provider's phrasing wins ties.
    """
    # Collect (sentence, weight) from all providers, sorted by weight desc
    weighted_pairs = sorted(
        [(v, w) for v, w in pairs if v],
        key=lambda x: x[1],
        reverse=True,
    )
    if not weighted_pairs:
        return None

    # Extract all sentences with their provider weight
    all_sentences: list[tuple[str, float]] = []
    for text, weight in weighted_pairs:
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        for s in sents:
            all_sentences.append((s, weight))

    # Deduplicate by word overlap
    unique: list[str] = []
    unique_word_sets: list[set[str]] = []

    for sent, _weight in all_sentences:
        sent_words = set(sent.lower().split())
        if not sent_words:
            continue
        is_duplicate = False
        for existing_words in unique_word_sets:
            if not existing_words:
                continue
            overlap = len(sent_words & existing_words) / min(len(sent_words), len(existing_words))
            if overlap >= 0.6:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(sent)
            unique_word_sets.append(sent_words)

    return " ".join(unique) if unique else None


STRATEGY_REGISTRY: dict[str, AggregationFn] = {
    "longest": strategy_longest,
    "quality_weighted": strategy_quality_weighted,
    "union": strategy_union_merge,
}


def aggregate_analyses(
    analyses: list[SemanticAnalysis],
    weights: list[float] | None = None,
    default_strategy: str = "longest",
    field_strategies: dict[str, str] | None = None,
) -> SemanticAnalysis:
    """Aggregate multiple SemanticAnalysis results into consensus.

    Args:
        analyses: List of analyses from different providers.
        weights: Optional provider reliability weights.
        default_strategy: Default aggregation strategy for all fields.
        field_strategies: Per-field strategy overrides,
            e.g. ``{"q03_key_claims": "union"}``.

    Returns:
        Consensus SemanticAnalysis.
    """
    if not analyses:
        raise ValueError("Cannot aggregate empty list of analyses")

    if len(analyses) == 1:
        return analyses[0]

    effective_weights = weights or [1.0] * len(analyses)
    field_strategies = field_strategies or {}

    # Warn and adjust if weights length doesn't match analyses
    if len(effective_weights) != len(analyses):
        logger.warning(
            "Weights length (%d) does not match analyses length (%d); "
            "padding with 1.0 or truncating",
            len(effective_weights), len(analyses),
        )
        if len(effective_weights) < len(analyses):
            effective_weights = list(effective_weights) + [1.0] * (
                len(analyses) - len(effective_weights)
            )
        else:
            effective_weights = list(effective_weights)[: len(analyses)]

    # Use first analysis for metadata fields
    base = analyses[0]

    q_values: dict[str, str | None] = {}
    for field_name in SemanticAnalysis.DIMENSION_FIELDS:
        strategy_name = field_strategies.get(field_name, default_strategy)
        strategy_fn = STRATEGY_REGISTRY.get(strategy_name)
        if strategy_fn is None:
            logger.warning(
                "Unknown aggregation strategy '%s' for field '%s'; "
                "falling back to 'longest'",
                strategy_name, field_name,
            )
            strategy_fn = strategy_longest
        pairs = [
            (getattr(a, field_name, None), w)
            for a, w in zip(analyses, effective_weights, strict=False)
        ]
        q_values[field_name] = strategy_fn(pairs)

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
    - Providers agree on key fields (research_core q01-q05)
    - Coverage spread is consistent across providers

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

    # Agreement on research_core fields (q01-q05) via word overlap
    core_fields = SemanticAnalysis.DIMENSION_GROUPS.get(
        "research_core", SemanticAnalysis.CORE_FIELDS
    )
    agreement_scores: list[float] = []

    for field_name in core_fields:
        values = [getattr(a, field_name, None) for a in analyses]
        non_none = [v for v in values if v]
        if len(non_none) < 2:
            continue
        # Pairwise word overlap
        token_sets = [set(v.lower().split()) for v in non_none]
        pair_overlaps: list[float] = []
        for idx_i in range(len(token_sets)):
            for idx_j in range(idx_i + 1, len(token_sets)):
                union = token_sets[idx_i] | token_sets[idx_j]
                intersection = token_sets[idx_i] & token_sets[idx_j]
                if union:
                    pair_overlaps.append(len(intersection) / len(union))
        if pair_overlaps:
            agreement_scores.append(sum(pair_overlaps) / len(pair_overlaps))

    # Coverage spread: how consistent are providers on which fields are filled
    if len(analyses) >= 2:
        fill_counts = [
            sum(1 for f in SemanticAnalysis.DIMENSION_FIELDS if getattr(a, f, None))
            for a in analyses
        ]
        max_fill = max(fill_counts)
        min_fill = min(fill_counts)
        spread = 1 - (max_fill - min_fill) / 40 if max_fill > 0 else 0.5
        agreement_scores.append(spread)

    avg_agreement = (
        sum(agreement_scores) / len(agreement_scores)
        if agreement_scores
        else 0.5
    )

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

    def _get_client(self, provider: ProviderConfig | str) -> Any:
        """Get or create LLM client for provider.

        Args:
            provider: ProviderConfig instance or provider name string
                (legacy callers).
        """
        if isinstance(provider, str):
            # Legacy path: bare name, use defaults
            cache_key = provider
            name = provider
            mode: ExtractionMode = "cli"
            model = None
        else:
            cache_key = f"{provider.name}:{provider.mode}:{provider.model}"
            name = provider.name
            mode = provider.mode
            model = provider.model

        if cache_key not in self._clients:
            from src.analysis.llm_factory import create_llm_client

            timeout = provider.timeout if isinstance(provider, ProviderConfig) else 120
            self._clients[cache_key] = create_llm_client(
                provider=name,  # type: ignore[arg-type]
                mode=mode,
                model=model,
                timeout=timeout,
            )
        return self._clients[cache_key]

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
        """Extract using a single provider via 6-pass pipeline.

        Runs the same 6-pass prompt sequence used by SectionExtractor
        to produce SemanticAnalysis with q-field dimensions.

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
        from datetime import datetime

        from src.analysis.semantic_prompts import (
            PASS_DEFINITIONS,
            build_pass_user_prompt,
        )

        NUM_PASSES = len(PASS_DEFINITIONS)

        start = time.time()

        try:
            client = self._get_client(provider)

            # Run 6-pass extraction (same as SectionExtractor._extract_6_pass)
            all_answers: dict[str, str | None] = {}
            total_cost = 0.0
            errors: list[str] = []

            for pass_num in range(1, NUM_PASSES + 1):
                pass_label, pass_questions = PASS_DEFINITIONS[pass_num - 1]
                pass_fields = [q[0] for q in pass_questions]

                # Build pass-specific prompt with q-field instructions
                prompt = build_pass_user_prompt(
                    pass_number=pass_num,
                    title=title,
                    authors=authors,
                    year=year,
                    document_type=item_type,
                    text=text,
                )

                result = client.extract(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    year=year,
                    item_type=item_type,
                    text=text,
                    prompt_override=prompt,
                )

                total_cost += getattr(result, "cost", 0.0) or 0.0

                if not result.success or not result.extraction:
                    errors.append(
                        f"pass {pass_num} ({pass_label}): "
                        f"{result.error or 'Unknown error'}"
                    )
                    continue

                # Extract q-field answers from this pass
                for field in pass_fields:
                    value = getattr(result.extraction, field, None)
                    if value is not None:
                        all_answers[field] = value

            duration = time.time() - start

            # Enforce per-provider timeout (log only, don't fail)
            if duration > provider.timeout:
                logger.warning(
                    f"Provider {provider.name} exceeded timeout "
                    f"({duration:.1f}s > {provider.timeout}s)"
                )

            # If all passes failed, return error
            if len(errors) == NUM_PASSES:
                return ProviderResponse(
                    provider=provider.name,
                    extraction=None,
                    success=False,
                    error=f"All {NUM_PASSES} passes failed: " + "; ".join(errors),
                    duration_seconds=duration,
                    cost=total_cost,
                )

            # Check cost limit
            if provider.max_cost is not None and total_cost > provider.max_cost:
                logger.warning(
                    f"Provider {provider.name} exceeded cost limit "
                    f"(${total_cost:.4f} > ${provider.max_cost:.4f})"
                )
                return ProviderResponse(
                    provider=provider.name,
                    extraction=None,
                    success=False,
                    error=f"Cost ${total_cost:.4f} exceeded limit ${provider.max_cost:.4f}",
                    duration_seconds=duration,
                    cost=total_cost,
                )

            # Build SemanticAnalysis from merged pass answers
            model_name = getattr(client, "model", None)
            if not isinstance(model_name, str):
                model_name = provider.name
            extraction = SemanticAnalysis(
                paper_id=paper_id,
                prompt_version="2.0.0",
                extraction_model=model_name,
                extracted_at=datetime.now().isoformat(),
                **all_answers,
            )

            return ProviderResponse(
                provider=provider.name,
                extraction=extraction,
                success=True,
                duration_seconds=duration,
                cost=total_cost,
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
            # Build consensus via mechanical aggregation
            analyses = [r.extraction for r in successful if r.extraction]
            provider_weights = [
                next(
                    (p.weight for p in enabled_providers if p.name == r.provider), 1.0
                )
                for r in successful
            ]
            consensus = aggregate_analyses(
                analyses,
                weights=provider_weights,
                default_strategy=self.config.aggregation_strategy,
                field_strategies=self.config.field_strategies or None,
            )
            confidence = calculate_consensus_confidence(analyses, self.config)

            # Optional synthesis round
            if (
                self.config.synthesis.enabled
                and len(successful) >= 2
            ):
                synthesis_result = self._run_synthesis_round(successful)
                if synthesis_result is not None:
                    consensus = synthesis_result
                    confidence = min(confidence + 0.1, 1.0)

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

    # ------------------------------------------------------------------
    # Synthesis round
    # ------------------------------------------------------------------

    def _build_synthesis_prompt(
        self, responses: list[ProviderResponse],
    ) -> str:
        """Build a prompt for the synthesis judge.

        Serializes each provider's extraction as labeled JSON sections so the
        judge can compare and merge.
        """
        import json

        sections: list[str] = []
        for resp in responses:
            if not resp.extraction:
                continue
            dims = resp.extraction.non_none_dimensions()
            sections.append(
                f"## Provider: {resp.provider}\n```json\n"
                f"{json.dumps(dims, indent=2, ensure_ascii=False)}\n```"
            )

        provider_block = "\n\n".join(sections)

        return (
            "You are a synthesis judge for academic paper extraction. "
            "Multiple LLM providers have independently extracted structured "
            "information from the same paper. Your task:\n\n"
            "1. Identify agreements across providers and keep those as-is.\n"
            "2. Resolve disagreements by comparing evidence quality, specificity, "
            "and accuracy.\n"
            "3. Include unique insights from any provider that add value.\n"
            "4. Return a single merged extraction as JSON with the same q-field "
            "keys (q01_research_question through q40_policy_recommendations).\n\n"
            "Provider extractions:\n\n"
            f"{provider_block}\n\n"
            "Return ONLY the merged JSON object with q-field keys and string values."
        )

    def _run_synthesis_round(
        self, responses: list[ProviderResponse],
    ) -> SemanticAnalysis | None:
        """Run synthesis round with a judge LLM.

        Args:
            responses: Successful provider responses to synthesize.

        Returns:
            Synthesized SemanticAnalysis, or None if synthesis fails.
        """
        synth = self.config.synthesis
        prompt = self._build_synthesis_prompt(responses)

        try:
            judge_provider = ProviderConfig(
                name=synth.judge_provider,
                mode=synth.judge_mode,
                model=synth.judge_model,
            )
            client = self._get_client(judge_provider)

            # Use the first successful extraction for metadata
            base = next(
                (r.extraction for r in responses if r.extraction), None,
            )
            if base is None:
                logger.warning("Synthesis skipped: no valid extractions to merge")
                return None

            result = client.extract(
                paper_id=base.paper_id,
                title="",
                authors="",
                year=None,
                item_type="",
                text="",
                prompt_override=prompt,
            )

            if result.success and result.extraction:
                # Preserve metadata from original extraction
                result.extraction.paper_id = base.paper_id
                result.extraction.prompt_version = base.prompt_version
                result.extraction.extraction_model = f"synthesis:{synth.judge_provider}"
                result.extraction.extracted_at = base.extracted_at
                return result.extraction

            logger.warning("Synthesis round failed: %s", result.error)
            return None

        except Exception as e:
            logger.error("Synthesis round error: %s", e)
            return None

    # ------------------------------------------------------------------
    # Generic query (non-extraction)
    # ------------------------------------------------------------------

    def query(
        self,
        prompt: str,
        query_id: str = "",
    ) -> QueryResult:
        """Fan out a generic prompt to all providers and collect responses.

        Unlike ``extract()``, this does NOT parse into SemanticAnalysis.
        Optionally runs a synthesis round if configured.

        Args:
            prompt: The prompt to send to all providers.
            query_id: Identifier for this query.

        Returns:
            QueryResult with per-provider responses and optional consensus.
        """
        enabled_providers = [p for p in self.config.providers if p.enabled]
        if not enabled_providers:
            return QueryResult(
                query_id=query_id,
                consensus_response=None,
                provider_responses=[],
                success=False,
            )

        query_responses: list[QueryResponse] = []

        for provider in enabled_providers:
            try:
                client = self._get_client(provider)
                result = client.raw_query(prompt)
                query_responses.append(QueryResponse(
                    provider=provider.name,
                    response=result[0],
                    success=result[1],
                    error=result[2],
                ))
            except Exception as exc:
                query_responses.append(QueryResponse(
                    provider=provider.name,
                    response=None,
                    success=False,
                    error=str(exc),
                ))

        successful = [r for r in query_responses if r.success and r.response]

        # Simple consensus: longest response (or synthesis if enabled)
        consensus = None
        if successful:
            consensus = max(
                (r.response for r in successful if r.response),
                key=len,
                default=None,
            )

        return QueryResult(
            query_id=query_id,
            consensus_response=consensus,
            provider_responses=query_responses,
            success=bool(successful),
        )
