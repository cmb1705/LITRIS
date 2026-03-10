"""Tests for LLM council consensus extraction."""

import pytest

from src.analysis.llm_council import (
    CouncilConfig,
    CouncilResult,
    LLMCouncil,
    ProviderConfig,
    ProviderResponse,
    aggregate_extractions,
    calculate_consensus_confidence,
)
from src.analysis.schemas import (
    KeyFinding,
    Methodology,
    PaperExtraction,
)


class TestAggregateExtractions:
    """Tests for extraction aggregation."""

    def test_empty_list_returns_default(self):
        """Empty list returns default extraction."""
        result = aggregate_extractions([])
        assert result.thesis_statement is None
        assert result.keywords == []

    def test_single_extraction_returned_unchanged(self):
        """Single extraction is returned as-is."""
        extraction = PaperExtraction(
            thesis_statement="Test thesis",
            keywords=["keyword1", "keyword2"],
        )
        result = aggregate_extractions([extraction])
        assert result.thesis_statement == "Test thesis"
        assert result.keywords == ["keyword1", "keyword2"]

    def test_longest_string_wins(self):
        """Longest string is selected for string fields."""
        extractions = [
            PaperExtraction(thesis_statement="Short"),
            PaperExtraction(thesis_statement="This is a much longer thesis statement"),
        ]
        result = aggregate_extractions(extractions)
        assert result.thesis_statement == "This is a much longer thesis statement"

    def test_list_union(self):
        """Lists are unioned with deduplication."""
        extractions = [
            PaperExtraction(keywords=["a", "b", "c"]),
            PaperExtraction(keywords=["b", "c", "d", "e"]),
        ]
        result = aggregate_extractions(extractions)
        assert set(result.keywords) == {"a", "b", "c", "d", "e"}

    def test_methodology_merge(self):
        """Methodology objects are merged."""
        extractions = [
            PaperExtraction(
                methodology=Methodology(
                    approach="quantitative",
                    data_sources=["surveys"],
                )
            ),
            PaperExtraction(
                methodology=Methodology(
                    approach="quantitative",
                    data_sources=["interviews"],
                    analysis_methods=["regression"],
                )
            ),
        ]
        result = aggregate_extractions(extractions)
        assert result.methodology.approach == "quantitative"
        assert set(result.methodology.data_sources) == {"surveys", "interviews"}
        assert result.methodology.analysis_methods == ["regression"]

    def test_key_findings_deduplicated(self):
        """Key findings are deduplicated by text."""
        extractions = [
            PaperExtraction(
                key_findings=[
                    KeyFinding(finding="Finding A"),
                    KeyFinding(finding="Finding B"),
                ]
            ),
            PaperExtraction(
                key_findings=[
                    KeyFinding(finding="Finding B"),  # Duplicate
                    KeyFinding(finding="Finding C"),
                ]
            ),
        ]
        result = aggregate_extractions(extractions)
        finding_texts = [f.finding for f in result.key_findings]
        assert len(finding_texts) == 3
        assert "Finding A" in finding_texts
        assert "Finding B" in finding_texts
        assert "Finding C" in finding_texts

    def test_confidence_averaged(self):
        """Extraction confidence is averaged."""
        extractions = [
            PaperExtraction(extraction_confidence=0.8),
            PaperExtraction(extraction_confidence=0.6),
        ]
        result = aggregate_extractions(extractions)
        assert result.extraction_confidence == pytest.approx(0.7, rel=0.01)

    def test_weighted_confidence(self):
        """Weights affect confidence averaging."""
        extractions = [
            PaperExtraction(extraction_confidence=0.8),
            PaperExtraction(extraction_confidence=0.4),
        ]
        # Higher weight on first extraction
        result = aggregate_extractions(extractions, weights=[2.0, 1.0])
        # Weighted average: (0.8*2 + 0.4*1) / 3 = 2.0/3 = 0.667
        expected = (0.8 * 2.0 + 0.4 * 1.0) / 3.0
        assert result.extraction_confidence == pytest.approx(expected, rel=0.01)


class TestCalculateConsensusConfidence:
    """Tests for consensus confidence calculation."""

    def test_empty_extractions_zero_confidence(self):
        """Empty extractions give zero confidence."""
        config = CouncilConfig(
            providers=[ProviderConfig(name="test1"), ProviderConfig(name="test2")]
        )
        confidence = calculate_consensus_confidence([], config)
        assert confidence == 0.0

    def test_full_response_high_confidence(self):
        """All providers responding gives higher confidence."""
        config = CouncilConfig(
            providers=[ProviderConfig(name="test1"), ProviderConfig(name="test2")]
        )
        extractions = [
            PaperExtraction(thesis_statement="Same thesis", keywords=["a", "b"]),
            PaperExtraction(thesis_statement="Same thesis", keywords=["a", "b"]),
        ]
        confidence = calculate_consensus_confidence(extractions, config)
        assert confidence > 0.5

    def test_partial_response_lower_confidence(self):
        """Fewer responses gives lower confidence."""
        config = CouncilConfig(
            providers=[
                ProviderConfig(name="test1"),
                ProviderConfig(name="test2"),
                ProviderConfig(name="test3"),
            ]
        )
        extractions = [
            PaperExtraction(keywords=["a", "b"]),
        ]
        confidence = calculate_consensus_confidence(extractions, config)
        # Only 1 of 3 providers responded
        assert confidence < 0.5


class TestProviderConfig:
    """Tests for provider configuration."""

    def test_default_values(self):
        """Provider config has sensible defaults."""
        config = ProviderConfig(name="anthropic")
        assert config.weight == 1.0
        assert config.timeout == 120
        assert config.max_cost is None
        assert config.enabled is True

    def test_custom_values(self):
        """Custom values override defaults."""
        config = ProviderConfig(
            name="openai",
            weight=1.5,
            timeout=60,
            max_cost=0.10,
            enabled=False,
        )
        assert config.name == "openai"
        assert config.weight == 1.5
        assert config.timeout == 60
        assert config.max_cost == 0.10
        assert config.enabled is False


class TestCouncilConfig:
    """Tests for council configuration."""

    def test_default_values(self):
        """Council config has sensible defaults."""
        config = CouncilConfig()
        assert config.providers == []
        assert config.min_responses == 2
        assert config.fallback_to_single is True
        assert config.parallel is True
        assert config.timeout == 180
        assert config.consensus_threshold == 0.5

    def test_with_providers(self):
        """Can configure multiple providers."""
        config = CouncilConfig(
            providers=[
                ProviderConfig(name="anthropic", weight=1.2),
                ProviderConfig(name="openai", weight=1.0),
            ],
            min_responses=1,
        )
        assert len(config.providers) == 2
        assert config.providers[0].name == "anthropic"
        assert config.providers[0].weight == 1.2


class TestCouncilResult:
    """Tests for council result structure."""

    def test_successful_result(self):
        """Successful result has consensus."""
        result = CouncilResult(
            paper_id="test123",
            consensus=PaperExtraction(thesis_statement="Test"),
            provider_responses=[
                ProviderResponse(
                    provider="anthropic",
                    extraction=PaperExtraction(),
                    success=True,
                )
            ],
            success=True,
            consensus_confidence=0.8,
        )
        assert result.success
        assert result.consensus is not None
        assert result.consensus_confidence == 0.8

    def test_failed_result(self):
        """Failed result has errors."""
        result = CouncilResult(
            paper_id="test123",
            consensus=None,
            provider_responses=[],
            success=False,
            errors=["No providers available"],
        )
        assert not result.success
        assert result.consensus is None
        assert "No providers available" in result.errors


class TestLLMCouncilExtract:
    """Tests for council extraction with mocked providers."""

    def test_no_providers_returns_error(self):
        """No enabled providers returns error."""
        config = CouncilConfig(providers=[])
        council = LLMCouncil(config)

        result = council.extract(
            paper_id="test",
            title="Test Paper",
            authors="Author",
            year=2024,
            item_type="article",
            text="Test content",
        )

        assert not result.success
        assert "No providers enabled" in result.errors

    def test_disabled_providers_skipped(self):
        """Disabled providers are skipped."""
        config = CouncilConfig(
            providers=[
                ProviderConfig(name="test1", enabled=False),
                ProviderConfig(name="test2", enabled=False),
            ]
        )
        council = LLMCouncil(config)

        result = council.extract(
            paper_id="test",
            title="Test Paper",
            authors="Author",
            year=2024,
            item_type="article",
            text="Test content",
        )

        assert not result.success
        assert "No providers enabled" in result.errors
