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


class TestProviderCostAndTimeout:
    """Tests for cost and timeout controls."""

    def test_cost_limit_rejects_expensive_response(self):
        """Provider response exceeding max_cost is rejected."""
        from unittest.mock import MagicMock, patch

        config = CouncilConfig(
            providers=[ProviderConfig(name="expensive", max_cost=0.05)],
            min_responses=1,
            fallback_to_single=False,
        )
        council = LLMCouncil(config)

        # Mock client that returns a result with cost > max_cost
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.extraction = PaperExtraction(thesis_statement="Test")
        mock_result.cost = 0.10  # Exceeds max_cost of 0.05
        mock_client.extract.return_value = mock_result

        with patch.object(council, "_get_client", return_value=mock_client):
            result = council.extract(
                paper_id="test",
                title="Test",
                authors="Author",
                year=2024,
                item_type="article",
                text="Content",
            )

        assert not result.success
        assert any("exceeded limit" in e for e in result.errors)

    def test_cost_within_limit_accepted(self):
        """Provider response within max_cost is accepted."""
        from unittest.mock import MagicMock, patch

        config = CouncilConfig(
            providers=[ProviderConfig(name="cheap", max_cost=0.10)],
            min_responses=1,
            fallback_to_single=True,
        )
        council = LLMCouncil(config)

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.extraction = PaperExtraction(thesis_statement="Test")
        mock_result.cost = 0.05  # Within max_cost
        mock_client.extract.return_value = mock_result

        with patch.object(council, "_get_client", return_value=mock_client):
            result = council.extract(
                paper_id="test",
                title="Test",
                authors="Author",
                year=2024,
                item_type="article",
                text="Content",
            )

        assert result.success
        assert result.provider_responses[0].cost == 0.05

    def test_no_cost_limit_allows_any_cost(self):
        """Provider with no max_cost accepts any cost."""
        from unittest.mock import MagicMock, patch

        config = CouncilConfig(
            providers=[ProviderConfig(name="unlimited", max_cost=None)],
            min_responses=1,
            fallback_to_single=True,
        )
        council = LLMCouncil(config)

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.extraction = PaperExtraction(thesis_statement="Test")
        mock_result.cost = 99.99
        mock_client.extract.return_value = mock_result

        with patch.object(council, "_get_client", return_value=mock_client):
            result = council.extract(
                paper_id="test",
                title="Test",
                authors="Author",
                year=2024,
                item_type="article",
                text="Content",
            )

        assert result.success


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


class TestLLMCouncilIntegration:
    """End-to-end integration tests with mocked multi-provider extraction."""

    def _make_mock_client(self, extraction):
        """Create a mock LLM client returning the given extraction."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.extraction = extraction
        mock_result.cost = 0.01
        mock_client.extract.return_value = mock_result
        return mock_client

    def _make_failing_client(self, error_msg="Provider error"):
        """Create a mock LLM client that raises an exception."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.extract.side_effect = RuntimeError(error_msg)
        return mock_client

    def test_two_provider_consensus(self):
        """Two providers produce a consensus extraction."""
        from unittest.mock import patch

        config = CouncilConfig(
            providers=[
                ProviderConfig(name="anthropic", weight=1.2),
                ProviderConfig(name="openai", weight=1.0),
            ],
            min_responses=2,
            parallel=False,  # Sequential for deterministic test
        )
        council = LLMCouncil(config)

        extraction_a = PaperExtraction(
            thesis_statement="Short thesis",
            keywords=["ml", "graphs"],
            extraction_confidence=0.9,
            methodology=Methodology(approach="quantitative"),
        )
        extraction_b = PaperExtraction(
            thesis_statement="A more detailed and longer thesis statement for testing",
            keywords=["ml", "networks", "citation"],
            extraction_confidence=0.7,
            methodology=Methodology(approach="quantitative", data_sources=["papers"]),
        )

        clients = {
            "anthropic": self._make_mock_client(extraction_a),
            "openai": self._make_mock_client(extraction_b),
        }

        with patch.object(council, "_get_client", side_effect=lambda name: clients[name]):
            result = council.extract(
                paper_id="test_paper",
                title="Test Paper",
                authors="Author et al.",
                year=2024,
                item_type="journalArticle",
                text="Full paper text...",
            )

        assert result.success
        assert result.consensus is not None
        # LONGEST strategy: longer thesis wins
        assert "more detailed" in result.consensus.thesis_statement
        # UNION strategy: keywords merged
        assert set(result.consensus.keywords) == {"ml", "graphs", "networks", "citation"}
        # MAJORITY_VOTE: both say quantitative
        assert result.consensus.methodology.approach == "quantitative"
        # UNION: data_sources merged
        assert "papers" in result.consensus.methodology.data_sources
        # AVERAGE: confidence averaged with weights
        assert 0.5 < result.consensus.extraction_confidence < 1.0
        # Both providers responded
        assert len(result.provider_responses) == 2
        assert all(r.success for r in result.provider_responses)
        # Confidence should be high with 2/2 responses
        assert result.consensus_confidence > 0.5
        # Cost tracked
        assert result.total_cost > 0

    def test_three_provider_with_one_failure(self):
        """Three providers, one fails, consensus still built from two."""
        from unittest.mock import patch

        config = CouncilConfig(
            providers=[
                ProviderConfig(name="anthropic", weight=1.0),
                ProviderConfig(name="openai", weight=1.0),
                ProviderConfig(name="google", weight=0.8),
            ],
            min_responses=2,
            parallel=False,
        )
        council = LLMCouncil(config)

        extraction_a = PaperExtraction(
            thesis_statement="Thesis from anthropic",
            keywords=["a", "b"],
        )
        extraction_b = PaperExtraction(
            thesis_statement="Thesis from openai provider is longer",
            keywords=["b", "c"],
        )

        clients = {
            "anthropic": self._make_mock_client(extraction_a),
            "openai": self._make_mock_client(extraction_b),
            "google": self._make_failing_client("API quota exceeded"),
        }

        with patch.object(council, "_get_client", side_effect=lambda name: clients[name]):
            result = council.extract(
                paper_id="test",
                title="Test",
                authors="Author",
                year=2024,
                item_type="article",
                text="Content",
            )

        assert result.success
        assert result.consensus is not None
        # 2 of 3 succeeded
        successful = [r for r in result.provider_responses if r.success]
        assert len(successful) == 2
        # Error recorded for google
        assert any("google" in e for e in result.errors)
        # Keywords unioned from both successful providers
        assert set(result.consensus.keywords) == {"a", "b", "c"}

    def test_fallback_to_single_when_below_min(self):
        """Falls back to single response when below min_responses."""
        from unittest.mock import patch

        config = CouncilConfig(
            providers=[
                ProviderConfig(name="anthropic"),
                ProviderConfig(name="openai"),
            ],
            min_responses=2,
            fallback_to_single=True,
            parallel=False,
        )
        council = LLMCouncil(config)

        extraction = PaperExtraction(thesis_statement="Only thesis")

        clients = {
            "anthropic": self._make_mock_client(extraction),
            "openai": self._make_failing_client(),
        }

        with patch.object(council, "_get_client", side_effect=lambda name: clients[name]):
            result = council.extract(
                paper_id="test",
                title="Test",
                authors="Author",
                year=2024,
                item_type="article",
                text="Content",
            )

        assert result.success
        assert result.consensus is not None
        assert result.consensus.thesis_statement == "Only thesis"
        # Lower confidence for single-response fallback
        assert result.consensus_confidence == 0.5

    def test_all_providers_fail_returns_error(self):
        """All providers failing returns error when fallback disabled."""
        from unittest.mock import patch

        config = CouncilConfig(
            providers=[
                ProviderConfig(name="anthropic"),
                ProviderConfig(name="openai"),
            ],
            min_responses=2,
            fallback_to_single=False,
            parallel=False,
        )
        council = LLMCouncil(config)

        clients = {
            "anthropic": self._make_failing_client("Error A"),
            "openai": self._make_failing_client("Error B"),
        }

        with patch.object(council, "_get_client", side_effect=lambda name: clients[name]):
            result = council.extract(
                paper_id="test",
                title="Test",
                authors="Author",
                year=2024,
                item_type="article",
                text="Content",
            )

        assert not result.success
        assert result.consensus is None
        assert len(result.errors) >= 2
