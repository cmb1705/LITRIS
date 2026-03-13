"""Tests for LLM council consensus extraction."""

import pytest

from src.analysis.llm_council import (
    CouncilConfig,
    CouncilResult,
    LLMCouncil,
    ProviderConfig,
    ProviderResponse,
    aggregate_analyses,
    calculate_consensus_confidence,
)
from src.analysis.schemas import SemanticAnalysis


def _make_analysis(**overrides) -> SemanticAnalysis:
    """Helper to create a SemanticAnalysis with required fields."""
    defaults = {
        "paper_id": "test_id",
        "prompt_version": "2.0.0",
        "extraction_model": "test-model",
        "extracted_at": "2026-01-01T00:00:00Z",
    }
    defaults.update(overrides)
    return SemanticAnalysis(**defaults)


class TestAggregateAnalyses:
    """Tests for analysis aggregation."""

    def test_empty_list_raises(self):
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot aggregate empty list"):
            aggregate_analyses([])

    def test_single_analysis_returned_unchanged(self):
        """Single analysis is returned as-is."""
        analysis = _make_analysis(
            q02_thesis="Test thesis",
            q01_research_question="What is the impact?",
        )
        result = aggregate_analyses([analysis])
        assert result.q02_thesis == "Test thesis"
        assert result.q01_research_question == "What is the impact?"

    def test_longest_string_wins(self):
        """Longest string is selected for q-fields (LONGEST strategy)."""
        analyses = [
            _make_analysis(q02_thesis="Short"),
            _make_analysis(q02_thesis="This is a much longer thesis statement"),
        ]
        result = aggregate_analyses(analyses)
        assert result.q02_thesis == "This is a much longer thesis statement"

    def test_longest_across_multiple_fields(self):
        """LONGEST strategy applies to all q-fields independently."""
        analyses = [
            _make_analysis(
                q01_research_question="Short question",
                q02_thesis="A longer thesis statement here",
            ),
            _make_analysis(
                q01_research_question="A much more detailed research question about networks",
                q02_thesis="Brief",
            ),
        ]
        result = aggregate_analyses(analyses)
        assert "detailed research question" in result.q01_research_question
        assert "longer thesis statement" in result.q02_thesis

    def test_none_fields_skipped(self):
        """None values are skipped when selecting longest."""
        analyses = [
            _make_analysis(q05_limitations=None),
            _make_analysis(q05_limitations="Some limitations noted."),
        ]
        result = aggregate_analyses(analyses)
        assert result.q05_limitations == "Some limitations noted."

    def test_all_none_stays_none(self):
        """If all providers return None for a field, result is None."""
        analyses = [
            _make_analysis(q40_policy_recommendations=None),
            _make_analysis(q40_policy_recommendations=None),
        ]
        result = aggregate_analyses(analyses)
        assert result.q40_policy_recommendations is None

    def test_metadata_from_first_analysis(self):
        """Required metadata fields come from the first analysis."""
        analyses = [
            _make_analysis(paper_id="first", extraction_model="model-a"),
            _make_analysis(paper_id="second", extraction_model="model-b"),
        ]
        result = aggregate_analyses(analyses)
        assert result.paper_id == "first"
        assert result.extraction_model == "model-a"


class TestCalculateConsensusConfidence:
    """Tests for consensus confidence calculation."""

    def test_empty_analyses_zero_confidence(self):
        """Empty analyses give zero confidence."""
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
        analyses = [
            _make_analysis(q02_thesis="Same thesis", q01_research_question="Same RQ"),
            _make_analysis(q02_thesis="Same thesis", q01_research_question="Same RQ"),
        ]
        confidence = calculate_consensus_confidence(analyses, config)
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
        analyses = [
            _make_analysis(q01_research_question="RQ about networks"),
        ]
        confidence = calculate_consensus_confidence(analyses, config)
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
            consensus=_make_analysis(q02_thesis="Test"),
            provider_responses=[
                ProviderResponse(
                    provider="anthropic",
                    extraction=_make_analysis(),
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
        mock_result.extraction = _make_analysis(q02_thesis="Test")
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
        mock_result.extraction = _make_analysis(q02_thesis="Test")
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
        mock_result.extraction = _make_analysis(q02_thesis="Test")
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
        """Two providers produce a consensus extraction via LONGEST strategy."""
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

        analysis_a = _make_analysis(
            q02_thesis="Short thesis",
            q01_research_question="What is the impact of graphs?",
            q07_methods="quantitative regression",
        )
        analysis_b = _make_analysis(
            q02_thesis="A more detailed and longer thesis statement for testing",
            q01_research_question="Impact?",
            q07_methods="quantitative regression with cross-validation and bootstrapping",
        )

        clients = {
            "anthropic": self._make_mock_client(analysis_a),
            "openai": self._make_mock_client(analysis_b),
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
        assert "more detailed" in result.consensus.q02_thesis
        # LONGEST strategy: longer research question wins
        assert "impact of graphs" in result.consensus.q01_research_question
        # LONGEST strategy: longer methods wins
        assert "cross-validation" in result.consensus.q07_methods
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

        analysis_a = _make_analysis(
            q02_thesis="Thesis from anthropic",
            q01_research_question="RQ A",
        )
        analysis_b = _make_analysis(
            q02_thesis="Thesis from openai provider is longer",
            q01_research_question="A much longer RQ B with details",
        )

        clients = {
            "anthropic": self._make_mock_client(analysis_a),
            "openai": self._make_mock_client(analysis_b),
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
        # LONGEST: longer thesis wins
        assert "openai provider" in result.consensus.q02_thesis

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

        analysis = _make_analysis(q02_thesis="Only thesis")

        clients = {
            "anthropic": self._make_mock_client(analysis),
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
        assert result.consensus.q02_thesis == "Only thesis"
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


class TestParallelExecution:
    """Tests for council parallel execution path."""

    def test_parallel_collects_all_results(self):
        """Parallel execution collects results from all providers."""
        from unittest.mock import MagicMock, patch

        config = CouncilConfig(
            providers=[
                ProviderConfig(name="provider_a"),
                ProviderConfig(name="provider_b"),
            ],
            min_responses=2,
            parallel=True,
        )
        council = LLMCouncil(config)

        def make_client(q02_thesis: str):
            client = MagicMock()
            result = MagicMock()
            result.success = True
            result.extraction = _make_analysis(q02_thesis=q02_thesis)
            result.cost = 0.01
            client.extract.return_value = result
            return client

        clients = {
            "provider_a": make_client("Thesis A"),
            "provider_b": make_client("Thesis B from provider B is longer"),
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
        assert len(result.provider_responses) == 2
        # Longest thesis wins consensus
        assert "provider B" in result.consensus.q02_thesis

    def test_provider_timeout_recorded_as_failure(self):
        """Provider that raises exception is recorded as failed response."""
        from unittest.mock import MagicMock, patch

        config = CouncilConfig(
            providers=[
                ProviderConfig(name="fast"),
                ProviderConfig(name="slow"),
            ],
            min_responses=1,
            fallback_to_single=True,
            parallel=False,  # Sequential so behavior is deterministic
        )
        council = LLMCouncil(config)

        fast_client = MagicMock()
        fast_result = MagicMock()
        fast_result.success = True
        fast_result.extraction = _make_analysis(q02_thesis="Fast result")
        fast_result.cost = 0.01
        fast_client.extract.return_value = fast_result

        slow_client = MagicMock()
        slow_client.extract.side_effect = TimeoutError("Provider timed out")

        clients = {"fast": fast_client, "slow": slow_client}

        with patch.object(council, "_get_client", side_effect=lambda name: clients[name]):
            result = council.extract(
                paper_id="test",
                title="Test",
                authors="Author",
                year=2024,
                item_type="article",
                text="Content",
            )

        assert result.success  # Fallback to single
        failed = [r for r in result.provider_responses if not r.success]
        assert len(failed) == 1
        assert "slow" in failed[0].provider
