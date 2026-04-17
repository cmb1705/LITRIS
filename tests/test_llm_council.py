"""Tests for LLM council consensus extraction."""

import pytest

from src.analysis.llm_council import (
    STRATEGY_REGISTRY,
    CouncilConfig,
    CouncilResult,
    LLMCouncil,
    ProviderConfig,
    ProviderResponse,
    SynthesisConfig,
    _compute_quality_score,
    aggregate_analyses,
    calculate_consensus_confidence,
    identify_gap_passes,
    strategy_quality_weighted,
    strategy_union_merge,
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
        # 6 passes x 0.10 = 0.60 total, exceeds 0.05
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.extraction = _make_analysis(q02_thesis="Test")
        mock_result.cost = 0.10
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
            providers=[ProviderConfig(name="cheap", max_cost=1.0)],
            min_responses=1,
            fallback_to_single=True,
        )
        council = LLMCouncil(config)

        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.extraction = _make_analysis(q02_thesis="Test")
        mock_result.cost = 0.01  # 6 passes x 0.01 = 0.06 total, within 1.0
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
        assert result.provider_responses[0].cost > 0

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
        mock_client.model = "test-model"
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
        mock_client.model = "test-model"
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
        mock_client.model = "test-model"
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

        with patch.object(
            council,
            "_get_client",
            side_effect=lambda p: clients[p.name if hasattr(p, "name") else p],
        ):
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

        with patch.object(
            council,
            "_get_client",
            side_effect=lambda p: clients[p.name if hasattr(p, "name") else p],
        ):
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

        with patch.object(
            council,
            "_get_client",
            side_effect=lambda p: clients[p.name if hasattr(p, "name") else p],
        ):
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

        with patch.object(
            council,
            "_get_client",
            side_effect=lambda p: clients[p.name if hasattr(p, "name") else p],
        ):
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
            client.model = "test-model"
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

        with patch.object(
            council,
            "_get_client",
            side_effect=lambda p: clients[p.name if hasattr(p, "name") else p],
        ):
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
        fast_client.model = "test-model"
        fast_result = MagicMock()
        fast_result.success = True
        fast_result.extraction = _make_analysis(q02_thesis="Fast result")
        fast_result.cost = 0.01
        fast_client.extract.return_value = fast_result

        slow_client = MagicMock()
        slow_client.model = "test-model"
        slow_client.extract.side_effect = TimeoutError("Provider timed out")

        clients = {"fast": fast_client, "slow": slow_client}

        with patch.object(
            council,
            "_get_client",
            side_effect=lambda p: clients[p.name if hasattr(p, "name") else p],
        ):
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


# =====================================================================
# Tier 1: ProviderConfig mode/model, quality scoring, weighted agg
# =====================================================================


class TestProviderConfigMode:
    """Tests for mode and model fields on ProviderConfig."""

    def test_default_mode_is_cli(self):
        """Default mode should be 'cli' to avoid API costs."""
        config = ProviderConfig(name="anthropic")
        assert config.mode == "cli"
        assert config.model is None

    def test_custom_mode_and_model(self):
        """Custom mode and model are accepted."""
        config = ProviderConfig(
            name="openai",
            mode="api",
            model="gpt-5.4",
        )
        assert config.mode == "api"
        assert config.model == "gpt-5.4"


class TestGetClientWithMode:
    """Tests for _get_client receiving mode/model from ProviderConfig."""

    def test_mode_passed_to_factory(self):
        """create_llm_client receives mode and model from ProviderConfig."""
        from unittest.mock import MagicMock, patch

        config = CouncilConfig(providers=[])
        council = LLMCouncil(config)

        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock()

        provider = ProviderConfig(name="anthropic", mode="cli", model="test-model")

        with patch("src.analysis.llm_factory.create_llm_client", mock_factory):
            council._get_client(provider)

        mock_factory.assert_called_once_with(
            provider="anthropic",
            mode="cli",
            model="test-model",
            timeout=120,
        )

    def test_cache_key_includes_mode_and_model(self):
        """Different mode/model combos produce different cache keys."""
        from unittest.mock import MagicMock, patch

        config = CouncilConfig(providers=[])
        council = LLMCouncil(config)

        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock()

        p1 = ProviderConfig(name="anthropic", mode="cli")
        p2 = ProviderConfig(name="anthropic", mode="api")

        with patch("src.analysis.llm_factory.create_llm_client", mock_factory):
            council._get_client(p1)
            council._get_client(p2)

        assert mock_factory.call_count == 2


class TestQualityScore:
    """Tests for _compute_quality_score."""

    def test_empty_string_scores_zero(self):
        """Empty string gets zero score."""
        assert _compute_quality_score("") == 0.0

    def test_short_text_penalized(self):
        """Text under 50 chars gets 0.5x multiplier."""
        score = _compute_quality_score("Short.")
        assert score < 0.2

    def test_citation_boosts_score(self):
        """Citation patterns increase score."""
        text = "This builds on prior work (Smith, 2020). The analysis uses (Jones et al., 2019)."
        score = _compute_quality_score(text)
        assert score > 0.3

    def test_numbers_boost_score(self):
        """Numeric content increases score."""
        text = "The model achieved 95.2% accuracy on 1000 samples across 5 domains."
        score = _compute_quality_score(text)
        assert score > 0.2

    def test_ideal_sentence_count(self):
        """2-5 sentences get maximum sentence bonus."""
        text = "First finding is important. Second finding builds on it. Third finding adds nuance."
        score = _compute_quality_score(text)
        assert score >= 0.3


class TestQualityWeightedAggregation:
    """Tests for quality_weighted aggregation strategy."""

    def test_higher_quality_shorter_wins(self):
        """A shorter but higher-quality response should beat verbose low-quality."""
        verbose = "word " * 200  # 200 words, no structure
        concise = (
            "The study found 95% accuracy (Smith, 2020). "
            "Cross-validation confirmed results across 5 datasets."
        )
        result = strategy_quality_weighted(
            [
                (verbose, 1.0),
                (concise, 1.0),
            ]
        )
        assert result == concise

    def test_weight_breaks_tie(self):
        """When quality is similar, higher weight wins."""
        text_a = "Finding one is notable. Finding two confirms it."
        text_b = "Finding one is notable. Finding two confirms it."
        result = strategy_quality_weighted(
            [
                (text_a, 0.5),
                (text_b, 2.0),
            ]
        )
        # Same quality, but text_b has higher weight
        assert result == text_b

    def test_all_none_returns_none(self):
        """All None values return None."""
        assert strategy_quality_weighted([(None, 1.0), (None, 1.0)]) is None

    def test_passes_through_aggregate(self):
        """quality_weighted strategy works via aggregate_analyses."""
        analyses = [
            _make_analysis(
                q02_thesis="word " * 200,
            ),
            _make_analysis(
                q02_thesis=(
                    "Network analysis reveals 87% clustering (Jones, 2021). "
                    "Results replicate across 3 independent datasets."
                ),
            ),
        ]
        result = aggregate_analyses(
            analyses,
            weights=[1.0, 1.0],
            default_strategy="quality_weighted",
        )
        assert "87%" in result.q02_thesis


class TestCouncilConfigNewFields:
    """Tests for new CouncilConfig fields."""

    def test_aggregation_strategy_default(self):
        """Default aggregation strategy is 'quality_weighted'."""
        config = CouncilConfig()
        assert config.aggregation_strategy == "quality_weighted"

    def test_field_strategies_default_empty(self):
        """field_strategies defaults to empty dict."""
        config = CouncilConfig()
        assert config.field_strategies == {}

    def test_synthesis_disabled_by_default(self):
        """Synthesis is disabled by default."""
        config = CouncilConfig()
        assert config.synthesis.enabled is False


# =====================================================================
# Tier 2: Strategy registry, union merge, per-field overrides
# =====================================================================


class TestStrategyRegistry:
    """Tests for strategy registry."""

    def test_all_builtin_strategies_registered(self):
        """All built-in strategies are in STRATEGY_REGISTRY."""
        assert "longest" in STRATEGY_REGISTRY
        assert "quality_weighted" in STRATEGY_REGISTRY
        assert "union" in STRATEGY_REGISTRY

    def test_registry_functions_callable(self):
        """All registered strategies are callable."""
        for name, fn in STRATEGY_REGISTRY.items():
            assert callable(fn), f"Strategy '{name}' is not callable"


class TestStrategyUnionMerge:
    """Tests for union merge strategy."""

    def test_unique_sentences_merged(self):
        """Unique sentences from different providers are combined."""
        result = strategy_union_merge(
            [
                ("First finding is important. Second finding is notable.", 1.0),
                ("Third finding adds context. Fourth insight is key.", 1.0),
            ]
        )
        assert "First finding" in result
        assert "Third finding" in result

    def test_duplicate_sentences_removed(self):
        """Near-duplicate sentences are deduplicated."""
        result = strategy_union_merge(
            [
                ("The study found significant results.", 1.0),
                ("The study found significant results.", 0.8),
            ]
        )
        # Should only appear once
        assert result.count("significant results") == 1

    def test_higher_weight_phrasing_preserved(self):
        """Higher-weight provider's sentences are processed first."""
        result = strategy_union_merge(
            [
                ("Lower weight version of the finding.", 0.5),
                ("Higher weight version of the finding.", 2.0),
            ]
        )
        # Higher weight is processed first, so its phrasing wins
        assert "Higher weight" in result

    def test_all_none_returns_none(self):
        """All None inputs return None."""
        assert strategy_union_merge([(None, 1.0), (None, 1.0)]) is None

    def test_overlap_threshold(self):
        """Sentences with <60% word overlap are both kept."""
        result = strategy_union_merge(
            [
                ("Network analysis reveals community structure.", 1.0),
                ("Graph theory provides mathematical framework.", 1.0),
            ]
        )
        assert "Network" in result
        assert "Graph" in result


class TestPerFieldStrategy:
    """Tests for per-field strategy overrides."""

    def test_field_strategy_overrides_default(self):
        """field_strategies override the default strategy for specific fields."""
        analyses = [
            _make_analysis(
                q02_thesis="Short thesis",
                q03_key_claims="Claim A is supported. Claim B needs evidence.",
            ),
            _make_analysis(
                q02_thesis="A much longer thesis with additional detail and nuance",
                q03_key_claims="Claim C is novel. Claim D extends prior work.",
            ),
        ]
        result = aggregate_analyses(
            analyses,
            weights=[1.0, 1.0],
            default_strategy="longest",
            field_strategies={"q03_key_claims": "union"},
        )
        # q02 uses longest (default)
        assert "longer thesis" in result.q02_thesis
        # q03 uses union -- both providers' claims should appear
        assert "Claim A" in result.q03_key_claims
        assert "Claim C" in result.q03_key_claims

    def test_unknown_strategy_falls_back_to_longest(self):
        """Unknown strategy name falls back to strategy_longest."""
        analyses = [
            _make_analysis(q02_thesis="Short"),
            _make_analysis(q02_thesis="Much longer thesis here"),
        ]
        result = aggregate_analyses(
            analyses,
            default_strategy="nonexistent_strategy",
        )
        assert result.q02_thesis == "Much longer thesis here"


class TestImprovedConfidence:
    """Tests for improved consensus confidence calculation."""

    def test_agreement_on_core_fields_boosts_confidence(self):
        """High word overlap on core fields increases confidence."""
        config = CouncilConfig(providers=[ProviderConfig(name="a"), ProviderConfig(name="b")])
        analyses = [
            _make_analysis(
                q01_research_question="What is the impact of network analysis on policy?",
                q02_thesis="Network analysis improves policy decisions.",
                q03_key_claims="Networks reveal hidden structure.",
                q04_evidence="Quantitative survey of 500 respondents.",
                q05_limitations="Limited to US context.",
            ),
            _make_analysis(
                q01_research_question="What is the impact of network analysis on policy making?",
                q02_thesis="Network analysis improves policy decision making.",
                q03_key_claims="Networks reveal hidden structure in organizations.",
                q04_evidence="Quantitative survey of 500 respondents across 3 cities.",
                q05_limitations="Limited to US context only.",
            ),
        ]
        confidence = calculate_consensus_confidence(analyses, config)
        assert confidence > 0.7

    def test_coverage_spread_affects_confidence(self):
        """Divergent field coverage lowers confidence."""
        config = CouncilConfig(providers=[ProviderConfig(name="a"), ProviderConfig(name="b")])
        # Provider A fills many fields, provider B fills few
        a_fields = {
            f"q{i:02d}_{name}": f"Value for {name}"
            for i, name in enumerate(["research_question", "thesis"], 1)
        }
        b_fields = {}
        for field_name in SemanticAnalysis.DIMENSION_FIELDS[:30]:
            a_fields[field_name] = f"Filled by A: {field_name}"

        analyses = [
            _make_analysis(**a_fields),
            _make_analysis(**b_fields),
        ]
        confidence = calculate_consensus_confidence(analyses, config)
        # Should be lower due to coverage divergence
        assert confidence < 0.9


# =====================================================================
# Tier 3: Synthesis round, query method, raw_query
# =====================================================================


class TestSynthesisConfig:
    """Tests for SynthesisConfig dataclass."""

    def test_defaults(self):
        """SynthesisConfig has sensible defaults."""
        config = SynthesisConfig()
        assert config.enabled is False
        assert config.judge_provider == "anthropic"
        assert config.judge_mode == "api"
        assert config.judge_model is None


class TestSynthesisPrompt:
    """Tests for synthesis prompt building."""

    def test_prompt_includes_all_providers(self):
        """Synthesis prompt includes extractions from all providers."""
        config = CouncilConfig(providers=[])
        council = LLMCouncil(config)

        responses = [
            ProviderResponse(
                provider="anthropic",
                extraction=_make_analysis(
                    q02_thesis="Anthropic thesis",
                    q07_methods="Qualitative",
                ),
                success=True,
            ),
            ProviderResponse(
                provider="openai",
                extraction=_make_analysis(
                    q02_thesis="OpenAI thesis",
                    q07_methods="Quantitative",
                ),
                success=True,
            ),
        ]

        prompt = council._build_synthesis_prompt(responses)
        assert "anthropic" in prompt.lower()
        assert "openai" in prompt.lower()
        assert "Anthropic thesis" in prompt
        assert "OpenAI thesis" in prompt


class TestSynthesisRound:
    """Tests for synthesis round execution."""

    def test_synthesis_used_as_consensus(self):
        """When synthesis is enabled and succeeds, it replaces mechanical aggregation."""
        from unittest.mock import MagicMock, patch

        synth_extraction = _make_analysis(
            q02_thesis="Synthesized best thesis from judge",
        )

        config = CouncilConfig(
            providers=[
                ProviderConfig(name="anthropic"),
                ProviderConfig(name="openai"),
            ],
            min_responses=2,
            parallel=False,
            synthesis=SynthesisConfig(
                enabled=True,
                judge_provider="anthropic",
                judge_mode="api",
            ),
        )
        council = LLMCouncil(config)

        analysis_a = _make_analysis(q02_thesis="Thesis A")
        analysis_b = _make_analysis(q02_thesis="Thesis B is longer for testing")

        def mock_get_client(provider):
            name = provider.name if hasattr(provider, "name") else provider
            mode = provider.mode if hasattr(provider, "mode") else "cli"
            client = MagicMock()

            if name == "anthropic" and mode == "api":
                # Judge synthesis call -- return synthesized extraction
                result = MagicMock()
                result.success = True
                result.extraction = synth_extraction
                result.cost = 0.01
                client.extract.return_value = result
            elif name == "anthropic":
                # Provider extraction call
                result = MagicMock()
                result.success = True
                result.extraction = analysis_a
                result.cost = 0.01
                client.extract.return_value = result
            else:
                result = MagicMock()
                result.success = True
                result.extraction = analysis_b
                result.cost = 0.01
                client.extract.return_value = result
            return client

        with patch.object(council, "_get_client", side_effect=mock_get_client):
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
        assert "Synthesized" in result.consensus.q02_thesis

    def test_synthesis_disabled_not_called(self):
        """Synthesis is not called when disabled."""
        from unittest.mock import MagicMock, patch

        config = CouncilConfig(
            providers=[
                ProviderConfig(name="anthropic"),
                ProviderConfig(name="openai"),
            ],
            min_responses=2,
            parallel=False,
            synthesis=SynthesisConfig(enabled=False),
        )
        council = LLMCouncil(config)

        analysis_a = _make_analysis(q02_thesis="Thesis A short")
        analysis_b = _make_analysis(q02_thesis="Thesis B is the longest one here")

        clients = {
            "anthropic": MagicMock(),
            "openai": MagicMock(),
        }

        for name, extraction in [("anthropic", analysis_a), ("openai", analysis_b)]:
            result_mock = MagicMock()
            result_mock.success = True
            result_mock.extraction = extraction
            result_mock.cost = 0.01
            clients[name].extract.return_value = result_mock

        with patch.object(
            council,
            "_get_client",
            side_effect=lambda p: clients[p.name if hasattr(p, "name") else p],
        ):
            with patch.object(council, "_run_synthesis_round") as mock_synth:
                result = council.extract(
                    paper_id="test",
                    title="Test",
                    authors="Author",
                    year=2024,
                    item_type="article",
                    text="Content",
                )

        mock_synth.assert_not_called()
        assert result.success
        # Should use mechanical aggregation (longest)
        assert "longest one" in result.consensus.q02_thesis


class TestQueryMethod:
    """Tests for generic query() method."""

    def test_query_fans_out_to_providers(self):
        """query() sends prompt to all enabled providers."""
        from unittest.mock import MagicMock, patch

        config = CouncilConfig(
            providers=[
                ProviderConfig(name="anthropic"),
                ProviderConfig(name="openai"),
            ],
        )
        council = LLMCouncil(config)

        def mock_get_client(provider):
            client = MagicMock()
            client.raw_query.return_value = (
                f"Response from {provider.name}",
                True,
                None,
            )
            return client

        with patch.object(council, "_get_client", side_effect=mock_get_client):
            result = council.query("What is network analysis?", query_id="q1")

        assert result.success
        assert len(result.provider_responses) == 2
        assert result.consensus_response is not None
        assert result.query_id == "q1"

    def test_query_no_providers(self):
        """query() with no providers returns failure."""
        config = CouncilConfig(providers=[])
        council = LLMCouncil(config)
        result = council.query("test prompt")
        assert not result.success
        assert result.provider_responses == []


class TestRawQuery:
    """Tests for BaseLLMClient.raw_query default implementation."""

    def test_raw_query_wraps_extract(self):
        """raw_query delegates to extract with prompt_override."""
        from unittest.mock import MagicMock

        from src.analysis.base_llm import BaseLLMClient

        # Create a concrete subclass for testing
        class MockClient(BaseLLMClient):
            @property
            def provider(self):
                return "anthropic"

            @property
            def default_model(self):
                return "test"

            @property
            def supported_modes(self):
                return ["api"]

            def extract(self, **kwargs):
                extraction = _make_analysis(
                    q02_thesis="Response to query",
                )
                result = MagicMock()
                result.success = True
                result.extraction = extraction
                return result

            def estimate_cost(self, text_length):
                return 0.0

        client = MockClient()
        text, success, error = client.raw_query("Tell me about networks")
        assert success
        assert text is not None
        assert "Response to query" in text

    def test_raw_query_handles_failure(self):
        """raw_query returns error tuple on exception."""
        from src.analysis.base_llm import BaseLLMClient

        class FailingClient(BaseLLMClient):
            @property
            def provider(self):
                return "anthropic"

            @property
            def default_model(self):
                return "test"

            @property
            def supported_modes(self):
                return ["api"]

            def extract(self, **kwargs):
                raise RuntimeError("API error")

            def estimate_cost(self, text_length):
                return 0.0

        client = FailingClient()
        text, success, error = client.raw_query("test")
        assert not success
        assert error is not None
        assert "API error" in error


# =====================================================================
# Graceful fallback and warning tests
# =====================================================================


class TestGracefulFallbacks:
    """Tests for graceful degradation when things go wrong."""

    def test_unknown_strategy_warns_and_uses_longest(self, caplog):
        """Unknown strategy name logs a warning and falls back to longest."""
        import logging

        analyses = [
            _make_analysis(q02_thesis="Short"),
            _make_analysis(q02_thesis="Much longer thesis here for testing"),
        ]
        with caplog.at_level(logging.WARNING, logger="src.analysis.llm_council"):
            result = aggregate_analyses(
                analyses,
                field_strategies={"q02_thesis": "nonexistent_strategy"},
            )
        assert result.q02_thesis == "Much longer thesis here for testing"
        assert any("Unknown aggregation strategy" in msg for msg in caplog.messages)

    def test_mismatched_weights_warns_and_pads(self, caplog):
        """Too few weights logs a warning and pads with 1.0."""
        import logging

        analyses = [
            _make_analysis(q02_thesis="A"),
            _make_analysis(q02_thesis="BB"),
            _make_analysis(q02_thesis="CCC"),
        ]
        with caplog.at_level(logging.WARNING, logger="src.analysis.llm_council"):
            result = aggregate_analyses(analyses, weights=[1.0])
        # Should still produce a result (longest)
        assert result.q02_thesis == "CCC"
        assert any("Weights length" in msg for msg in caplog.messages)

    def test_mismatched_weights_truncates(self, caplog):
        """Too many weights logs a warning and truncates."""
        import logging

        analyses = [
            _make_analysis(q02_thesis="A"),
            _make_analysis(q02_thesis="BB"),
        ]
        with caplog.at_level(logging.WARNING, logger="src.analysis.llm_council"):
            result = aggregate_analyses(analyses, weights=[1.0, 2.0, 3.0])
        assert result.q02_thesis == "BB"
        assert any("Weights length" in msg for msg in caplog.messages)

    def test_synthesis_with_no_valid_extractions_returns_none(self, caplog):
        """Synthesis round returns None when no responses have extractions."""
        import logging

        config = CouncilConfig(
            providers=[],
            synthesis=SynthesisConfig(enabled=True),
        )
        council = LLMCouncil(config)

        responses = [
            ProviderResponse(provider="a", extraction=None, success=True),
            ProviderResponse(provider="b", extraction=None, success=True),
        ]

        with caplog.at_level(logging.WARNING, logger="src.analysis.llm_council"):
            result = council._run_synthesis_round(responses)

        assert result is None
        assert any("no valid extractions" in msg for msg in caplog.messages)

    def test_synthesis_judge_failure_falls_back_to_mechanical(self):
        """If synthesis judge fails, mechanical aggregation is preserved."""
        from unittest.mock import MagicMock, patch

        config = CouncilConfig(
            providers=[
                ProviderConfig(name="anthropic"),
                ProviderConfig(name="openai"),
            ],
            min_responses=2,
            parallel=False,
            synthesis=SynthesisConfig(
                enabled=True,
                judge_provider="anthropic",
                judge_mode="api",
            ),
        )
        council = LLMCouncil(config)

        analysis_a = _make_analysis(q02_thesis="Thesis A short")
        analysis_b = _make_analysis(q02_thesis="Thesis B is the longest one here")

        def mock_get_client(provider):
            name = provider.name if hasattr(provider, "name") else provider
            mode = provider.mode if hasattr(provider, "mode") else "cli"
            client = MagicMock()

            if name == "anthropic" and mode == "api":
                # Judge fails
                client.extract.side_effect = RuntimeError("Judge API error")
            elif name == "anthropic":
                result = MagicMock()
                result.success = True
                result.extraction = analysis_a
                result.cost = 0.01
                client.extract.return_value = result
            else:
                result = MagicMock()
                result.success = True
                result.extraction = analysis_b
                result.cost = 0.01
                client.extract.return_value = result
            return client

        with patch.object(council, "_get_client", side_effect=mock_get_client):
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
        # Should fall back to mechanical aggregation (longest)
        assert "longest one" in result.consensus.q02_thesis

    def test_query_provider_error_captured(self):
        """query() captures provider errors without crashing."""
        from unittest.mock import MagicMock, patch

        config = CouncilConfig(
            providers=[
                ProviderConfig(name="good_provider"),
                ProviderConfig(name="bad_provider"),
            ],
        )
        council = LLMCouncil(config)

        def mock_get_client(provider):
            name = provider.name if hasattr(provider, "name") else provider
            client = MagicMock()
            if name == "bad_provider":
                client.raw_query.side_effect = RuntimeError("Connection refused")
            else:
                client.raw_query.return_value = ("Good response", True, None)
            return client

        with patch.object(council, "_get_client", side_effect=mock_get_client):
            result = council.query("test prompt")

        assert result.success  # Still succeeds with 1 good provider
        assert len(result.provider_responses) == 2
        failed = [r for r in result.provider_responses if not r.success]
        assert len(failed) == 1
        assert "Connection refused" in failed[0].error

    def test_quality_score_handles_punctuation_only(self):
        """Quality score handles text that is only punctuation."""
        score = _compute_quality_score("...")
        assert score == 0.0  # No sentences, no content

    def test_union_merge_single_provider(self):
        """Union merge with single provider returns its content."""
        result = strategy_union_merge(
            [
                ("First sentence. Second sentence.", 1.0),
            ]
        )
        assert "First sentence" in result
        assert "Second sentence" in result

    def test_aggregate_with_all_providers_returning_none(self):
        """Aggregation handles all providers returning None for a field."""
        analyses = [
            _make_analysis(),
            _make_analysis(),
            _make_analysis(),
        ]
        result = aggregate_analyses(analyses, default_strategy="quality_weighted")
        # All q-fields are None, should remain None
        assert result.q02_thesis is None
        assert result.q07_methods is None

    def test_confidence_single_analysis(self):
        """Confidence with single analysis returns reasonable value."""
        config = CouncilConfig(providers=[ProviderConfig(name="only_one")])
        analyses = [_make_analysis(q02_thesis="Solo thesis")]
        confidence = calculate_consensus_confidence(analyses, config)
        # 1/1 response rate = 0.6, no agreement data = 0.5 default
        assert 0.5 <= confidence <= 1.0

    def test_synthesis_prompt_empty_dimensions(self):
        """Synthesis prompt with extractions that have no filled dimensions."""
        config = CouncilConfig(providers=[])
        council = LLMCouncil(config)

        responses = [
            ProviderResponse(
                provider="anthropic",
                extraction=_make_analysis(),  # All dimensions None
                success=True,
            ),
        ]

        prompt = council._build_synthesis_prompt(responses)
        # Should still produce a valid prompt, just with empty JSON
        assert "Provider: anthropic" in prompt


# =====================================================================
# Tier 4: Gap-filling extraction
# =====================================================================


class TestGapFilling:
    """Tests for gap-filling extraction via fill_gaps()."""

    def test_identifies_gap_passes(self):
        """identify_gap_passes maps unfilled fields to correct pass numbers."""
        # Fill only pass 1 fields (q01-q05), leave rest as None
        analysis = _make_analysis(
            q01_research_question="Filled",
            q02_thesis="Filled",
            q03_key_claims="Filled",
            q04_evidence="Filled",
            q05_limitations="Filled",
        )
        gap_passes = identify_gap_passes(analysis)

        # Pass 1 should NOT be in gaps (all filled)
        assert 1 not in gap_passes
        # Passes 2-6 should be present
        assert 2 in gap_passes
        assert 3 in gap_passes
        assert 4 in gap_passes
        assert 5 in gap_passes
        assert 6 in gap_passes
        # Pass 2 should contain q06-q10
        assert "q06_paradigm" in gap_passes[2]
        assert "q10_framework" in gap_passes[2]

    def test_fills_missing_fields(self):
        """fill_gaps runs gap passes and merges results into original analysis."""
        from unittest.mock import MagicMock, patch

        # Original analysis has pass 1 filled, everything else None
        original = _make_analysis(
            q01_research_question="Original research question",
            q02_thesis="Original thesis statement",
            q03_key_claims="Original key claims",
            q04_evidence="Original evidence description",
            q05_limitations="Original limitations",
        )

        config = CouncilConfig(
            providers=[
                ProviderConfig(name="anthropic", weight=1.0),
                ProviderConfig(name="openai", weight=1.0),
            ],
        )
        council = LLMCouncil(config)

        # Mock the secondary provider to fill pass 2 fields
        mock_client = MagicMock()
        mock_client.model = "openai-model"
        gap_extraction = _make_analysis(
            q06_paradigm="Positivist paradigm",
            q07_methods="Quantitative survey methods",
            q08_data="500 respondents over 3 years",
            q09_reproducibility="High reproducibility with shared data",
            q10_framework="Social network theory",
        )
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.extraction = gap_extraction
        mock_result.cost = 0.01
        mock_client.extract.return_value = mock_result

        with patch.object(council, "_get_client", return_value=mock_client):
            merged, gaps_filled = council.fill_gaps(
                analysis=original,
                paper_id="test_id",
                title="Test Paper",
                authors="Test Author",
                year=2024,
                item_type="article",
                text="Full paper text here.",
            )

        # Original pass 1 fields preserved
        assert merged.q01_research_question == "Original research question"
        assert merged.q02_thesis == "Original thesis statement"
        # Gap-fill fields now populated
        assert merged.q06_paradigm is not None
        assert gaps_filled > 0

    def test_no_gaps_returns_unchanged(self):
        """fill_gaps returns original analysis when all fields are filled."""
        # Build analysis with ALL 40 fields filled
        all_fields = {}
        for field_name in SemanticAnalysis.DIMENSION_FIELDS:
            all_fields[field_name] = f"Filled value for {field_name}"

        analysis = _make_analysis(**all_fields)

        config = CouncilConfig(
            providers=[ProviderConfig(name="anthropic")],
        )
        council = LLMCouncil(config)

        merged, gaps_filled = council.fill_gaps(
            analysis=analysis,
            paper_id="test_id",
            title="Test",
            authors="Author",
            year=2024,
            item_type="article",
            text="Content",
        )

        assert gaps_filled == 0
        # Should be the exact same object (no-op path)
        assert merged is analysis

    def test_uses_different_provider(self):
        """fill_gaps selects a provider different from the original extraction_model."""
        from unittest.mock import MagicMock, patch

        original = _make_analysis(
            extraction_model="anthropic",
            q01_research_question="Filled",
        )

        config = CouncilConfig(
            providers=[
                ProviderConfig(name="anthropic", weight=1.0),
                ProviderConfig(name="openai", weight=1.0),
            ],
        )
        council = LLMCouncil(config)

        mock_client = MagicMock()
        mock_client.model = "openai-model"
        gap_extraction = _make_analysis(
            q06_paradigm="Positivist approach to analysis",
        )
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.extraction = gap_extraction
        mock_result.cost = 0.01
        mock_client.extract.return_value = mock_result

        provider_used = None

        def capture_get_client(provider):
            nonlocal provider_used
            provider_used = provider
            return mock_client

        with patch.object(council, "_get_client", side_effect=capture_get_client):
            council.fill_gaps(
                analysis=original,
                paper_id="test_id",
                title="Test",
                authors="Author",
                year=2024,
                item_type="article",
                text="Content",
            )

        # Should have picked "openai" (not "anthropic" which matches extraction_model)
        assert provider_used is not None
        assert provider_used.name == "openai"
