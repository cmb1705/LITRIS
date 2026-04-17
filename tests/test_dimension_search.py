"""Tests for dimension_search module."""

from unittest.mock import MagicMock

import pytest

from src.analysis.schemas import SemanticAnalysis
from src.query.dimension_search import (
    VALID_DIMENSIONS,
    VALID_GROUPS,
    _normalize_dimension,
    search_dimension,
    search_group,
)


class TestValidDimensions:
    """Tests for VALID_DIMENSIONS constant."""

    def test_has_40_entries(self):
        """VALID_DIMENSIONS contains 40 dimension keys."""
        assert len(VALID_DIMENSIONS) == 40

    def test_format_is_q_nn(self):
        """Each entry follows qNN pattern."""
        for dim in VALID_DIMENSIONS:
            assert len(dim) == 3
            assert dim[0] == "q"
            assert dim[1:].isdigit()

    def test_sequential_q01_to_q40(self):
        """Entries run from q01 to q40."""
        assert VALID_DIMENSIONS[0] == "q01"
        assert VALID_DIMENSIONS[-1] == "q40"
        expected = [f"q{i:02d}" for i in range(1, 41)]
        assert VALID_DIMENSIONS == expected


class TestValidGroups:
    """Tests for VALID_GROUPS constant."""

    def test_has_six_groups(self):
        """VALID_GROUPS contains 6 entries."""
        assert len(VALID_GROUPS) == 6

    def test_expected_group_names(self):
        """Groups match SemanticAnalysis.DIMENSION_GROUPS keys."""
        expected = {"research_core", "methodology", "context", "meta", "scholarly", "impact"}
        assert set(VALID_GROUPS) == expected

    def test_matches_schema_groups(self):
        """VALID_GROUPS is derived from SemanticAnalysis.DIMENSION_GROUPS."""
        assert VALID_GROUPS == list(SemanticAnalysis.DIMENSION_GROUPS.keys())


class TestNormalizeDimension:
    """Tests for _normalize_dimension()."""

    def test_short_form_valid(self):
        """Short form q01-q40 passes through."""
        assert _normalize_dimension("q01") == "q01"
        assert _normalize_dimension("q07") == "q07"
        assert _normalize_dimension("q40") == "q40"

    def test_full_field_name_valid(self):
        """Full field name is normalized to short form."""
        assert _normalize_dimension("q01_research_question") == "q01"
        assert _normalize_dimension("q07_methods") == "q07"
        assert _normalize_dimension("q40_policy_recommendations") == "q40"

    def test_invalid_dimension_raises(self):
        """Invalid dimension identifier raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dimension"):
            _normalize_dimension("q00")

        with pytest.raises(ValueError, match="Invalid dimension"):
            _normalize_dimension("q41")

        with pytest.raises(ValueError, match="Invalid dimension"):
            _normalize_dimension("thesis")

        with pytest.raises(ValueError, match="Invalid dimension"):
            _normalize_dimension("invalid")

    def test_wrong_full_name_raises(self):
        """Valid short form prefix but wrong full name raises."""
        with pytest.raises(ValueError, match="Invalid dimension"):
            _normalize_dimension("q01_wrong_suffix")

    def test_all_40_short_forms(self):
        """All 40 short forms normalize correctly."""
        for i in range(1, 41):
            short = f"q{i:02d}"
            assert _normalize_dimension(short) == short

    def test_all_40_full_names(self):
        """All 40 full field names normalize correctly."""
        for field in SemanticAnalysis.DIMENSION_FIELDS:
            short = field[:3]
            assert _normalize_dimension(field) == short


class TestSearchDimension:
    """Tests for search_dimension()."""

    def test_calls_engine_with_correct_chunk_type(self):
        """search_dimension passes the right chunk type to engine.search()."""
        mock_engine = MagicMock()
        mock_engine.search.return_value = []

        search_dimension(mock_engine, "test query", "q02")

        mock_engine.search.assert_called_once_with(
            query="test query",
            top_k=10,
            chunk_types=["dim_q02"],
        )

    def test_full_field_name_resolves(self):
        """search_dimension works with full field names."""
        mock_engine = MagicMock()
        mock_engine.search.return_value = []

        search_dimension(mock_engine, "test query", "q07_methods")

        mock_engine.search.assert_called_once_with(
            query="test query",
            top_k=10,
            chunk_types=["dim_q07"],
        )

    def test_passes_top_k(self):
        """Custom top_k is forwarded to engine.search()."""
        mock_engine = MagicMock()
        mock_engine.search.return_value = []

        search_dimension(mock_engine, "test", "q01", top_k=5)

        call_kwargs = mock_engine.search.call_args[1]
        assert call_kwargs["top_k"] == 5

    def test_passes_extra_kwargs(self):
        """Extra keyword arguments are forwarded to engine.search()."""
        mock_engine = MagicMock()
        mock_engine.search.return_value = []

        search_dimension(
            mock_engine,
            "test",
            "q01",
            year_min=2020,
            collections=["ML"],
        )

        call_kwargs = mock_engine.search.call_args[1]
        assert call_kwargs["year_min"] == 2020
        assert call_kwargs["collections"] == ["ML"]

    def test_returns_engine_results(self):
        """Returns whatever engine.search() returns."""
        mock_engine = MagicMock()
        mock_results = [MagicMock(), MagicMock()]
        mock_engine.search.return_value = mock_results

        result = search_dimension(mock_engine, "test", "q01")

        assert result == mock_results

    def test_invalid_dimension_raises(self):
        """Invalid dimension raises ValueError before calling engine."""
        mock_engine = MagicMock()

        with pytest.raises(ValueError, match="Invalid dimension"):
            search_dimension(mock_engine, "test", "invalid")

        mock_engine.search.assert_not_called()


class TestSearchGroup:
    """Tests for search_group()."""

    def test_research_core_group(self):
        """research_core group searches q01-q05 dimensions."""
        mock_engine = MagicMock()
        mock_engine.search.return_value = []

        search_group(mock_engine, "test query", "research_core")

        call_kwargs = mock_engine.search.call_args[1]
        expected_types = ["dim_q01", "dim_q02", "dim_q03", "dim_q04", "dim_q05"]
        assert call_kwargs["chunk_types"] == expected_types

    def test_methodology_group(self):
        """methodology group searches q06-q10 dimensions."""
        mock_engine = MagicMock()
        mock_engine.search.return_value = []

        search_group(mock_engine, "test query", "methodology")

        call_kwargs = mock_engine.search.call_args[1]
        expected_types = ["dim_q06", "dim_q07", "dim_q08", "dim_q09", "dim_q10"]
        assert call_kwargs["chunk_types"] == expected_types

    def test_invalid_group_raises(self):
        """Invalid group name raises ValueError."""
        mock_engine = MagicMock()

        with pytest.raises(ValueError, match="Invalid group"):
            search_group(mock_engine, "test", "nonexistent_group")

        mock_engine.search.assert_not_called()

    def test_passes_top_k(self):
        """Custom top_k is forwarded."""
        mock_engine = MagicMock()
        mock_engine.search.return_value = []

        search_group(mock_engine, "test", "meta", top_k=20)

        call_kwargs = mock_engine.search.call_args[1]
        assert call_kwargs["top_k"] == 20

    def test_passes_extra_kwargs(self):
        """Extra keyword arguments are forwarded."""
        mock_engine = MagicMock()
        mock_engine.search.return_value = []

        search_group(
            mock_engine,
            "test",
            "context",
            year_min=2020,
            quality_min=3,
        )

        call_kwargs = mock_engine.search.call_args[1]
        assert call_kwargs["year_min"] == 2020
        assert call_kwargs["quality_min"] == 3

    def test_returns_engine_results(self):
        """Returns whatever engine.search() returns."""
        mock_engine = MagicMock()
        mock_results = [MagicMock()]
        mock_engine.search.return_value = mock_results

        result = search_group(mock_engine, "test", "impact")

        assert result == mock_results

    def test_all_groups_produce_valid_chunk_types(self):
        """Every group produces chunk types that start with dim_q."""
        mock_engine = MagicMock()
        mock_engine.search.return_value = []

        for group in VALID_GROUPS:
            search_group(mock_engine, "test", group)
            call_kwargs = mock_engine.search.call_args[1]
            for ct in call_kwargs["chunk_types"]:
                assert ct.startswith("dim_q"), f"Bad chunk type {ct} in group {group}"

    def test_impact_group_has_9_dimensions(self):
        """impact group covers q32-q40 (9 dimensions)."""
        mock_engine = MagicMock()
        mock_engine.search.return_value = []

        search_group(mock_engine, "test", "impact")

        call_kwargs = mock_engine.search.call_args[1]
        assert len(call_kwargs["chunk_types"]) == 9
