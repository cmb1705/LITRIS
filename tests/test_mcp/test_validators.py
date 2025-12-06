"""Tests for MCP input validators."""

import pytest

from src.mcp.validators import (
    ValidationError,
    validate_query,
    validate_paper_id,
    validate_top_k,
    validate_year,
    validate_chunk_types,
    validate_recency_boost,
)


class TestValidateQuery:
    """Tests for validate_query function."""

    def test_valid_query(self):
        """Valid query passes validation."""
        result = validate_query("citation network analysis")
        assert result == "citation network analysis"

    def test_query_with_whitespace(self):
        """Query with leading/trailing whitespace is trimmed."""
        result = validate_query("  citation analysis  ")
        assert result == "citation analysis"

    def test_empty_query_raises(self):
        """Empty query raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_query("")

    def test_whitespace_only_query_raises(self):
        """Whitespace-only query raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_query("   ")

    def test_query_too_long_raises(self):
        """Query exceeding max length raises ValidationError."""
        long_query = "a" * 1001
        with pytest.raises(ValidationError, match="too long"):
            validate_query(long_query)

    def test_query_at_max_length(self):
        """Query at exactly max length passes."""
        max_query = "a" * 1000
        result = validate_query(max_query)
        assert len(result) == 1000


class TestValidatePaperId:
    """Tests for validate_paper_id function."""

    def test_valid_paper_id(self):
        """Valid paper ID passes validation."""
        result = validate_paper_id("paper_123")
        assert result == "paper_123"

    def test_paper_id_with_hyphen(self):
        """Paper ID with hyphen is valid."""
        result = validate_paper_id("paper-123-abc")
        assert result == "paper-123-abc"

    def test_empty_paper_id_raises(self):
        """Empty paper ID raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_paper_id("")

    def test_invalid_characters_raises(self):
        """Paper ID with invalid characters raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid paper ID format"):
            validate_paper_id("paper@123")


class TestValidateTopK:
    """Tests for validate_top_k function."""

    def test_valid_top_k(self):
        """Valid top_k passes through unchanged."""
        assert validate_top_k(10) == 10

    def test_top_k_below_minimum(self):
        """top_k below 1 is clamped to 1."""
        assert validate_top_k(0) == 1
        assert validate_top_k(-5) == 1

    def test_top_k_above_maximum(self):
        """top_k above 50 is clamped to 50."""
        assert validate_top_k(100) == 50

    def test_top_k_at_boundaries(self):
        """top_k at boundaries passes through."""
        assert validate_top_k(1) == 1
        assert validate_top_k(50) == 50


class TestValidateYear:
    """Tests for validate_year function."""

    def test_valid_year(self):
        """Valid year passes validation."""
        assert validate_year(2023) == 2023

    def test_year_below_minimum_raises(self):
        """Year below 1800 raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid year"):
            validate_year(1799)

    def test_year_above_maximum_raises(self):
        """Year above 2100 raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid year"):
            validate_year(2101)

    def test_custom_param_name_in_error(self):
        """Custom parameter name appears in error message."""
        with pytest.raises(ValidationError, match="year_min"):
            validate_year(1799, "year_min")


class TestValidateChunkTypes:
    """Tests for validate_chunk_types function."""

    def test_valid_chunk_types(self):
        """Valid chunk types pass validation."""
        result = validate_chunk_types(["thesis", "methodology"])
        assert result == ["thesis", "methodology"]

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert validate_chunk_types([]) == []

    def test_none_returns_empty(self):
        """None returns empty list."""
        assert validate_chunk_types(None) == []

    def test_invalid_chunk_type_raises(self):
        """Invalid chunk type raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid chunk types"):
            validate_chunk_types(["thesis", "invalid_type"])

    def test_all_valid_types(self):
        """All valid chunk types pass."""
        valid_types = [
            "abstract",
            "thesis",
            "contribution",
            "methodology",
            "findings",
            "claims",
            "limitations",
            "future_work",
            "full_summary",
        ]
        result = validate_chunk_types(valid_types)
        assert result == valid_types


class TestValidateRecencyBoost:
    """Tests for validate_recency_boost function."""

    def test_valid_boost(self):
        """Valid boost value passes through."""
        assert validate_recency_boost(0.5) == 0.5

    def test_boost_below_zero(self):
        """Boost below 0.0 is clamped to 0.0."""
        assert validate_recency_boost(-0.5) == 0.0

    def test_boost_above_one(self):
        """Boost above 1.0 is clamped to 1.0."""
        assert validate_recency_boost(1.5) == 1.0

    def test_boost_at_boundaries(self):
        """Boost at boundaries passes through."""
        assert validate_recency_boost(0.0) == 0.0
        assert validate_recency_boost(1.0) == 1.0
