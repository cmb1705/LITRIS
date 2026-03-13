"""Tests for quality rating derivation from q21_quality prose and search filtering."""

import pytest

from src.indexing.embeddings import EmbeddingGenerator
from src.mcp.validators import ValidationError, validate_quality_min


class TestDeriveQualityRating:
    """Tests for _derive_quality_rating() which parses q21_quality prose."""

    def test_explicit_rating_pattern_slash(self):
        """Test '4/5' pattern in prose."""
        assert EmbeddingGenerator._derive_quality_rating("Solid work. 4/5.") == 4

    def test_explicit_rating_pattern_rated(self):
        """Test 'rated X' pattern."""
        assert EmbeddingGenerator._derive_quality_rating("This paper is rated 3.") == 3

    def test_explicit_rating_pattern_score(self):
        """Test 'score: X' pattern."""
        assert EmbeddingGenerator._derive_quality_rating("Quality score: 5") == 5

    def test_none_returns_zero(self):
        """Test None input returns 0 (no rating)."""
        assert EmbeddingGenerator._derive_quality_rating(None) == 0

    def test_empty_string_returns_zero(self):
        """Test empty string returns 0."""
        assert EmbeddingGenerator._derive_quality_rating("") == 0

    def test_keyword_excellent(self):
        """Test 'excellent' keyword maps to 5."""
        assert EmbeddingGenerator._derive_quality_rating(
            "Excellent methodology with comprehensive analysis."
        ) == 5

    def test_keyword_strong(self):
        """Test 'strong' keyword maps to 4."""
        assert EmbeddingGenerator._derive_quality_rating(
            "Strong methodology with rigorous evaluation."
        ) == 4

    def test_keyword_weak(self):
        """Test 'weak' keyword maps to 2."""
        assert EmbeddingGenerator._derive_quality_rating(
            "Weak methodology with limited sample."
        ) == 2

    def test_default_returns_three(self):
        """Test neutral prose defaults to 3."""
        assert EmbeddingGenerator._derive_quality_rating(
            "The methodology is adequate for the research questions posed."
        ) == 3

    def test_boundary_1_slash_5(self):
        """Test minimum rating 1/5."""
        assert EmbeddingGenerator._derive_quality_rating("Very poor. 1/5.") == 1

    def test_boundary_5_slash_5(self):
        """Test maximum rating 5/5."""
        assert EmbeddingGenerator._derive_quality_rating("Outstanding. 5/5.") == 5


class TestValidateQualityMin:
    """Tests for validate_quality_min validator."""

    def test_valid_values(self):
        for i in range(1, 6):
            assert validate_quality_min(i) == i

    def test_zero_rejected(self):
        with pytest.raises(ValidationError):
            validate_quality_min(0)

    def test_six_rejected(self):
        with pytest.raises(ValidationError):
            validate_quality_min(6)

    def test_negative_rejected(self):
        with pytest.raises(ValidationError):
            validate_quality_min(-1)


class TestFormatExtraction:
    """Tests for quality-related fields in _format_extraction."""

    def test_format_groups_dimensions_by_pass(self):
        from src.mcp.adapters import LitrisAdapter

        adapter = LitrisAdapter.__new__(LitrisAdapter)
        extraction = {
            "q01_research_question": "What is X?",
            "q02_thesis": "X is Y.",
            "q21_summary": "Summary of quality.",
            "q22_contribution": "Novel contribution.",
        }
        formatted = adapter._format_extraction(extraction)
        assert "pass_1_research_core" in formatted
        assert formatted["pass_1_research_core"]["q01_research_question"] == "What is X?"
        assert formatted["pass_1_research_core"]["q02_thesis"] == "X is Y."

    def test_format_handles_missing_fields(self):
        from src.mcp.adapters import LitrisAdapter

        adapter = LitrisAdapter.__new__(LitrisAdapter)
        extraction = {
            "q02_thesis": "Test thesis.",
        }
        formatted = adapter._format_extraction(extraction)
        assert formatted["pass_1_research_core"]["q01_research_question"] is None
        assert formatted["pass_1_research_core"]["q02_thesis"] == "Test thesis."


class TestPromptVersion:
    """Test that prompt version was bumped."""

    def test_prompt_version_1_4(self):
        from src.analysis.prompts import EXTRACTION_PROMPT_VERSION

        assert EXTRACTION_PROMPT_VERSION == "1.4.0"

    def test_quality_in_system_prompt(self):
        from src.analysis.prompts import EXTRACTION_SYSTEM_PROMPT

        assert "1-5" in EXTRACTION_SYSTEM_PROMPT
        assert "quality" in EXTRACTION_SYSTEM_PROMPT.lower()

    def test_quality_in_all_templates(self):
        from src.analysis.prompts import DOCUMENT_TYPE_PROMPTS

        for key, template in DOCUMENT_TYPE_PROMPTS.items():
            assert "quality_rating" in template, f"quality_rating missing from {key} template"
            assert "quality_explanation" in template, f"quality_explanation missing from {key} template"
