"""Tests for quality_rating field in extraction schema and search filtering."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from src.analysis.schemas import PaperExtraction
from src.mcp.validators import ValidationError, validate_quality_min


class TestQualityRatingSchema:
    """Tests for quality_rating and quality_explanation in PaperExtraction."""

    def test_valid_quality_rating(self):
        extraction = PaperExtraction(quality_rating=4, quality_explanation="Strong methodology")
        assert extraction.quality_rating == 4
        assert extraction.quality_explanation == "Strong methodology"

    def test_quality_rating_none_default(self):
        extraction = PaperExtraction()
        assert extraction.quality_rating is None
        assert extraction.quality_explanation is None

    def test_quality_rating_min_boundary(self):
        extraction = PaperExtraction(quality_rating=1)
        assert extraction.quality_rating == 1

    def test_quality_rating_max_boundary(self):
        extraction = PaperExtraction(quality_rating=5)
        assert extraction.quality_rating == 5

    def test_quality_rating_zero_rejected(self):
        with pytest.raises(PydanticValidationError):
            PaperExtraction(quality_rating=0)

    def test_quality_rating_six_rejected(self):
        with pytest.raises(PydanticValidationError):
            PaperExtraction(quality_rating=6)

    def test_quality_rating_negative_rejected(self):
        with pytest.raises(PydanticValidationError):
            PaperExtraction(quality_rating=-1)

    def test_to_index_dict_includes_quality_fields(self):
        extraction = PaperExtraction(
            quality_rating=3,
            quality_explanation="Adequate but limited sample",
        )
        d = extraction.to_index_dict()
        assert d["quality_rating"] == 3
        assert d["quality_explanation"] == "Adequate but limited sample"

    def test_to_index_dict_quality_none(self):
        extraction = PaperExtraction()
        d = extraction.to_index_dict()
        assert d["quality_rating"] is None
        assert d["quality_explanation"] is None

    def test_backward_compat_existing_extraction(self):
        """Existing extractions without quality fields should load fine."""
        data = {
            "thesis_statement": "Test thesis",
            "extraction_confidence": 0.8,
        }
        extraction = PaperExtraction(**data)
        assert extraction.quality_rating is None
        assert extraction.thesis_statement == "Test thesis"


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
    """Tests for quality fields in _format_extraction."""

    def test_format_includes_quality_fields(self):
        from src.mcp.adapters import LitrisAdapter

        adapter = LitrisAdapter.__new__(LitrisAdapter)
        extraction = {
            "thesis_statement": "Test",
            "quality_rating": 4,
            "quality_explanation": "Strong evidence",
            "extraction_confidence": 0.9,
        }
        formatted = adapter._format_extraction(extraction)
        assert formatted["quality_rating"] == 4
        assert formatted["quality_explanation"] == "Strong evidence"

    def test_format_quality_none_when_missing(self):
        from src.mcp.adapters import LitrisAdapter

        adapter = LitrisAdapter.__new__(LitrisAdapter)
        extraction = {
            "thesis_statement": "Test",
            "extraction_confidence": 0.9,
        }
        formatted = adapter._format_extraction(extraction)
        assert formatted["quality_rating"] is None
        assert formatted["quality_explanation"] is None


class TestPromptVersion:
    """Test that prompt version was bumped."""

    def test_prompt_version_1_3(self):
        from src.analysis.prompts import EXTRACTION_PROMPT_VERSION

        assert EXTRACTION_PROMPT_VERSION == "1.3.0"

    def test_quality_in_system_prompt(self):
        from src.analysis.prompts import EXTRACTION_SYSTEM_PROMPT

        assert "1-5" in EXTRACTION_SYSTEM_PROMPT
        assert "quality" in EXTRACTION_SYSTEM_PROMPT.lower()

    def test_quality_in_all_templates(self):
        from src.analysis.prompts import DOCUMENT_TYPE_PROMPTS

        for key, template in DOCUMENT_TYPE_PROMPTS.items():
            assert "quality_rating" in template, f"quality_rating missing from {key} template"
            assert "quality_explanation" in template, f"quality_explanation missing from {key} template"
