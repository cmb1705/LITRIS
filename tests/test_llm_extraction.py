"""Tests for LLM extraction schemas and prompts."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.analysis.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_prompt,
    build_validation_prompt,
)
from src.analysis.schemas import (
    EvidenceType,
    ExtractionResult,
    KeyClaim,
    KeyFinding,
    Methodology,
    PaperExtraction,
    SignificanceLevel,
    SupportType,
)


class TestSchemas:
    """Tests for extraction schemas."""

    def test_methodology_default(self):
        """Test Methodology with defaults."""
        method = Methodology()
        assert method.approach is None
        assert method.data_sources == []

    def test_methodology_full(self):
        """Test Methodology with all fields."""
        method = Methodology(
            approach="quantitative",
            design="experimental",
            data_sources=["survey", "interviews"],
            analysis_methods=["regression", "thematic analysis"],
            sample_size="500 participants",
            time_period="2020-2023",
        )
        assert method.approach == "quantitative"
        assert len(method.data_sources) == 2

    def test_key_finding_default(self):
        """Test KeyFinding with defaults."""
        finding = KeyFinding(finding="Test finding")
        assert finding.evidence_type == EvidenceType.EMPIRICAL
        assert finding.significance == SignificanceLevel.MEDIUM

    def test_key_finding_full(self):
        """Test KeyFinding with all fields."""
        finding = KeyFinding(
            finding="Significant correlation found",
            evidence_type=EvidenceType.QUANTITATIVE,
            significance=SignificanceLevel.HIGH,
            page_reference="p. 15",
        )
        assert finding.evidence_type == EvidenceType.QUANTITATIVE
        assert finding.page_reference == "p. 15"

    def test_key_claim_default(self):
        """Test KeyClaim with defaults."""
        claim = KeyClaim(claim="Test claim")
        assert claim.support_type == SupportType.LOGIC
        assert claim.strength == SignificanceLevel.MEDIUM

    def test_paper_extraction_minimal(self):
        """Test PaperExtraction with minimal fields."""
        extraction = PaperExtraction()
        assert extraction.thesis_statement is None
        assert extraction.research_questions == []
        assert extraction.extraction_confidence == 0.5

    def test_paper_extraction_full(self):
        """Test PaperExtraction with all fields."""
        extraction = PaperExtraction(
            thesis_statement="This paper argues that X leads to Y",
            research_questions=["RQ1: Does X affect Y?", "RQ2: How?"],
            theoretical_framework="Social network theory",
            methodology=Methodology(approach="mixed"),
            key_findings=[KeyFinding(finding="X correlates with Y")],
            key_claims=[KeyClaim(claim="X is important")],
            conclusions="The findings support the hypothesis",
            limitations=["Small sample size"],
            future_directions=["Longitudinal study"],
            contribution_summary="Novel method for X",
            discipline_tags=["network science", "bibliometrics"],
            extraction_confidence=0.85,
        )

        assert extraction.thesis_statement is not None
        assert len(extraction.research_questions) == 2
        assert extraction.extraction_confidence == 0.85

    def test_paper_extraction_to_index_dict(self):
        """Test conversion to index dictionary."""
        extraction = PaperExtraction(
            thesis_statement="Test thesis",
            discipline_tags=["tag1", "tag2"],
        )
        result = extraction.to_index_dict()

        assert result["thesis_statement"] == "Test thesis"
        assert result["discipline_tags"] == ["tag1", "tag2"]
        assert "extraction_confidence" in result

    def test_extraction_result_success(self):
        """Test ExtractionResult for successful extraction."""
        extraction = PaperExtraction(thesis_statement="Test")
        result = ExtractionResult(
            paper_id="test123",
            success=True,
            extraction=extraction,
            duration_seconds=5.5,
            model_used="claude-opus-4-5-20251101",
        )

        assert result.success
        assert result.extraction is not None
        assert result.error is None

    def test_extraction_result_failure(self):
        """Test ExtractionResult for failed extraction."""
        result = ExtractionResult(
            paper_id="test123",
            success=False,
            error="PDF extraction failed",
            duration_seconds=0.5,
        )

        assert not result.success
        assert result.extraction is None
        assert "PDF" in result.error


class TestPrompts:
    """Tests for prompt building."""

    def test_system_prompt_content(self):
        """Test system prompt contains key instructions."""
        assert "academic" in EXTRACTION_SYSTEM_PROMPT.lower()
        assert "extract" in EXTRACTION_SYSTEM_PROMPT.lower()
        assert "structured" in EXTRACTION_SYSTEM_PROMPT.lower()

    def test_build_extraction_prompt(self):
        """Test extraction prompt building."""
        prompt = build_extraction_prompt(
            title="Test Paper",
            authors="John Doe",
            year=2023,
            item_type="journalArticle",
            text="This is the paper content.",
        )

        assert "Test Paper" in prompt
        assert "John Doe" in prompt
        assert "2023" in prompt
        assert "journalArticle" in prompt
        assert "paper content" in prompt

    def test_build_extraction_prompt_missing_year(self):
        """Test extraction prompt with missing year."""
        prompt = build_extraction_prompt(
            title="Test Paper",
            authors="John Doe",
            year=None,
            item_type="journalArticle",
            text="Content",
        )

        assert "Unknown" in prompt

    def test_build_validation_prompt(self):
        """Test validation prompt building."""
        extraction_json = json.dumps({"thesis_statement": "Test"})
        prompt = build_validation_prompt(
            text_excerpt="Sample text excerpt",
            extraction_json=extraction_json,
        )

        assert "Sample text excerpt" in prompt
        assert "thesis_statement" in prompt


class TestEnumValues:
    """Tests for enum values."""

    def test_evidence_type_values(self):
        """Test EvidenceType enum values."""
        assert EvidenceType.EMPIRICAL.value == "empirical"
        assert EvidenceType.THEORETICAL.value == "theoretical"
        assert EvidenceType.MIXED.value == "mixed"

    def test_significance_level_values(self):
        """Test SignificanceLevel enum values."""
        assert SignificanceLevel.HIGH.value == "high"
        assert SignificanceLevel.MEDIUM.value == "medium"
        assert SignificanceLevel.LOW.value == "low"

    def test_support_type_values(self):
        """Test SupportType enum values."""
        assert SupportType.DATA.value == "data"
        assert SupportType.CITATION.value == "citation"
        assert SupportType.LOGIC.value == "logic"


class TestExtractionStats:
    """Tests for ExtractionStats."""

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        from src.analysis.section_extractor import ExtractionStats

        stats = ExtractionStats(total=10, successful=7, failed=3)
        assert stats.success_rate == 0.7

    def test_success_rate_zero_total(self):
        """Test success rate with zero total."""
        from src.analysis.section_extractor import ExtractionStats

        stats = ExtractionStats(total=0, successful=0, failed=0)
        assert stats.success_rate == 0.0


class TestLLMClientMocked:
    """Tests for LLMClient with mocked API."""

    @pytest.fixture
    def mock_response_json(self):
        """Sample valid JSON response."""
        return json.dumps({
            "thesis_statement": "Test thesis",
            "research_questions": ["RQ1"],
            "theoretical_framework": "Test framework",
            "methodology": {
                "approach": "quantitative",
                "design": "survey",
                "data_sources": ["database"],
                "analysis_methods": ["regression"],
            },
            "key_findings": [
                {
                    "finding": "Finding 1",
                    "evidence_type": "empirical",
                    "significance": "high",
                }
            ],
            "key_claims": [],
            "conclusions": "Test conclusions",
            "limitations": ["Limitation 1"],
            "future_directions": ["Future 1"],
            "contribution_summary": "Test contribution",
            "discipline_tags": ["test"],
            "extraction_confidence": 0.85,
            "extraction_notes": None,
        })

    def test_parse_response(self, mock_response_json):
        """Test JSON response parsing."""
        from src.analysis.llm_client import LLMClient

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            client = LLMClient(mode="api")
            extraction = client._parse_response(mock_response_json)

        assert extraction.thesis_statement == "Test thesis"
        assert len(extraction.research_questions) == 1
        assert extraction.methodology.approach == "quantitative"
        assert len(extraction.key_findings) == 1
        assert extraction.extraction_confidence == 0.85

    def test_parse_response_with_markdown(self, mock_response_json):
        """Test parsing response with markdown code block."""
        from src.analysis.llm_client import LLMClient

        wrapped = f"```json\n{mock_response_json}\n```"

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            client = LLMClient(mode="api")
            extraction = client._parse_response(wrapped)

        assert extraction.thesis_statement == "Test thesis"

    def test_estimate_cost(self):
        """Test cost estimation."""
        from src.analysis.llm_client import LLMClient

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            client = LLMClient(mode="api")
            cost = client.estimate_cost(100000)

        assert cost > 0
        assert isinstance(cost, float)
