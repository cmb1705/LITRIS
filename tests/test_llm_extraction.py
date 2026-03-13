"""Tests for LLM extraction schemas and prompts."""

import json
from unittest.mock import patch

import pytest

from src.analysis.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_prompt,
    build_validation_prompt,
)
from src.analysis.schemas import (
    ExtractionResult,
    SemanticAnalysis,
)

# --- Helper ---

def _make_analysis(**overrides) -> SemanticAnalysis:
    """Create a SemanticAnalysis with required fields and optional overrides."""
    defaults = {
        "paper_id": "test_id",
        "prompt_version": "2.0.0",
        "extraction_model": "test-model",
        "extracted_at": "2026-01-01T00:00:00Z",
    }
    defaults.update(overrides)
    return SemanticAnalysis(**defaults)


class TestSchemas:
    """Tests for SemanticAnalysis schema."""

    def test_semantic_analysis_minimal(self):
        """Test SemanticAnalysis with only required fields."""
        analysis = _make_analysis()
        assert analysis.paper_id == "test_id"
        assert analysis.prompt_version == "2.0.0"
        assert analysis.extraction_model == "test-model"
        assert analysis.q01_research_question is None
        assert analysis.q02_thesis is None
        assert analysis.dimension_coverage == 0.0
        assert analysis.coverage_flags == []

    def test_semantic_analysis_with_dimensions(self):
        """Test SemanticAnalysis with populated q-fields."""
        analysis = _make_analysis(
            q01_research_question="Does X affect Y?",
            q02_thesis="This paper argues that X leads to Y",
            q03_key_claims="X is correlated with Y at p<0.01",
            q04_evidence="Regression analysis on 500-participant survey",
            q05_limitations="Small sample size, single institution",
            q07_methods="Quantitative survey with regression analysis",
            q17_field="network science, bibliometrics",
            q22_contribution="Novel method for X",
            dimension_coverage=0.85,
        )

        assert analysis.q01_research_question == "Does X affect Y?"
        assert analysis.q02_thesis is not None
        assert analysis.q07_methods is not None
        assert analysis.dimension_coverage == 0.85

    def test_semantic_analysis_to_index_dict(self):
        """Test conversion to index dictionary."""
        analysis = _make_analysis(
            q02_thesis="Test thesis",
            q17_field="tag1, tag2",
        )
        result = analysis.to_index_dict()

        assert result["q02_thesis"] == "Test thesis"
        assert result["q17_field"] == "tag1, tag2"
        assert result["paper_id"] == "test_id"
        assert "dimension_coverage" in result

    def test_semantic_analysis_non_none_dimensions(self):
        """Test non_none_dimensions returns only populated fields."""
        analysis = _make_analysis(
            q01_research_question="RQ1",
            q02_thesis="Thesis",
        )
        non_none = analysis.non_none_dimensions()

        assert "q01_research_question" in non_none
        assert "q02_thesis" in non_none
        assert "q03_key_claims" not in non_none
        assert len(non_none) == 2

    def test_semantic_analysis_dimension_fields(self):
        """Test that DIMENSION_FIELDS class var lists all 40 q-fields."""
        assert len(SemanticAnalysis.DIMENSION_FIELDS) == 40
        assert SemanticAnalysis.DIMENSION_FIELDS[0] == "q01_research_question"
        assert SemanticAnalysis.DIMENSION_FIELDS[-1] == "q40_policy_recommendations"

    def test_semantic_analysis_core_fields(self):
        """Test CORE_FIELDS contains the essential dimensions."""
        assert "q01_research_question" in SemanticAnalysis.CORE_FIELDS
        assert "q02_thesis" in SemanticAnalysis.CORE_FIELDS
        assert len(SemanticAnalysis.CORE_FIELDS) == 5

    def test_semantic_analysis_dimension_groups(self):
        """Test DIMENSION_GROUPS covers all 40 dimensions."""
        all_grouped = []
        for fields in SemanticAnalysis.DIMENSION_GROUPS.values():
            all_grouped.extend(fields)
        assert set(all_grouped) == set(SemanticAnalysis.DIMENSION_FIELDS)

    def test_extraction_result_success(self):
        """Test ExtractionResult for successful extraction."""
        extraction = _make_analysis(paper_id="test123")
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
        extraction_json = json.dumps({"q02_thesis": "Test"})
        prompt = build_validation_prompt(
            text_excerpt="Sample text excerpt",
            extraction_json=extraction_json,
        )

        assert "Sample text excerpt" in prompt
        assert "q02_thesis" in prompt


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


class TestSectionExtractorConfig:
    """Tests for SectionExtractor configuration wiring."""

    def test_section_extractor_passes_ocr_config(self, tmp_path, monkeypatch):
        """Ensure OCR settings are forwarded to PDFExtractor."""
        from src.analysis import section_extractor as se_module
        captured = {}

        class DummyPDFExtractor:
            def __init__(self, cache_dir=None, enable_ocr=False, ocr_config=None):
                captured["cache_dir"] = cache_dir
                captured["enable_ocr"] = enable_ocr
                captured["ocr_config"] = ocr_config

        class DummyLLMClient:
            def __init__(self, mode=None, model=None, max_tokens=None, **kwargs):
                pass

            @property
            def model(self):
                return "test-model"

        def dummy_create_llm_client(**kwargs):
            return DummyLLMClient()

        monkeypatch.setattr(se_module, "PDFExtractor", DummyPDFExtractor)
        monkeypatch.setattr(se_module, "create_llm_client", dummy_create_llm_client)

        se_module.SectionExtractor(
            cache_dir=tmp_path,
            ocr_enabled=True,
            ocr_config={"lang": "deu", "dpi": 400},
        )

        assert captured["enable_ocr"] is True
        assert captured["ocr_config"] == {"lang": "deu", "dpi": 400}


class TestLLMClientMocked:
    """Tests for LLMClient with mocked API."""

    @pytest.fixture
    def mock_response_json(self):
        """Sample valid JSON response for PaperExtraction (legacy format).

        Note: AnthropicLLMClient._parse_response still returns PaperExtraction.
        This fixture provides the old-format JSON it expects.
        """
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
        """Test JSON response parsing returns PaperExtraction."""
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
