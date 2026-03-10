"""Tests for RAPTOR hierarchical summarization."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.analysis.raptor import (
    RaptorSummaries,
    _format_extraction_for_prompt,
    generate_raptor_batch,
    generate_raptor_summaries,
)
from src.analysis.schemas import (
    KeyFinding,
    Methodology,
    PaperExtraction,
)
from src.indexing.embeddings import CHUNK_TYPES, EmbeddingGenerator
from src.zotero.models import PaperMetadata


@pytest.fixture
def sample_paper():
    """Create a sample PaperMetadata for testing."""
    return PaperMetadata(
        paper_id="test_001",
        zotero_key="ABC123",
        zotero_item_id=12345,
        title="Network Analysis in Social Media",
        authors=[{"first_name": "John", "last_name": "Smith"}],
        publication_year=2024,
        item_type="journalArticle",
        abstract="This paper examines network structures.",
        date_added=datetime(2024, 1, 1),
        date_modified=datetime(2024, 1, 2),
    )


@pytest.fixture
def sample_extraction():
    """Create a sample PaperExtraction for testing."""
    return PaperExtraction(
        thesis_statement="Social media networks exhibit small-world properties.",
        contribution_summary="Novel graph-based method for community detection.",
        theoretical_framework="Network theory",
        research_questions=["How do communities form online?"],
        methodology=Methodology(
            approach="quantitative",
            design="network analysis",
            data_sources=["Twitter API"],
            analysis_methods=["graph clustering", "modularity optimization"],
            sample_size="10,000 users",
        ),
        key_findings=[
            KeyFinding(finding="Communities form around shared interests."),
            KeyFinding(finding="Bridge nodes connect disparate communities."),
        ],
        conclusions="Small-world properties enable rapid information diffusion.",
        limitations=["Single platform", "English-only data"],
        future_directions=["Cross-platform analysis"],
    )


@pytest.fixture
def raptor_summaries():
    """Create sample RAPTOR summaries."""
    return RaptorSummaries(
        paper_id="test_001",
        section_summary="This study examines social media networks...",
        paper_overview="A quantitative study of community detection...",
        core_contribution="Novel graph-based community detection reveals small-world properties in social media.",
    )


class TestRaptorSummaries:
    """Tests for the RaptorSummaries dataclass."""

    def test_creates_with_all_fields(self, raptor_summaries):
        assert raptor_summaries.paper_id == "test_001"
        assert "social media" in raptor_summaries.section_summary
        assert "quantitative" in raptor_summaries.paper_overview
        assert "graph-based" in raptor_summaries.core_contribution

    def test_fields_are_strings(self, raptor_summaries):
        assert isinstance(raptor_summaries.section_summary, str)
        assert isinstance(raptor_summaries.paper_overview, str)
        assert isinstance(raptor_summaries.core_contribution, str)


class TestFormatExtractionForPrompt:
    """Tests for formatting extraction data into prompt template values."""

    def test_formats_all_fields(self, sample_paper, sample_extraction):
        result = _format_extraction_for_prompt(sample_paper, sample_extraction)

        assert result["title"] == "Network Analysis in Social Media"
        assert result["year"] == "2024"
        assert "Smith" in result["authors"]
        assert "small-world" in result["thesis"]
        assert "community detection" in result["contribution"]
        assert "Network theory" in result["framework"]
        assert "How do communities form" in result["research_questions"]
        assert "quantitative" in result["methodology"]
        assert "Communities form" in result["findings"]
        assert "information diffusion" in result["conclusions"]
        assert "Single platform" in result["limitations"]
        assert "Cross-platform" in result["future_directions"]

    def test_handles_missing_fields(self):
        paper = PaperMetadata(
            paper_id="empty_001",
            zotero_key="ZZZ",
            zotero_item_id=99999,
            title="",
            authors=[],
            publication_year=None,
            item_type="document",
            date_added=datetime(2024, 1, 1),
            date_modified=datetime(2024, 1, 1),
        )
        extraction = PaperExtraction()
        result = _format_extraction_for_prompt(paper, extraction)

        # title="" is cleaned to "Untitled" by PaperMetadata; author_string returns "Unknown"
        assert result["year"] == "n/a"
        assert "(not extracted)" in result["thesis"]
        assert "(not extracted)" in result["methodology"]
        assert "(none)" in result["research_questions"]

    def test_handles_empty_methodology(self):
        paper = PaperMetadata(
            paper_id="meth_001",
            zotero_key="MMM",
            zotero_item_id=88888,
            title="Test",
            authors=[],
            publication_year=2024,
            item_type="journalArticle",
            date_added=datetime(2024, 1, 1),
            date_modified=datetime(2024, 1, 1),
        )
        extraction = PaperExtraction(
            methodology=Methodology(),  # All None/empty
        )
        result = _format_extraction_for_prompt(paper, extraction)
        assert "(not extracted)" in result["methodology"]


class TestGenerateRaptorSummaries:
    """Tests for LLM-based RAPTOR summary generation."""

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_generates_valid_summaries(self, mock_factory, sample_paper, sample_extraction):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            json.dumps({
                "section_summary": "Detailed narrative about the paper...",
                "paper_overview": "A concise overview of the study...",
                "core_contribution": "Key contribution in one sentence.",
            }),
            100,
            200,
        )
        mock_factory.return_value = mock_client

        result = generate_raptor_summaries(sample_paper, sample_extraction)

        assert result is not None
        assert result.paper_id == "test_001"
        assert result.section_summary == "Detailed narrative about the paper..."
        assert result.paper_overview == "A concise overview of the study..."
        assert result.core_contribution == "Key contribution in one sentence."
        mock_factory.assert_called_once()

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_strips_markdown_fences(self, mock_factory, sample_paper, sample_extraction):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            '```json\n{"section_summary": "A", "paper_overview": "B", "core_contribution": "C"}\n```',
            100,
            200,
        )
        mock_factory.return_value = mock_client

        result = generate_raptor_summaries(sample_paper, sample_extraction)

        assert result is not None
        assert result.section_summary == "A"

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_returns_none_on_invalid_json(self, mock_factory, sample_paper, sample_extraction):
        mock_client = MagicMock()
        mock_client._call_api.return_value = ("not valid json", 100, 200)
        mock_factory.return_value = mock_client

        result = generate_raptor_summaries(sample_paper, sample_extraction)
        assert result is None

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_returns_none_on_missing_fields(self, mock_factory, sample_paper, sample_extraction):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            json.dumps({"section_summary": "Only one field"}),
            100,
            200,
        )
        mock_factory.return_value = mock_client

        result = generate_raptor_summaries(sample_paper, sample_extraction)
        assert result is None

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_returns_none_on_non_dict_response(self, mock_factory, sample_paper, sample_extraction):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (json.dumps(["a list"]), 100, 200)
        mock_factory.return_value = mock_client

        result = generate_raptor_summaries(sample_paper, sample_extraction)
        assert result is None

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_returns_none_on_api_exception(self, mock_factory, sample_paper, sample_extraction):
        mock_client = MagicMock()
        mock_client._call_api.side_effect = RuntimeError("API down")
        mock_factory.return_value = mock_client

        result = generate_raptor_summaries(sample_paper, sample_extraction)
        assert result is None

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_passes_provider_and_model(self, mock_factory, sample_paper, sample_extraction):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            json.dumps({
                "section_summary": "S",
                "paper_overview": "O",
                "core_contribution": "C",
            }),
            10,
            20,
        )
        mock_factory.return_value = mock_client

        generate_raptor_summaries(
            sample_paper,
            sample_extraction,
            provider="openai",
            model="gpt-4.1-mini",
        )

        mock_factory.assert_called_once_with(
            provider="openai",
            mode="api",
            model="gpt-4.1-mini",
            max_tokens=2048,
            timeout=60,
        )


class TestGenerateRaptorBatch:
    """Tests for batch RAPTOR generation."""

    @patch("src.analysis.raptor.generate_raptor_summaries")
    def test_processes_matching_papers(self, mock_generate, sample_paper, sample_extraction):
        mock_generate.return_value = RaptorSummaries(
            paper_id="test_001",
            section_summary="S",
            paper_overview="O",
            core_contribution="C",
        )

        result = generate_raptor_batch(
            papers=[sample_paper],
            extractions={"test_001": sample_extraction},
        )

        assert "test_001" in result
        assert result["test_001"].section_summary == "S"
        mock_generate.assert_called_once()

    @patch("src.analysis.raptor.generate_raptor_summaries")
    def test_skips_papers_without_extraction(self, mock_generate, sample_paper):
        result = generate_raptor_batch(
            papers=[sample_paper],
            extractions={},  # No matching extraction
        )

        assert len(result) == 0
        mock_generate.assert_not_called()

    @patch("src.analysis.raptor.generate_raptor_summaries")
    def test_handles_failed_generations(self, mock_generate, sample_paper, sample_extraction):
        mock_generate.return_value = None  # Simulates failure

        result = generate_raptor_batch(
            papers=[sample_paper],
            extractions={"test_001": sample_extraction},
        )

        assert len(result) == 0

    @patch("src.analysis.raptor.generate_raptor_summaries")
    def test_batch_multiple_papers(self, mock_generate):
        papers = [
            PaperMetadata(
                paper_id=f"paper_{i}",
                zotero_key=f"KEY{i}",
                zotero_item_id=i,
                title=f"Paper {i}",
                authors=[],
                publication_year=2024,
                item_type="journalArticle",
                date_added=datetime(2024, 1, 1),
                date_modified=datetime(2024, 1, 1),
            )
            for i in range(3)
        ]
        extractions = {
            f"paper_{i}": PaperExtraction(thesis_statement=f"Thesis {i}")
            for i in range(3)
        }
        mock_generate.side_effect = [
            RaptorSummaries(paper_id="paper_0", section_summary="S0", paper_overview="O0", core_contribution="C0"),
            None,  # paper_1 fails
            RaptorSummaries(paper_id="paper_2", section_summary="S2", paper_overview="O2", core_contribution="C2"),
        ]

        result = generate_raptor_batch(papers=papers, extractions=extractions)

        assert len(result) == 2
        assert "paper_0" in result
        assert "paper_2" in result
        assert "paper_1" not in result


class TestChunkTypeIntegration:
    """Tests for RAPTOR chunk types in embedding generation."""

    def test_new_chunk_types_in_list(self):
        assert "section_summary" in CHUNK_TYPES
        assert "paper_overview" in CHUNK_TYPES
        assert "core_contribution" in CHUNK_TYPES

    @patch("src.indexing.embeddings._SentenceTransformer")
    def test_create_chunks_without_raptor(self, mock_st, sample_paper, sample_extraction):
        """Existing behavior: no RAPTOR summaries, same chunks as before."""
        mock_model = MagicMock()
        mock_st.return_value = mock_model

        gen = EmbeddingGenerator(model_name="test-model")
        chunks = gen.create_chunks(sample_paper, sample_extraction)

        chunk_types = [c.chunk_type for c in chunks]
        assert "section_summary" not in chunk_types
        assert "paper_overview" not in chunk_types
        assert "core_contribution" not in chunk_types

    @patch("src.indexing.embeddings._SentenceTransformer")
    def test_create_chunks_with_raptor(self, mock_st, sample_paper, sample_extraction, raptor_summaries):
        """With RAPTOR summaries, 3 additional chunks are created."""
        mock_model = MagicMock()
        mock_st.return_value = mock_model

        gen = EmbeddingGenerator(model_name="test-model")
        chunks_without = gen.create_chunks(sample_paper, sample_extraction)
        chunks_with = gen.create_chunks(sample_paper, sample_extraction, raptor_summaries=raptor_summaries)

        assert len(chunks_with) == len(chunks_without) + 3

        raptor_chunks = [c for c in chunks_with if c.chunk_type in ("section_summary", "paper_overview", "core_contribution")]
        assert len(raptor_chunks) == 3

        raptor_types = {c.chunk_type for c in raptor_chunks}
        assert raptor_types == {"section_summary", "paper_overview", "core_contribution"}

    @patch("src.indexing.embeddings._SentenceTransformer")
    def test_raptor_chunks_have_correct_ids(self, mock_st, sample_paper, sample_extraction, raptor_summaries):
        mock_model = MagicMock()
        mock_st.return_value = mock_model

        gen = EmbeddingGenerator(model_name="test-model")
        chunks = gen.create_chunks(sample_paper, sample_extraction, raptor_summaries=raptor_summaries)

        raptor_chunks = {c.chunk_type: c for c in chunks if c.chunk_type in ("section_summary", "paper_overview", "core_contribution")}

        assert raptor_chunks["section_summary"].chunk_id == "test_001_section_summary"
        assert raptor_chunks["paper_overview"].chunk_id == "test_001_paper_overview"
        assert raptor_chunks["core_contribution"].chunk_id == "test_001_core_contribution"

    @patch("src.indexing.embeddings._SentenceTransformer")
    def test_raptor_chunks_have_metadata(self, mock_st, sample_paper, sample_extraction, raptor_summaries):
        mock_model = MagicMock()
        mock_st.return_value = mock_model

        gen = EmbeddingGenerator(model_name="test-model")
        chunks = gen.create_chunks(sample_paper, sample_extraction, raptor_summaries=raptor_summaries)

        for chunk in chunks:
            if chunk.chunk_type == "section_summary":
                assert chunk.metadata["title"] == "Network Analysis in Social Media"
                assert chunk.metadata["year"] == 2024
                break

    @patch("src.indexing.embeddings._SentenceTransformer")
    def test_raptor_chunks_with_empty_fields_skipped(self, mock_st, sample_paper, sample_extraction):
        """RAPTOR summaries with empty strings should not create chunks."""
        mock_model = MagicMock()
        mock_st.return_value = mock_model

        partial_raptor = RaptorSummaries(
            paper_id="test_001",
            section_summary="Has content",
            paper_overview="",  # Empty
            core_contribution="",  # Empty
        )

        gen = EmbeddingGenerator(model_name="test-model")
        chunks = gen.create_chunks(sample_paper, sample_extraction, raptor_summaries=partial_raptor)

        raptor_types = [c.chunk_type for c in chunks if c.chunk_type in ("section_summary", "paper_overview", "core_contribution")]
        assert raptor_types == ["section_summary"]
