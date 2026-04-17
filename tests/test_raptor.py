"""Tests for RAPTOR hierarchical summarization."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.analysis.raptor import (
    RaptorSummaries,
    _format_analysis_for_prompt,
    generate_raptor_batch,
    generate_raptor_summaries,
)
from src.analysis.schemas import SemanticAnalysis
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
def sample_analysis():
    """Create a sample SemanticAnalysis for testing."""
    return SemanticAnalysis(
        paper_id="test_001",
        prompt_version="2.0.0",
        extraction_model="test-model",
        extracted_at="2026-01-01T00:00:00Z",
        q01_research_question="How do communities form online?",
        q02_thesis="Social media networks exhibit small-world properties.",
        q03_key_claims="Communities form around shared interests. Bridge nodes connect disparate communities.",
        q04_evidence="Empirical analysis of 10,000 Twitter users shows clustering around interest-based communities.",
        q05_limitations="Single platform. English-only data.",
        q06_paradigm="Quantitative network analysis.",
        q07_methods="Graph clustering and modularity optimization using Twitter API data.",
        q08_data="Twitter API; 10,000 users sampled over 6 months.",
        q10_framework="Network theory.",
        q17_field="Computational social science.",
        q19_implications="Small-world properties enable rapid information diffusion across social networks.",
        q20_future_work="Cross-platform analysis.",
        q22_contribution="Novel graph-based method for community detection.",
    )


@pytest.fixture
def raptor_summaries():
    """Create sample RAPTOR summaries."""
    return RaptorSummaries(
        paper_id="test_001",
        paper_overview="A quantitative study of community detection...",
        core_contribution="Novel graph-based community detection reveals small-world properties in social media.",
    )


class TestRaptorSummaries:
    """Tests for the RaptorSummaries dataclass."""

    def test_creates_with_all_fields(self, raptor_summaries):
        assert raptor_summaries.paper_id == "test_001"
        assert "quantitative" in raptor_summaries.paper_overview
        assert "graph-based" in raptor_summaries.core_contribution

    def test_fields_are_strings(self, raptor_summaries):
        assert isinstance(raptor_summaries.paper_overview, str)
        assert isinstance(raptor_summaries.core_contribution, str)


class TestFormatAnalysisForPrompt:
    """Tests for formatting SemanticAnalysis data into prompt template values."""

    def test_formats_all_fields(self, sample_paper, sample_analysis):
        result = _format_analysis_for_prompt(sample_paper, sample_analysis)

        assert result["title"] == "Network Analysis in Social Media"
        assert result["year"] == "2024"
        assert "Smith" in result["authors"]
        assert "small-world" in result["thesis"]
        assert "community detection" in result["contribution"]
        assert "Network theory" in result["framework"]
        assert "How do communities form" in result["research_question"]
        assert "Graph clustering" in result["methods"]
        assert "Communities form" in result["key_claims"]
        assert "information diffusion" in result["implications"]
        assert "Single platform" in result["limitations"]
        assert "Cross-platform" in result["future_work"]

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
        analysis = SemanticAnalysis(
            paper_id="empty_001",
            prompt_version="2.0.0",
            extraction_model="test-model",
            extracted_at="2026-01-01T00:00:00Z",
        )
        result = _format_analysis_for_prompt(paper, analysis)

        # title="" is cleaned to "Untitled" by PaperMetadata; author_string returns "Unknown"
        assert result["year"] == "n/a"
        assert "(not extracted)" in result["thesis"]
        assert "(not extracted)" in result["methods"]
        assert "(none)" in result["research_question"]

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
        analysis = SemanticAnalysis(
            paper_id="meth_001",
            prompt_version="2.0.0",
            extraction_model="test-model",
            extracted_at="2026-01-01T00:00:00Z",
            # All q-fields default to None
        )
        result = _format_analysis_for_prompt(paper, analysis)
        assert "(not extracted)" in result["methods"]


class TestGenerateRaptorSummaries:
    """Tests for LLM-based RAPTOR summary generation."""

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_generates_valid_summaries(self, mock_factory, sample_paper, sample_analysis):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            json.dumps(
                {
                    "paper_overview": "A concise overview of the study...",
                    "core_contribution": "Key contribution in one sentence.",
                }
            ),
            100,
            200,
        )
        mock_factory.return_value = mock_client

        result = generate_raptor_summaries(sample_paper, sample_analysis)

        assert result is not None
        assert result.paper_id == "test_001"
        assert result.paper_overview == "A concise overview of the study..."
        assert result.core_contribution == "Key contribution in one sentence."
        mock_factory.assert_called_once()

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_strips_markdown_fences(self, mock_factory, sample_paper, sample_analysis):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            '```json\n{"paper_overview": "B", "core_contribution": "C"}\n```',
            100,
            200,
        )
        mock_factory.return_value = mock_client

        result = generate_raptor_summaries(sample_paper, sample_analysis)

        assert result is not None
        assert result.paper_overview == "B"

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_returns_none_on_invalid_json(self, mock_factory, sample_paper, sample_analysis):
        mock_client = MagicMock()
        mock_client._call_api.return_value = ("not valid json", 100, 200)
        mock_factory.return_value = mock_client

        result = generate_raptor_summaries(sample_paper, sample_analysis)
        assert result is None

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_returns_none_on_missing_fields(self, mock_factory, sample_paper, sample_analysis):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            json.dumps({"paper_overview": "Only one field"}),
            100,
            200,
        )
        mock_factory.return_value = mock_client

        result = generate_raptor_summaries(sample_paper, sample_analysis)
        assert result is None

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_returns_none_on_non_dict_response(self, mock_factory, sample_paper, sample_analysis):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (json.dumps(["a list"]), 100, 200)
        mock_factory.return_value = mock_client

        result = generate_raptor_summaries(sample_paper, sample_analysis)
        assert result is None

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_returns_none_on_api_exception(self, mock_factory, sample_paper, sample_analysis):
        mock_client = MagicMock()
        mock_client._call_api.side_effect = RuntimeError("API down")
        mock_factory.return_value = mock_client

        result = generate_raptor_summaries(sample_paper, sample_analysis)
        assert result is None

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_passes_provider_and_model(self, mock_factory, sample_paper, sample_analysis):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            json.dumps(
                {
                    "paper_overview": "O",
                    "core_contribution": "C",
                }
            ),
            10,
            20,
        )
        mock_factory.return_value = mock_client

        generate_raptor_summaries(
            sample_paper,
            sample_analysis,
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
    def test_processes_matching_papers(self, mock_generate, sample_paper, sample_analysis):
        mock_generate.return_value = RaptorSummaries(
            paper_id="test_001",
            paper_overview="O",
            core_contribution="C",
        )

        result = generate_raptor_batch(
            papers=[sample_paper],
            analyses={"test_001": sample_analysis},
        )

        assert "test_001" in result
        assert result["test_001"].paper_overview == "O"
        mock_generate.assert_called_once()

    @patch("src.analysis.raptor.generate_raptor_summaries")
    def test_skips_papers_without_analysis(self, mock_generate, sample_paper):
        result = generate_raptor_batch(
            papers=[sample_paper],
            analyses={},  # No matching analysis
        )

        assert len(result) == 0
        mock_generate.assert_not_called()

    @patch("src.analysis.raptor.generate_raptor_summaries")
    def test_handles_failed_generations(self, mock_generate, sample_paper, sample_analysis):
        mock_generate.return_value = None  # Simulates failure

        result = generate_raptor_batch(
            papers=[sample_paper],
            analyses={"test_001": sample_analysis},
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
        analyses = {
            f"paper_{i}": SemanticAnalysis(
                paper_id=f"paper_{i}",
                prompt_version="2.0.0",
                extraction_model="test-model",
                extracted_at="2026-01-01T00:00:00Z",
                q02_thesis=f"Thesis {i}",
            )
            for i in range(3)
        }
        mock_generate.side_effect = [
            RaptorSummaries(paper_id="paper_0", paper_overview="O0", core_contribution="C0"),
            None,  # paper_1 fails
            RaptorSummaries(paper_id="paper_2", paper_overview="O2", core_contribution="C2"),
        ]

        result = generate_raptor_batch(papers=papers, analyses=analyses)

        assert len(result) == 2
        assert "paper_0" in result
        assert "paper_2" in result
        assert "paper_1" not in result


class TestChunkTypeIntegration:
    """Tests for RAPTOR chunk types in embedding generation."""

    def test_new_chunk_types_in_list(self):
        assert "raptor_overview" in CHUNK_TYPES
        assert "raptor_core" in CHUNK_TYPES

    @patch("src.indexing.embeddings._SentenceTransformer")
    def test_create_chunks_without_raptor(self, mock_st, sample_paper, sample_analysis):
        """Existing behavior: no RAPTOR summaries, same chunks as before."""
        mock_model = MagicMock()
        mock_st.return_value = mock_model

        gen = EmbeddingGenerator(model_name="test-model")
        chunks = gen.create_chunks(sample_paper, sample_analysis)

        chunk_types = [c.chunk_type for c in chunks]
        assert "raptor_overview" not in chunk_types
        assert "raptor_core" not in chunk_types

    @patch("src.indexing.embeddings._SentenceTransformer")
    def test_create_chunks_with_raptor(
        self, mock_st, sample_paper, sample_analysis, raptor_summaries
    ):
        """With RAPTOR summaries, 2 additional chunks are created."""
        mock_model = MagicMock()
        mock_st.return_value = mock_model

        gen = EmbeddingGenerator(model_name="test-model")
        chunks_without = gen.create_chunks(sample_paper, sample_analysis)
        chunks_with = gen.create_chunks(
            sample_paper, sample_analysis, raptor_summaries=raptor_summaries
        )

        assert len(chunks_with) == len(chunks_without) + 2

        raptor_chunks = [
            c for c in chunks_with if c.chunk_type in ("raptor_overview", "raptor_core")
        ]
        assert len(raptor_chunks) == 2

        raptor_types = {c.chunk_type for c in raptor_chunks}
        assert raptor_types == {"raptor_overview", "raptor_core"}

    @patch("src.indexing.embeddings._SentenceTransformer")
    def test_raptor_chunks_have_correct_ids(
        self, mock_st, sample_paper, sample_analysis, raptor_summaries
    ):
        mock_model = MagicMock()
        mock_st.return_value = mock_model

        gen = EmbeddingGenerator(model_name="test-model")
        chunks = gen.create_chunks(sample_paper, sample_analysis, raptor_summaries=raptor_summaries)

        raptor_chunks = {
            c.chunk_type: c for c in chunks if c.chunk_type in ("raptor_overview", "raptor_core")
        }

        assert raptor_chunks["raptor_overview"].chunk_id == "test_001_raptor_overview"
        assert raptor_chunks["raptor_core"].chunk_id == "test_001_raptor_core"

    @patch("src.indexing.embeddings._SentenceTransformer")
    def test_raptor_chunks_have_metadata(
        self, mock_st, sample_paper, sample_analysis, raptor_summaries
    ):
        mock_model = MagicMock()
        mock_st.return_value = mock_model

        gen = EmbeddingGenerator(model_name="test-model")
        chunks = gen.create_chunks(sample_paper, sample_analysis, raptor_summaries=raptor_summaries)

        for chunk in chunks:
            if chunk.chunk_type == "raptor_overview":
                assert chunk.metadata["title"] == "Network Analysis in Social Media"
                assert chunk.metadata["year"] == 2024
                break

    @patch("src.indexing.embeddings._SentenceTransformer")
    def test_raptor_chunks_with_empty_fields_skipped(self, mock_st, sample_paper, sample_analysis):
        """RAPTOR summaries with empty strings should not create chunks."""
        mock_model = MagicMock()
        mock_st.return_value = mock_model

        partial_raptor = RaptorSummaries(
            paper_id="test_001",
            paper_overview="Has content",
            core_contribution="",  # Empty
        )

        gen = EmbeddingGenerator(model_name="test-model")
        chunks = gen.create_chunks(sample_paper, sample_analysis, raptor_summaries=partial_raptor)

        raptor_types = [
            c.chunk_type for c in chunks if c.chunk_type in ("raptor_overview", "raptor_core")
        ]
        assert raptor_types == ["raptor_overview"]
