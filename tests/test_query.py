"""Tests for search engine and result formatting."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.indexing.structured_store import StructuredStore
from src.query.retrieval import (
    OutputFormat,
    format_brief,
    format_json,
    format_markdown,
    format_paper_detail,
    format_results,
    format_summary,
    save_results,
)
from src.query.search import EnrichedResult, SearchEngine


class TestEnrichedResult:
    """Tests for EnrichedResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create sample enriched result."""
        return EnrichedResult(
            paper_id="paper_001",
            title="Test Paper Title",
            authors="John Doe, Jane Smith",
            year=2024,
            collections=["Research", "ML"],
            item_type="journalArticle",
            chunk_type="abstract",
            matched_text="This is the matched text from the paper.",
            score=0.85,
            paper_data={"doi": "10.1234/test"},
            extraction_data={"thesis_statement": "Main thesis"},
        )

    def test_result_creation(self, sample_result):
        """Test result creation."""
        assert sample_result.paper_id == "paper_001"
        assert sample_result.year == 2024
        assert sample_result.score == 0.85

    def test_result_to_dict(self, sample_result):
        """Test conversion to dictionary."""
        d = sample_result.to_dict()
        assert d["paper_id"] == "paper_001"
        assert d["title"] == "Test Paper Title"
        assert d["score"] == 0.85
        assert "paper" in d
        assert "extraction" in d


class TestStructuredStore:
    """Tests for StructuredStore class."""

    @pytest.fixture
    def store(self, temp_index_dir):
        """Create structured store in temp directory."""
        return StructuredStore(temp_index_dir)

    @pytest.fixture
    def sample_papers(self):
        """Create sample paper data."""
        return {
            "paper_001": {
                "paper_id": "paper_001",
                "title": "Paper One",
                "author_string": "John Doe",
                "publication_year": 2024,
                "item_type": "journalArticle",
                "collections": ["ML"],
                "date_added": "2024-01-01",
            },
            "paper_002": {
                "paper_id": "paper_002",
                "title": "Paper Two",
                "author_string": "Jane Smith",
                "publication_year": 2023,
                "item_type": "conferencePaper",
                "collections": ["NLP"],
                "date_added": "2024-01-02",
            },
        }

    @pytest.fixture
    def sample_extractions(self):
        """Create sample extraction data."""
        return {
            "paper_001": {
                "paper_id": "paper_001",
                "extraction": {
                    "thesis_statement": "Paper one thesis",
                    "discipline_tags": ["Machine Learning"],
                },
            },
            "paper_002": {
                "paper_id": "paper_002",
                "extraction": {
                    "thesis_statement": "Paper two thesis",
                    "discipline_tags": ["NLP", "Deep Learning"],
                },
            },
        }

    def test_save_and_load_papers(self, store, sample_papers):
        """Test saving and loading papers."""
        store.save_papers(sample_papers)
        loaded = store.load_papers()

        assert len(loaded) == 2
        assert "paper_001" in loaded
        assert loaded["paper_001"]["title"] == "Paper One"

    def test_save_and_load_extractions(self, store, sample_extractions):
        """Test saving and loading extractions."""
        store.save_extractions(sample_extractions)
        loaded = store.load_extractions()

        assert len(loaded) == 2
        assert "paper_001" in loaded

    def test_get_paper(self, store, sample_papers):
        """Test getting single paper."""
        store.save_papers(sample_papers)

        paper = store.get_paper("paper_001")
        assert paper is not None
        assert paper["title"] == "Paper One"

        missing = store.get_paper("nonexistent")
        assert missing is None

    def test_get_extraction(self, store, sample_extractions):
        """Test getting single extraction."""
        store.save_extractions(sample_extractions)

        extraction = store.get_extraction("paper_001")
        assert extraction is not None

        missing = store.get_extraction("nonexistent")
        assert missing is None

    def test_get_paper_with_extraction(self, store, sample_papers, sample_extractions):
        """Test getting combined paper and extraction."""
        store.save_papers(sample_papers)
        store.save_extractions(sample_extractions)

        combined = store.get_paper_with_extraction("paper_001")
        assert combined is not None
        assert "paper" in combined
        assert "extraction" in combined

    def test_search_papers_by_title(self, store, sample_papers):
        """Test searching papers by title."""
        store.save_papers(sample_papers)

        results = store.search_papers(title_contains="One")
        assert len(results) == 1
        assert results[0]["paper_id"] == "paper_001"

    def test_search_papers_by_author(self, store, sample_papers):
        """Test searching papers by author."""
        store.save_papers(sample_papers)

        results = store.search_papers(author_contains="Jane")
        assert len(results) == 1
        assert results[0]["paper_id"] == "paper_002"

    def test_search_papers_by_year(self, store, sample_papers):
        """Test searching papers by year range."""
        store.save_papers(sample_papers)

        results = store.search_papers(year_min=2024)
        assert len(results) == 1
        assert results[0]["publication_year"] == 2024

        results = store.search_papers(year_max=2023)
        assert len(results) == 1
        assert results[0]["publication_year"] == 2023

    def test_search_papers_by_collection(self, store, sample_papers):
        """Test searching papers by collection."""
        store.save_papers(sample_papers)

        results = store.search_papers(collection="ML")
        assert len(results) == 1
        assert results[0]["paper_id"] == "paper_001"

    def test_search_papers_by_item_type(self, store, sample_papers):
        """Test searching papers by item type."""
        store.save_papers(sample_papers)

        results = store.search_papers(item_type="journalArticle")
        assert len(results) == 1

    def test_generate_summary(self, store, sample_papers, sample_extractions):
        """Test summary generation."""
        store.save_papers(sample_papers)
        store.save_extractions(sample_extractions)

        summary = store.generate_summary()
        assert summary["total_papers"] == 2
        assert summary["total_extractions"] == 2
        assert "papers_by_type" in summary
        assert "papers_by_year" in summary

    def test_get_paper_ids(self, store, sample_papers):
        """Test getting all paper IDs."""
        store.save_papers(sample_papers)
        ids = store.get_paper_ids()
        assert ids == {"paper_001", "paper_002"}

    def test_get_extracted_paper_ids(self, store, sample_extractions):
        """Test getting extracted paper IDs."""
        store.save_extractions(sample_extractions)
        ids = store.get_extracted_paper_ids()
        assert ids == {"paper_001", "paper_002"}

    def test_get_missing_extractions(self, store, sample_papers):
        """Test finding papers without extractions."""
        store.save_papers(sample_papers)
        store.save_extractions({"paper_001": {"paper_id": "paper_001"}})

        missing = store.get_missing_extractions()
        assert "paper_002" in missing
        assert "paper_001" not in missing

    def test_clear_cache(self, store, sample_papers):
        """Test cache clearing."""
        store.save_papers(sample_papers)
        _ = store.load_papers()  # Load to populate cache

        store.clear_cache()
        assert store._papers_cache is None
        assert store._extractions_cache is None


class TestResultFormatting:
    """Tests for result formatting functions."""

    @pytest.fixture
    def sample_results(self):
        """Create sample search results."""
        return [
            EnrichedResult(
                paper_id="p1",
                title="First Paper",
                authors="Author One",
                year=2024,
                collections=["ML"],
                item_type="journalArticle",
                chunk_type="abstract",
                matched_text="This is the matched abstract text.",
                score=0.92,
            ),
            EnrichedResult(
                paper_id="p2",
                title="Second Paper",
                authors="Author Two",
                year=2023,
                collections=["NLP"],
                item_type="conferencePaper",
                chunk_type="thesis",
                matched_text="This is the matched thesis statement.",
                score=0.85,
            ),
        ]

    def test_format_json(self, sample_results):
        """Test JSON formatting."""
        output = format_json(sample_results, "test query")
        data = json.loads(output)

        assert data["query"] == "test query"
        assert data["result_count"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["rank"] == 1

    def test_format_markdown(self, sample_results):
        """Test Markdown formatting."""
        output = format_markdown(sample_results, "test query")

        assert "# Literature Search Report" in output
        assert "test query" in output
        assert "First Paper" in output
        assert "Second Paper" in output
        assert "**Score:**" in output

    def test_format_brief(self, sample_results):
        """Test brief text formatting."""
        output = format_brief(sample_results, "test query")

        assert "Query: test query" in output
        assert "Found 2 results" in output
        assert "First Paper" in output
        assert "92.0%" in output  # Score percentage

    def test_format_results_dispatcher(self, sample_results):
        """Test format_results dispatches correctly."""
        json_out = format_results(sample_results, "query", "json")
        assert json_out.startswith("{")

        md_out = format_results(sample_results, "query", "markdown")
        assert md_out.startswith("#")

        brief_out = format_results(sample_results, "query", "brief")
        assert brief_out.startswith("Query:")

    def test_format_with_extraction(self, sample_results):
        """Test formatting with extraction data."""
        sample_results[0].extraction_data = {
            "extraction": {
                "thesis_statement": "The main thesis",
                "contribution_summary": "Novel contribution",
                "key_findings": [{"finding": "Key finding 1"}],
            }
        }

        output = format_markdown(sample_results, "query", include_extraction=True)
        assert "### Thesis" in output or "The main thesis" in output

    def test_save_results(self, sample_results, tmp_path):
        """Test saving results to file."""
        output_dir = tmp_path / "results"

        filepath = save_results(
            sample_results,
            "test query",
            output_dir,
            "json",
        )

        assert filepath.exists()
        assert filepath.suffix == ".json"
        assert (output_dir / "latest.json").exists()


class TestFormatPaperDetail:
    """Tests for paper detail formatting."""

    def test_format_basic_paper(self):
        """Test formatting basic paper without extraction."""
        paper = {
            "title": "Test Paper",
            "authors": [{"full_name": "John Doe"}],
            "publication_year": 2024,
            "item_type": "journalArticle",
            "paper_id": "p1",
        }

        output = format_paper_detail(paper)
        assert "# Test Paper" in output
        assert "**Authors:**" in output
        assert "**Year:** 2024" in output

    def test_format_paper_with_abstract(self):
        """Test formatting paper with abstract."""
        paper = {
            "title": "Test Paper",
            "abstract": "This is the abstract.",
            "paper_id": "p1",
        }

        output = format_paper_detail(paper)
        assert "## Abstract" in output
        assert "This is the abstract." in output

    def test_format_paper_with_extraction(self):
        """Test formatting paper with extraction data."""
        paper = {"title": "Test Paper", "paper_id": "p1"}
        extraction = {
            "extraction": {
                "thesis_statement": "Main thesis",
                "research_questions": ["RQ1", "RQ2"],
                "conclusions": "Final conclusions",
                "limitations": ["Limit 1"],
                "future_directions": ["Direction 1"],
            }
        }

        output = format_paper_detail(paper, extraction)
        assert "## Thesis Statement" in output
        assert "Main thesis" in output
        assert "## Research Questions" in output
        assert "## Conclusions" in output


class TestFormatSummary:
    """Tests for summary formatting."""

    def test_format_summary_basic(self):
        """Test basic summary formatting."""
        summary = {
            "generated_at": "2024-01-01T00:00:00",
            "total_papers": 100,
            "total_extractions": 95,
        }

        output = format_summary(summary)
        assert "# Index Summary" in output
        assert "**Total Papers:** 100" in output
        assert "**Total Extractions:** 95" in output

    def test_format_summary_with_stats(self):
        """Test summary with detailed statistics."""
        summary = {
            "generated_at": "2024-01-01T00:00:00",
            "total_papers": 100,
            "total_extractions": 95,
            "papers_by_type": {"journalArticle": 60, "conferencePaper": 40},
            "papers_by_year": {"2024": 30, "2023": 70},
            "papers_by_collection": {"ML": 50, "NLP": 50},
            "top_disciplines": {"Machine Learning": 40, "NLP": 35},
        }

        output = format_summary(summary)
        assert "## Papers by Type" in output
        assert "journalArticle: 60" in output
        assert "## Papers by Year" in output

    def test_format_summary_with_vector_store(self):
        """Test summary with vector store stats."""
        summary = {
            "generated_at": "2024-01-01",
            "total_papers": 50,
            "total_extractions": 50,
            "vector_store": {
                "total_chunks": 500,
                "unique_papers": 50,
            },
        }

        output = format_summary(summary)
        assert "## Vector Store" in output
        assert "**Total Chunks:** 500" in output


class TestSearchEngine:
    """Tests for SearchEngine class."""

    @pytest.fixture
    def mock_dependencies(self, temp_index_dir):
        """Mock search engine dependencies."""
        with patch("src.query.search.StructuredStore") as mock_store, \
             patch("src.query.search.VectorStore") as mock_vector, \
             patch("src.query.search.EmbeddingGenerator") as mock_embed:

            # Setup mock returns
            mock_store_instance = MagicMock()
            mock_store_instance.get_paper_with_extraction.return_value = {
                "paper": {"title": "Test"},
                "extraction": {},
            }
            mock_store_instance.load_summary.return_value = {"total_papers": 10}
            mock_store_instance.generate_summary.return_value = {
                "papers_by_collection": {"ML": 5},
                "papers_by_type": {"journalArticle": 10},
                "papers_by_year": {"2024": 10},
            }
            mock_store.return_value = mock_store_instance

            mock_vector_instance = MagicMock()
            mock_vector_instance.search.return_value = []
            mock_vector_instance.get_stats.return_value = {"total_chunks": 100}
            mock_vector.return_value = mock_vector_instance

            mock_embed_instance = MagicMock()
            mock_embed_instance.embed_text.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance

            yield {
                "store": mock_store_instance,
                "vector": mock_vector_instance,
                "embed": mock_embed_instance,
                "index_dir": temp_index_dir,
            }

    def test_engine_initialization(self, mock_dependencies):
        """Test engine initialization."""
        engine = SearchEngine(
            index_dir=mock_dependencies["index_dir"],
        )
        assert engine is not None

    def test_search_empty_results(self, mock_dependencies):
        """Test search with no results."""
        engine = SearchEngine(
            index_dir=mock_dependencies["index_dir"],
        )

        results = engine.search("test query")
        assert results == []

    def test_get_summary(self, mock_dependencies):
        """Test getting summary."""
        engine = SearchEngine(
            index_dir=mock_dependencies["index_dir"],
        )

        summary = engine.get_summary()
        assert "total_papers" in summary
        assert "vector_store" in summary

    def test_get_collections(self, mock_dependencies):
        """Test getting collections."""
        engine = SearchEngine(
            index_dir=mock_dependencies["index_dir"],
        )

        collections = engine.get_collections()
        assert "ML" in collections

    def test_get_item_types(self, mock_dependencies):
        """Test getting item types."""
        engine = SearchEngine(
            index_dir=mock_dependencies["index_dir"],
        )

        item_types = engine.get_item_types()
        assert "journalArticle" in item_types

    def test_get_year_range(self, mock_dependencies):
        """Test getting year range."""
        engine = SearchEngine(
            index_dir=mock_dependencies["index_dir"],
        )

        min_year, max_year = engine.get_year_range()
        assert min_year == 2024
        assert max_year == 2024

    def test_get_paper(self, mock_dependencies):
        """Test getting single paper."""
        engine = SearchEngine(
            index_dir=mock_dependencies["index_dir"],
        )

        paper = engine.get_paper("paper_001")
        assert paper is not None
        assert "paper" in paper
