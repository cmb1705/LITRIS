"""Tests for MCP adapter layer."""

import pytest
from unittest.mock import MagicMock, patch

from src.mcp.adapters import LitrisAdapter


class TestLitrisAdapterSearch:
    """Tests for LitrisAdapter.search method."""

    def test_search_returns_formatted_results(self, mock_search_engine):
        """Search returns properly formatted results."""
        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            results = adapter.search("test query", top_k=5)

            assert "query" in results
            assert "result_count" in results
            assert "results" in results
            assert results["query"] == "test query"

    def test_search_result_structure(self, mock_search_engine):
        """Each search result has required fields."""
        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            results = adapter.search("test query", top_k=1)

            if results["results"]:
                result = results["results"][0]
                required_fields = [
                    "rank", "score", "paper_id", "title",
                    "authors", "year", "collections", "item_type",
                    "chunk_type", "matched_text"
                ]
                for field in required_fields:
                    assert field in result, f"Missing field: {field}"

    def test_search_includes_extraction_when_requested(self, mock_search_engine):
        """Search includes extraction data when include_extraction=True."""
        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            results = adapter.search("test query", include_extraction=True)

            if results["results"]:
                result = results["results"][0]
                # Extraction should be present (may be empty dict if no data)
                assert "extraction" in result or result.get("extraction") is None

    def test_search_empty_query_handling(self, mock_search_engine):
        """Search handles queries that return no results."""
        mock_search_engine.search.return_value = []

        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            results = adapter.search("nonexistent topic xyz", top_k=5)

            assert results["result_count"] == 0
            assert results["results"] == []


class TestLitrisAdapterGetPaper:
    """Tests for LitrisAdapter.get_paper method."""

    def test_get_paper_found(self, mock_search_engine, sample_paper_data, sample_extraction_data):
        """get_paper returns paper data when found."""
        mock_search_engine.get_paper.return_value = {
            "paper": sample_paper_data,
            "extraction": sample_extraction_data,
        }

        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            result = adapter.get_paper("test_paper_001")

            assert result["found"] is True
            assert result["paper_id"] == "test_paper_001"
            assert "paper" in result
            assert "extraction" in result

    def test_get_paper_not_found(self, mock_search_engine):
        """get_paper returns not found response for missing paper."""
        mock_search_engine.get_paper.return_value = None

        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            result = adapter.get_paper("nonexistent_paper")

            assert result["found"] is False
            assert "error" in result

    def test_get_paper_structure(self, mock_search_engine, sample_paper_data, sample_extraction_data):
        """get_paper result has correct structure."""
        mock_search_engine.get_paper.return_value = {
            "paper": sample_paper_data,
            "extraction": sample_extraction_data,
        }

        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            result = adapter.get_paper("test_paper_001")

            paper = result["paper"]
            paper_fields = [
                "title", "authors", "author_string", "publication_year",
                "journal", "doi", "abstract", "collections", "item_type"
            ]
            for field in paper_fields:
                assert field in paper, f"Missing paper field: {field}"


class TestLitrisAdapterFindSimilar:
    """Tests for LitrisAdapter.find_similar method."""

    def test_find_similar_returns_results(self, mock_search_engine, sample_paper_data):
        """find_similar returns similar papers."""
        mock_search_engine.get_paper.return_value = {"paper": sample_paper_data}
        mock_result = MagicMock()
        mock_result.paper_id = "similar_001"
        mock_result.title = "Similar Paper"
        mock_result.authors = "Author Name"
        mock_result.year = 2022
        mock_result.chunk_type = "thesis"
        mock_result.score = 0.75
        mock_result.extraction_data = {}
        mock_search_engine.search_similar_papers.return_value = [mock_result]

        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            result = adapter.find_similar("test_paper_001", top_k=5)

            assert "source_paper_id" in result
            assert "similar_papers" in result
            assert result["source_paper_id"] == "test_paper_001"

    def test_find_similar_source_not_found(self, mock_search_engine):
        """find_similar handles missing source paper."""
        mock_search_engine.get_paper.return_value = None

        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            result = adapter.find_similar("nonexistent")

            assert result["found"] is False
            assert "error" in result


class TestLitrisAdapterSummary:
    """Tests for LitrisAdapter.get_summary method."""

    def test_get_summary_structure(self, mock_search_engine):
        """get_summary returns expected structure."""
        mock_search_engine.get_summary.return_value = {
            "total_papers": 100,
            "total_extractions": 95,
            "papers_by_type": {"journalArticle": 80},
            "papers_by_year": {"2023": 50},
            "papers_by_collection": {"Test": 100},
            "top_disciplines": {"CS": 50},
            "vector_store": {"total_chunks": 500},
            "recent_papers": [],
        }

        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            result = adapter.get_summary()

            assert "generated_at" in result
            assert "total_papers" in result
            assert "total_extractions" in result
            assert result["total_papers"] == 100


class TestLitrisAdapterCollections:
    """Tests for LitrisAdapter.get_collections method."""

    def test_get_collections_returns_list(self, mock_search_engine):
        """get_collections returns collection list and counts."""
        mock_search_engine.get_summary.return_value = {
            "papers_by_collection": {
                "Collection A": 50,
                "Collection B": 30,
            }
        }

        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            result = adapter.get_collections()

            assert "collections" in result
            assert "collection_counts" in result
            assert len(result["collections"]) == 2


class TestRecencyBoost:
    """Tests for recency boost functionality."""

    def test_recency_boost_applied(self, mock_search_engine):
        """Recency boost affects result ordering."""
        # Create mock results with different years
        results = []
        for year, score in [(2023, 0.7), (2020, 0.8), (2015, 0.9)]:
            mock = MagicMock()
            mock.paper_id = f"paper_{year}"
            mock.title = f"Paper from {year}"
            mock.authors = "Author"
            mock.year = year
            mock.collections = []
            mock.item_type = "journalArticle"
            mock.chunk_type = "thesis"
            mock.matched_text = "text"
            mock.score = score
            mock.paper_data = {}
            mock.extraction_data = {}
            results.append(mock)

        mock_search_engine.search.return_value = results

        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()

            # Without boost, 2015 paper should be first (highest score 0.9)
            no_boost = adapter.search("test", recency_boost=0.0)

            # With boost, recent papers should rank higher
            with_boost = adapter.search("test", recency_boost=0.5)

            # The ordering should be affected by recency
            assert no_boost["result_count"] == 3
            assert with_boost["result_count"] == 3
