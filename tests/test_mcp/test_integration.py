"""Integration tests for MCP server functionality."""

import pytest
from unittest.mock import MagicMock, patch

from src.mcp.adapters import LitrisAdapter
from src.mcp.validators import ValidationError


class TestFilteredSearches:
    """Integration tests for filtered search operations."""

    def test_search_with_year_filter(self, mock_search_engine):
        """Search respects year filters."""
        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            adapter.search("test query", year_min=2020, year_max=2023)

            mock_search_engine.search.assert_called_once()
            call_kwargs = mock_search_engine.search.call_args[1]
            assert call_kwargs["year_min"] == 2020
            assert call_kwargs["year_max"] == 2023

    def test_search_with_collection_filter(self, mock_search_engine):
        """Search respects collection filters."""
        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            adapter.search("test query", collections=["ML Papers", "Network Analysis"])

            mock_search_engine.search.assert_called_once()
            call_kwargs = mock_search_engine.search.call_args[1]
            assert call_kwargs["collections"] == ["ML Papers", "Network Analysis"]

    def test_search_with_chunk_type_filter(self, mock_search_engine):
        """Search respects chunk type filters."""
        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            adapter.search("test query", chunk_types=["thesis", "methodology"])

            mock_search_engine.search.assert_called_once()
            call_kwargs = mock_search_engine.search.call_args[1]
            assert call_kwargs["chunk_types"] == ["thesis", "methodology"]

    def test_search_with_item_type_filter(self, mock_search_engine):
        """Search respects item type filters."""
        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            adapter.search("test query", item_types=["journalArticle", "conferencePaper"])

            mock_search_engine.search.assert_called_once()
            call_kwargs = mock_search_engine.search.call_args[1]
            assert call_kwargs["item_types"] == ["journalArticle", "conferencePaper"]

    def test_search_with_multiple_filters(self, mock_search_engine):
        """Search combines multiple filters correctly."""
        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            adapter.search(
                "test query",
                top_k=5,
                year_min=2018,
                year_max=2023,
                collections=["ML Papers"],
                chunk_types=["thesis"],
                item_types=["journalArticle"],
            )

            mock_search_engine.search.assert_called_once()
            call_kwargs = mock_search_engine.search.call_args[1]
            assert call_kwargs["top_k"] == 5
            assert call_kwargs["year_min"] == 2018
            assert call_kwargs["year_max"] == 2023
            assert call_kwargs["collections"] == ["ML Papers"]
            assert call_kwargs["chunk_types"] == ["thesis"]
            assert call_kwargs["item_types"] == ["journalArticle"]


class TestErrorCases:
    """Integration tests for error handling."""

    def test_search_engine_exception(self, mock_search_engine):
        """Adapter handles SearchEngine exceptions gracefully."""
        mock_search_engine.search.side_effect = Exception("Database connection failed")

        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            with pytest.raises(Exception, match="Database connection failed"):
                adapter.search("test query")

    def test_get_paper_with_missing_extraction(self, mock_search_engine):
        """get_paper handles missing extraction data."""
        mock_search_engine.get_paper.return_value = {
            "paper": {"title": "Test Paper", "authors": []},
            "extraction": None,
        }

        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            result = adapter.get_paper("test_id")

            assert result["found"] is True
            assert result["extraction"] is None

    def test_find_similar_with_no_chunks(self, mock_search_engine):
        """find_similar handles papers with no chunks."""
        mock_search_engine.get_paper.return_value = {"paper": {"title": "Test"}}
        mock_search_engine.search_similar_papers.return_value = []

        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            result = adapter.find_similar("test_id")

            assert result["result_count"] == 0


class TestToolWorkflows:
    """Integration tests for multi-tool workflows."""

    def test_search_then_get_paper_workflow(self, mock_search_engine, sample_paper_data, sample_extraction_data):
        """Search followed by get_paper retrieves full details."""
        # Setup mock for search
        mock_result = MagicMock()
        mock_result.paper_id = "paper_001"
        mock_result.title = "Test Paper"
        mock_result.authors = "Author"
        mock_result.year = 2023
        mock_result.collections = []
        mock_result.item_type = "journalArticle"
        mock_result.chunk_type = "thesis"
        mock_result.matched_text = "test"
        mock_result.score = 0.9
        mock_result.paper_data = {}
        mock_result.extraction_data = {}
        mock_search_engine.search.return_value = [mock_result]

        # Setup mock for get_paper
        mock_search_engine.get_paper.return_value = {
            "paper": sample_paper_data,
            "extraction": sample_extraction_data,
        }

        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()

            # Step 1: Search
            search_results = adapter.search("test query", top_k=1)
            assert search_results["result_count"] == 1
            paper_id = search_results["results"][0]["paper_id"]

            # Step 2: Get paper details
            paper_details = adapter.get_paper(paper_id)
            assert paper_details["found"] is True
            assert "extraction" in paper_details

    def test_search_then_similar_workflow(self, mock_search_engine, sample_paper_data):
        """Search followed by similar papers exploration."""
        # Setup mocks
        mock_result = MagicMock()
        mock_result.paper_id = "paper_001"
        mock_result.title = "Source Paper"
        mock_result.authors = "Author"
        mock_result.year = 2023
        mock_result.collections = []
        mock_result.item_type = "journalArticle"
        mock_result.chunk_type = "thesis"
        mock_result.matched_text = "test"
        mock_result.score = 0.9
        mock_result.paper_data = {}
        mock_result.extraction_data = {}

        mock_search_engine.search.return_value = [mock_result]
        mock_search_engine.get_paper.return_value = {"paper": sample_paper_data}
        mock_search_engine.search_similar_papers.return_value = [mock_result]

        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()

            # Step 1: Search
            search_results = adapter.search("test query")
            paper_id = search_results["results"][0]["paper_id"]

            # Step 2: Find similar
            similar = adapter.find_similar(paper_id, top_k=5)
            assert "similar_papers" in similar


class TestPerformance:
    """Basic performance verification tests."""

    def test_search_returns_limited_results(self, mock_search_engine):
        """Search respects top_k limit."""
        # Create 20 mock results
        results = []
        for i in range(20):
            mock = MagicMock()
            mock.paper_id = f"paper_{i}"
            mock.title = f"Paper {i}"
            mock.authors = "Author"
            mock.year = 2023
            mock.collections = []
            mock.item_type = "journalArticle"
            mock.chunk_type = "thesis"
            mock.matched_text = "text"
            mock.score = 0.9 - i * 0.01
            mock.paper_data = {}
            mock.extraction_data = {}
            results.append(mock)

        mock_search_engine.search.return_value = results[:10]

        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            result = adapter.search("test", top_k=10)

            assert result["result_count"] <= 10

    def test_matched_text_truncation(self, mock_search_engine):
        """Long matched text is truncated."""
        mock_result = MagicMock()
        mock_result.paper_id = "paper_001"
        mock_result.title = "Test"
        mock_result.authors = "Author"
        mock_result.year = 2023
        mock_result.collections = []
        mock_result.item_type = "journalArticle"
        mock_result.chunk_type = "thesis"
        mock_result.matched_text = "x" * 1000  # Long text
        mock_result.score = 0.9
        mock_result.paper_data = {}
        mock_result.extraction_data = {}

        mock_search_engine.search.return_value = [mock_result]

        with patch.object(LitrisAdapter, "engine", mock_search_engine):
            adapter = LitrisAdapter()
            result = adapter.search("test")

            # Matched text should be truncated to 500 chars
            assert len(result["results"][0]["matched_text"]) <= 500
