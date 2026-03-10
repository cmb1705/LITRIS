"""Tests for multi-round agentic search with gap analysis."""

from unittest.mock import MagicMock, patch

from src.mcp.validators import validate_max_rounds
from src.query.agentic import (
    AgenticRound,
    AgenticSearchResult,
    GapAnalysis,
    _format_results_for_analysis,
    analyze_gaps,
)
from src.query.search import EnrichedResult, SearchEngine


class TestFormatResultsForAnalysis:
    """Tests for formatting results into LLM-readable summaries."""

    def test_formats_basic_results(self):
        results = [
            {
                "title": "Network Analysis Methods",
                "year": 2023,
                "authors": "Smith, J.",
                "extraction": {
                    "thesis_statement": "Networks reveal hidden structure.",
                    "discipline_tags": ["network science", "graph theory"],
                },
            },
        ]
        output = _format_results_for_analysis(results)

        assert "Network Analysis Methods" in output
        assert "2023" in output
        assert "Smith, J." in output
        assert "Networks reveal hidden structure" in output
        assert "network science" in output

    def test_handles_empty_results(self):
        output = _format_results_for_analysis([])
        assert "no results found" in output

    def test_handles_missing_fields(self):
        results = [{"title": "Minimal Paper"}]
        output = _format_results_for_analysis(results)
        assert "Minimal Paper" in output
        assert "n/a" in output  # missing year

    def test_truncates_long_thesis(self):
        results = [
            {
                "title": "Long Paper",
                "year": 2024,
                "authors": "Doe",
                "extraction": {
                    "thesis_statement": "A" * 500,
                },
            },
        ]
        output = _format_results_for_analysis(results)
        # Thesis should be truncated to 200 chars
        assert len(output) < 600


class TestAnalyzeGaps:
    """Tests for LLM-based gap analysis."""

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_returns_gaps_and_queries(self, mock_factory):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            '{"gaps": ["missing methodology papers"], '
            '"follow_up_queries": ["qualitative research methods"]}',
            100,
            50,
        )
        mock_factory.return_value = mock_client

        result = analyze_gaps("test query", [{"title": "Paper 1"}])

        assert isinstance(result, GapAnalysis)
        assert len(result.gaps) == 1
        assert "missing methodology" in result.gaps[0]
        assert len(result.follow_up_queries) == 1

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_returns_empty_on_no_gaps(self, mock_factory):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            '{"gaps": [], "follow_up_queries": []}',
            100,
            50,
        )
        mock_factory.return_value = mock_client

        result = analyze_gaps("test query", [{"title": "Paper 1"}])

        assert result.gaps == []
        assert result.follow_up_queries == []

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_returns_empty_on_failure(self, mock_factory):
        mock_client = MagicMock()
        mock_client._call_api.side_effect = Exception("API error")
        mock_factory.return_value = mock_client

        result = analyze_gaps("test query", [{"title": "Paper 1"}])

        assert result.gaps == []
        assert result.follow_up_queries == []

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_handles_markdown_fenced_json(self, mock_factory):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            '```json\n{"gaps": ["gap1"], "follow_up_queries": ["q1"]}\n```',
            100,
            50,
        )
        mock_factory.return_value = mock_client

        result = analyze_gaps("test query", [])

        assert len(result.gaps) == 1
        assert len(result.follow_up_queries) == 1

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_clamps_to_three_items(self, mock_factory):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            '{"gaps": ["g1", "g2", "g3", "g4", "g5"], '
            '"follow_up_queries": ["q1", "q2", "q3", "q4"]}',
            100,
            50,
        )
        mock_factory.return_value = mock_client

        result = analyze_gaps("test query", [])

        assert len(result.gaps) == 3
        assert len(result.follow_up_queries) == 3

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_handles_non_dict_response(self, mock_factory):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            '["not", "a", "dict"]',
            100,
            50,
        )
        mock_factory.return_value = mock_client

        result = analyze_gaps("test query", [])

        assert result.gaps == []
        assert result.follow_up_queries == []


class TestSearchEngineAgentic:
    """Tests for SearchEngine.search_agentic()."""

    def _make_engine(self, tmp_path):
        """Create a SearchEngine with mocked components."""
        from src.indexing.structured_store import StructuredStore

        store = StructuredStore(tmp_path)
        store.save_papers([
            {"paper_id": "p1", "title": "Paper One", "author_string": "Smith",
             "publication_year": 2024, "collections": [], "item_type": "journalArticle"},
            {"paper_id": "p2", "title": "Paper Two", "author_string": "Jones",
             "publication_year": 2023, "collections": [], "item_type": "journalArticle"},
            {"paper_id": "p3", "title": "Paper Three", "author_string": "Lee",
             "publication_year": 2022, "collections": [], "item_type": "journalArticle"},
            {"paper_id": "p4", "title": "Paper Four", "author_string": "Park",
             "publication_year": 2021, "collections": [], "item_type": "journalArticle"},
        ])
        store.save_extractions({
            "p1": {"paper_id": "p1", "extraction": {"thesis_statement": "Thesis 1"}},
            "p2": {"paper_id": "p2", "extraction": {"thesis_statement": "Thesis 2"}},
            "p3": {"paper_id": "p3", "extraction": {"thesis_statement": "Thesis 3"}},
            "p4": {"paper_id": "p4", "extraction": {"thesis_statement": "Thesis 4"}},
        })

        engine = SearchEngine.__new__(SearchEngine)
        engine.structured_store = store
        engine.vector_store = MagicMock()
        engine.embedding_generator = MagicMock()

        return engine

    def _make_search_result(self, paper_id, title, score):
        from src.indexing.vector_store import SearchResult

        return SearchResult(
            paper_id=paper_id,
            chunk_id=f"{paper_id}_c1",
            chunk_type="thesis",
            text=f"text for {paper_id}",
            score=score,
            metadata={"title": title},
        )

    @patch("src.query.agentic.analyze_gaps")
    def test_initial_search_only_when_no_gaps(self, mock_analyze, tmp_path):
        """If gap analysis finds no gaps, stops after round 0."""
        engine = self._make_engine(tmp_path)
        engine.embedding_generator.embed_text.return_value = [0.1] * 384
        engine.vector_store.search.return_value = [
            self._make_search_result("p1", "Paper One", 0.9),
            self._make_search_result("p2", "Paper Two", 0.8),
        ]

        mock_analyze.return_value = GapAnalysis(gaps=[], follow_up_queries=[])

        results, metadata = engine.search_agentic("test query", top_k=10, max_rounds=2)

        assert len(results) == 2
        assert metadata.total_papers == 2
        # Round 0 (initial) + Round 1 (stopped early)
        assert len(metadata.rounds) == 2
        assert metadata.rounds[1].new_papers == 0

    @patch("src.query.agentic.analyze_gaps")
    def test_follow_up_finds_new_papers(self, mock_analyze, tmp_path):
        """Follow-up queries should discover papers not in initial results."""
        engine = self._make_engine(tmp_path)
        engine.embedding_generator.embed_text.return_value = [0.1] * 384

        # Initial search returns p1, p2
        # Follow-up search returns p2 (duplicate) and p3 (new)
        engine.vector_store.search.side_effect = [
            [
                self._make_search_result("p1", "Paper One", 0.9),
                self._make_search_result("p2", "Paper Two", 0.8),
            ],
            [
                self._make_search_result("p2", "Paper Two", 0.85),
                self._make_search_result("p3", "Paper Three", 0.7),
            ],
        ]

        mock_analyze.side_effect = [
            GapAnalysis(
                gaps=["missing methodology"],
                follow_up_queries=["qualitative methods"],
            ),
            GapAnalysis(gaps=[], follow_up_queries=[]),
        ]

        results, metadata = engine.search_agentic("test query", top_k=10, max_rounds=2)

        paper_ids = [r.paper_id for r in results]
        assert "p1" in paper_ids
        assert "p2" in paper_ids
        assert "p3" in paper_ids
        assert metadata.rounds[1].new_papers == 1
        assert metadata.rounds[1].gap_analysis.gaps == ["missing methodology"]

    @patch("src.query.agentic.analyze_gaps")
    def test_deduplicates_across_rounds(self, mock_analyze, tmp_path):
        """Papers found in multiple rounds should not be duplicated."""
        engine = self._make_engine(tmp_path)
        engine.embedding_generator.embed_text.return_value = [0.1] * 384

        # Both rounds return p1
        engine.vector_store.search.side_effect = [
            [self._make_search_result("p1", "Paper One", 0.9)],
            [self._make_search_result("p1", "Paper One", 0.95)],
        ]

        mock_analyze.return_value = GapAnalysis(
            gaps=["gap"], follow_up_queries=["follow up"],
        )

        results, metadata = engine.search_agentic("test", top_k=10, max_rounds=1)

        assert len(results) == 1
        assert results[0].paper_id == "p1"

    @patch("src.query.agentic.analyze_gaps")
    def test_respects_top_k(self, mock_analyze, tmp_path):
        """Final results should be limited to top_k."""
        engine = self._make_engine(tmp_path)
        engine.embedding_generator.embed_text.return_value = [0.1] * 384

        engine.vector_store.search.side_effect = [
            [
                self._make_search_result("p1", "Paper One", 0.9),
                self._make_search_result("p2", "Paper Two", 0.8),
            ],
            [
                self._make_search_result("p3", "Paper Three", 0.7),
                self._make_search_result("p4", "Paper Four", 0.6),
            ],
        ]

        mock_analyze.side_effect = [
            GapAnalysis(gaps=["gap"], follow_up_queries=["q"]),
            GapAnalysis(gaps=[], follow_up_queries=[]),
        ]

        results, _ = engine.search_agentic("test", top_k=2, max_rounds=2)

        assert len(results) == 2
        # Should be the highest-scoring papers
        assert results[0].score >= results[1].score

    @patch("src.query.agentic.analyze_gaps")
    def test_stops_when_no_new_papers(self, mock_analyze, tmp_path):
        """Should stop early if follow-up finds no new papers."""
        engine = self._make_engine(tmp_path)
        engine.embedding_generator.embed_text.return_value = [0.1] * 384

        # Both rounds return the same paper
        engine.vector_store.search.return_value = [
            self._make_search_result("p1", "Paper One", 0.9),
        ]

        mock_analyze.return_value = GapAnalysis(
            gaps=["gap"], follow_up_queries=["q"],
        )

        results, metadata = engine.search_agentic("test", top_k=10, max_rounds=3)

        # Should stop after round 1 since no new papers
        assert len(metadata.rounds) == 2  # round 0 + round 1
        assert metadata.rounds[1].new_papers == 0

    @patch("src.query.agentic.analyze_gaps")
    def test_passes_filters_to_search(self, mock_analyze, tmp_path):
        """Filters should be passed through to each search round."""
        engine = self._make_engine(tmp_path)
        engine.embedding_generator.embed_text.return_value = [0.1] * 384
        engine.vector_store.search.return_value = []

        mock_analyze.return_value = GapAnalysis(gaps=[], follow_up_queries=[])

        engine.search_agentic(
            "test",
            year_min=2020,
            year_max=2025,
            collections=["ML"],
            quality_min=3,
        )

        call_kwargs = engine.vector_store.search.call_args[1]
        assert call_kwargs["year_min"] == 2020
        assert call_kwargs["year_max"] == 2025
        assert call_kwargs["collections"] == ["ML"]
        assert call_kwargs["quality_min"] == 3

    @patch("src.query.agentic.analyze_gaps")
    def test_metadata_tracks_rounds(self, mock_analyze, tmp_path):
        """Metadata should record round-by-round details."""
        engine = self._make_engine(tmp_path)
        engine.embedding_generator.embed_text.return_value = [0.1] * 384

        engine.vector_store.search.side_effect = [
            [self._make_search_result("p1", "Paper One", 0.9)],
            [self._make_search_result("p2", "Paper Two", 0.8)],
        ]

        mock_analyze.side_effect = [
            GapAnalysis(
                gaps=["missing topic X"],
                follow_up_queries=["topic X methods"],
            ),
            GapAnalysis(gaps=[], follow_up_queries=[]),
        ]

        _, metadata = engine.search_agentic("test", top_k=10, max_rounds=2)

        assert metadata.original_query == "test"
        assert metadata.rounds[0].round_number == 0
        assert metadata.rounds[0].queries_used == ["test"]
        assert metadata.rounds[1].round_number == 1
        assert metadata.rounds[1].queries_used == ["topic X methods"]
        assert metadata.rounds[1].gap_analysis.gaps == ["missing topic X"]


class TestValidateMaxRounds:
    """Tests for validate_max_rounds."""

    def test_valid_values(self):
        for i in range(1, 6):
            assert validate_max_rounds(i) == i

    def test_clamps_low(self):
        assert validate_max_rounds(0) == 1
        assert validate_max_rounds(-5) == 1

    def test_clamps_high(self):
        assert validate_max_rounds(6) == 5
        assert validate_max_rounds(100) == 5


class TestAgenticAdapterIntegration:
    """Tests for agentic search in the MCP adapter layer."""

    def test_adapter_includes_rounds_metadata(self):
        from src.mcp.adapters import LitrisAdapter

        adapter = LitrisAdapter.__new__(LitrisAdapter)
        mock_engine = MagicMock()

        mock_results = [
            EnrichedResult(
                paper_id="p1", title="Paper One", authors="Smith",
                year=2024, collections=[], item_type="journalArticle",
                chunk_type="thesis", matched_text="some text",
                score=0.9, paper_data={}, extraction_data={},
            ),
        ]
        mock_metadata = AgenticSearchResult(
            original_query="test",
            rounds=[
                AgenticRound(
                    round_number=0, queries_used=["test"],
                    papers_found=1, new_papers=1,
                ),
                AgenticRound(
                    round_number=1, queries_used=["follow up"],
                    papers_found=0, new_papers=0,
                    gap_analysis=GapAnalysis(
                        gaps=["missing theory"],
                        follow_up_queries=["follow up"],
                    ),
                ),
            ],
            total_papers=1,
        )
        mock_engine.search_agentic.return_value = (mock_results, mock_metadata)
        adapter._engine = mock_engine

        result = adapter.search_agentic(query="test", top_k=10, max_rounds=2)

        assert result["result_count"] == 1
        assert result["total_papers_explored"] == 1
        assert result["rounds_completed"] == 2
        assert len(result["rounds"]) == 2
        assert result["rounds"][0]["round"] == 0
        assert result["rounds"][1]["gaps_identified"] == ["missing theory"]
        assert result["results"][0]["paper_id"] == "p1"
