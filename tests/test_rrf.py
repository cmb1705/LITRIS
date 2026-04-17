"""Tests for Reciprocal Rank Fusion multi-query search."""

from unittest.mock import MagicMock, patch

import pytest

from src.mcp.validators import validate_n_variants
from src.query.rrf import generate_query_variants, rrf_score
from src.query.search import EnrichedResult, SearchEngine


class TestRRFScore:
    """Tests for the RRF scoring function."""

    def test_single_ranking(self):
        rankings = [["a", "b", "c"]]
        results = rrf_score(rankings, k=60)

        assert results[0][0] == "a"
        assert results[1][0] == "b"
        assert results[2][0] == "c"
        # score = 1/(60+rank)
        assert results[0][1] == pytest.approx(1 / 61)
        assert results[1][1] == pytest.approx(1 / 62)
        assert results[2][1] == pytest.approx(1 / 63)

    def test_multiple_rankings_boost_overlap(self):
        rankings = [
            ["a", "b", "c"],
            ["b", "a", "d"],
        ]
        results = rrf_score(rankings, k=60)
        scores = dict(results)

        # "a" appears at rank 1 and rank 2: 1/61 + 1/62
        assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
        # "b" appears at rank 2 and rank 1: 1/62 + 1/61
        assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
        # "a" and "b" have same score
        assert scores["a"] == pytest.approx(scores["b"])
        # "c" only in first ranking at rank 3
        assert scores["c"] == pytest.approx(1 / 63)
        # "d" only in second ranking at rank 3
        assert scores["d"] == pytest.approx(1 / 63)

    def test_overlap_beats_single(self):
        rankings = [
            ["a", "b"],
            ["a", "c"],
        ]
        results = rrf_score(rankings, k=60)
        scores = dict(results)

        # "a" in both rankings should beat "b" or "c" in one
        assert scores["a"] > scores["b"]
        assert scores["a"] > scores["c"]

    def test_tie_breaking_alphabetical(self):
        rankings = [["b", "a"]]
        results = rrf_score(rankings, k=60)

        # "b" has higher score (rank 1), so comes first
        assert results[0][0] == "b"
        assert results[1][0] == "a"

        # Same score ties broken alphabetically
        rankings = [["b"], ["a"]]
        results = rrf_score(rankings, k=60)
        # Both at rank 1 in different lists, same score
        assert results[0][1] == results[1][1]
        # Alphabetical tie-break: "a" before "b"
        assert results[0][0] == "a"
        assert results[1][0] == "b"

    def test_empty_input(self):
        assert rrf_score([]) == []
        assert rrf_score([[]]) == []

    def test_custom_k_value(self):
        rankings = [["a", "b"]]
        results = rrf_score(rankings, k=10)
        scores = dict(results)

        assert scores["a"] == pytest.approx(1 / 11)
        assert scores["b"] == pytest.approx(1 / 12)


class TestGenerateQueryVariants:
    """Tests for LLM-based query reformulation."""

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_returns_original_plus_variants(self, mock_factory):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            '["variant 1", "variant 2", "variant 3"]',
            100,
            50,
        )
        mock_factory.return_value = mock_client

        result = generate_query_variants("test query", n_variants=3)

        assert result[0] == "test query"
        assert len(result) == 4  # original + 3 variants
        assert "variant 1" in result

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_returns_original_on_failure(self, mock_factory):
        mock_client = MagicMock()
        mock_client._call_api.side_effect = Exception("API error")
        mock_factory.return_value = mock_client

        result = generate_query_variants("test query", n_variants=3)

        assert result == ["test query"]

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_handles_markdown_fenced_json(self, mock_factory):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            '```json\n["variant A", "variant B"]\n```',
            100,
            50,
        )
        mock_factory.return_value = mock_client

        result = generate_query_variants("test query", n_variants=2)

        assert len(result) == 3
        assert result[0] == "test query"
        assert "variant A" in result

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_clamps_to_n_variants(self, mock_factory):
        mock_client = MagicMock()
        # LLM returns more than requested
        mock_client._call_api.return_value = (
            '["v1", "v2", "v3", "v4", "v5"]',
            100,
            50,
        )
        mock_factory.return_value = mock_client

        result = generate_query_variants("test query", n_variants=2)

        # original + 2 variants (clamped)
        assert len(result) == 3

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_handles_non_list_response(self, mock_factory):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            '{"error": "unexpected format"}',
            100,
            50,
        )
        mock_factory.return_value = mock_client

        result = generate_query_variants("test query", n_variants=3)

        assert result == ["test query"]


class TestSearchEngineRRF:
    """Tests for SearchEngine.search_rrf()."""

    def _make_engine(self, tmp_path):
        """Create a SearchEngine with mocked components."""
        from src.indexing.structured_store import StructuredStore

        store = StructuredStore(tmp_path)
        store.save_papers(
            [
                {
                    "paper_id": "p1",
                    "title": "Paper One",
                    "author_string": "Smith",
                    "publication_year": 2024,
                    "collections": [],
                    "item_type": "journalArticle",
                },
                {
                    "paper_id": "p2",
                    "title": "Paper Two",
                    "author_string": "Jones",
                    "publication_year": 2023,
                    "collections": [],
                    "item_type": "journalArticle",
                },
                {
                    "paper_id": "p3",
                    "title": "Paper Three",
                    "author_string": "Lee",
                    "publication_year": 2022,
                    "collections": [],
                    "item_type": "journalArticle",
                },
            ]
        )
        store.save_extractions(
            {
                "p1": {"paper_id": "p1", "extraction": {"q02_thesis": "Thesis 1"}},
                "p2": {"paper_id": "p2", "extraction": {"q02_thesis": "Thesis 2"}},
                "p3": {"paper_id": "p3", "extraction": {"q02_thesis": "Thesis 3"}},
            }
        )

        engine = SearchEngine.__new__(SearchEngine)
        engine.structured_store = store
        engine.vector_store = MagicMock()
        engine.embedding_generator = MagicMock()

        return engine

    @patch("src.query.rrf.generate_query_variants")
    def test_deduplicates_results(self, mock_variants, tmp_path):
        engine = self._make_engine(tmp_path)
        mock_variants.return_value = ["query 1", "query 2"]

        from src.indexing.vector_store import SearchResult

        # Both queries return the same papers
        engine.embedding_generator.embed_text.return_value = [0.1] * 384
        engine.vector_store.search.side_effect = [
            [
                SearchResult(
                    paper_id="p1",
                    chunk_id="c1",
                    chunk_type="dim_q02",
                    text="text1",
                    score=0.9,
                    metadata={"title": "Paper One"},
                ),
                SearchResult(
                    paper_id="p2",
                    chunk_id="c2",
                    chunk_type="dim_q02",
                    text="text2",
                    score=0.8,
                    metadata={"title": "Paper Two"},
                ),
            ],
            [
                SearchResult(
                    paper_id="p2",
                    chunk_id="c3",
                    chunk_type="dim_q02",
                    text="text2b",
                    score=0.85,
                    metadata={"title": "Paper Two"},
                ),
                SearchResult(
                    paper_id="p3",
                    chunk_id="c4",
                    chunk_type="dim_q02",
                    text="text3",
                    score=0.7,
                    metadata={"title": "Paper Three"},
                ),
            ],
        ]

        results, variants = engine.search_rrf("test", top_k=10)

        # No duplicate paper_ids
        paper_ids = [r.paper_id for r in results]
        assert len(paper_ids) == len(set(paper_ids))
        assert len(results) == 3  # p1, p2, p3

    @patch("src.query.rrf.generate_query_variants")
    def test_respects_top_k(self, mock_variants, tmp_path):
        engine = self._make_engine(tmp_path)
        mock_variants.return_value = ["query 1"]

        from src.indexing.vector_store import SearchResult

        engine.embedding_generator.embed_text.return_value = [0.1] * 384
        engine.vector_store.search.return_value = [
            SearchResult(
                paper_id="p1",
                chunk_id="c1",
                chunk_type="dim_q02",
                text="t1",
                score=0.9,
                metadata={"title": "Paper One"},
            ),
            SearchResult(
                paper_id="p2",
                chunk_id="c2",
                chunk_type="dim_q02",
                text="t2",
                score=0.8,
                metadata={"title": "Paper Two"},
            ),
            SearchResult(
                paper_id="p3",
                chunk_id="c3",
                chunk_type="dim_q02",
                text="t3",
                score=0.7,
                metadata={"title": "Paper Three"},
            ),
        ]

        results, _ = engine.search_rrf("test", top_k=2)

        assert len(results) == 2

    @patch("src.query.rrf.generate_query_variants")
    def test_passes_filters(self, mock_variants, tmp_path):
        engine = self._make_engine(tmp_path)
        mock_variants.return_value = ["query 1"]

        engine.embedding_generator.embed_text.return_value = [0.1] * 384
        engine.vector_store.search.return_value = []

        engine.search_rrf(
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

    @patch("src.query.rrf.generate_query_variants")
    def test_returns_query_variants(self, mock_variants, tmp_path):
        engine = self._make_engine(tmp_path)
        mock_variants.return_value = ["original", "variant 1", "variant 2"]

        engine.embedding_generator.embed_text.return_value = [0.1] * 384
        engine.vector_store.search.return_value = []

        _, variants = engine.search_rrf("original", n_variants=2)

        assert variants == ["original", "variant 1", "variant 2"]

    @patch("src.query.rrf.generate_query_variants")
    def test_rrf_scoring_boosts_overlap(self, mock_variants, tmp_path):
        engine = self._make_engine(tmp_path)
        mock_variants.return_value = ["q1", "q2"]

        from src.indexing.vector_store import SearchResult

        engine.embedding_generator.embed_text.return_value = [0.1] * 384
        # p1 appears in both rankings, p2 only in first, p3 only in second
        engine.vector_store.search.side_effect = [
            [
                SearchResult(
                    paper_id="p1",
                    chunk_id="c1",
                    chunk_type="dim_q02",
                    text="t1",
                    score=0.9,
                    metadata={"title": "Paper One"},
                ),
                SearchResult(
                    paper_id="p2",
                    chunk_id="c2",
                    chunk_type="dim_q02",
                    text="t2",
                    score=0.8,
                    metadata={"title": "Paper Two"},
                ),
            ],
            [
                SearchResult(
                    paper_id="p1",
                    chunk_id="c3",
                    chunk_type="dim_q02",
                    text="t1b",
                    score=0.85,
                    metadata={"title": "Paper One"},
                ),
                SearchResult(
                    paper_id="p3",
                    chunk_id="c4",
                    chunk_type="dim_q02",
                    text="t3",
                    score=0.7,
                    metadata={"title": "Paper Three"},
                ),
            ],
        ]

        results, _ = engine.search_rrf("test", top_k=10)

        # p1 should be ranked first (appears in both rankings)
        assert results[0].paper_id == "p1"
        assert results[0].score > results[1].score


class TestValidateNVariants:
    """Tests for validate_n_variants."""

    def test_valid_values(self):
        for i in range(1, 11):
            assert validate_n_variants(i) == i

    def test_clamps_low(self):
        assert validate_n_variants(0) == 1
        assert validate_n_variants(-5) == 1

    def test_clamps_high(self):
        assert validate_n_variants(11) == 10
        assert validate_n_variants(100) == 10


class TestRRFAdapterIntegration:
    """Tests for RRF in the MCP adapter layer."""

    def test_adapter_includes_variants(self):
        from src.mcp.adapters import LitrisAdapter

        adapter = LitrisAdapter.__new__(LitrisAdapter)
        mock_engine = MagicMock()

        mock_results = [
            EnrichedResult(
                paper_id="p1",
                title="Paper One",
                authors="Smith",
                year=2024,
                collections=[],
                item_type="journalArticle",
                chunk_type="dim_q02",
                matched_text="some text",
                score=0.032,
                paper_data={},
                extraction_data={},
            ),
        ]
        mock_engine.search_rrf.return_value = (
            mock_results,
            ["original query", "variant 1", "variant 2"],
        )
        adapter._engine = mock_engine

        result = adapter.search_rrf(
            query="original query",
            top_k=10,
            n_variants=2,
        )

        assert "query_variants" in result
        assert result["query_variants"] == ["original query", "variant 1", "variant 2"]
        assert result["result_count"] == 1
        assert result["results"][0]["paper_id"] == "p1"
