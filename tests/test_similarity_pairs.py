"""Tests for pre-computed similarity pairs."""

from unittest.mock import MagicMock

from src.indexing.structured_store import StructuredStore
from src.query.search import EnrichedResult, SearchEngine


class TestStructuredStoreSimilarity:
    """Tests for similarity pairs in StructuredStore."""

    def test_save_and_load_round_trip(self, tmp_path):
        store = StructuredStore(tmp_path)
        pairs = {
            "paper_a": [
                {"similar_paper_id": "paper_b", "similarity_score": 0.95},
                {"similar_paper_id": "paper_c", "similarity_score": 0.82},
            ],
            "paper_b": [
                {"similar_paper_id": "paper_a", "similarity_score": 0.95},
            ],
        }
        store.save_similarity_pairs(pairs, metadata={"embedding_model": "test-model"})

        loaded = store.load_similarity_pairs()
        assert len(loaded) == 2
        assert loaded["paper_a"][0]["similar_paper_id"] == "paper_b"
        assert loaded["paper_a"][0]["similarity_score"] == 0.95
        assert loaded["paper_b"][0]["similarity_score"] == 0.95

    def test_load_returns_empty_when_missing(self, tmp_path):
        store = StructuredStore(tmp_path)
        loaded = store.load_similarity_pairs()
        assert loaded == {}

    def test_save_metadata_fields(self, tmp_path):
        store = StructuredStore(tmp_path)
        pairs = {"paper_x": [{"similar_paper_id": "paper_y", "similarity_score": 0.7}]}
        store.save_similarity_pairs(pairs, metadata={"top_n_per_paper": 20})

        from src.utils.file_utils import safe_read_json

        raw = safe_read_json(store.similarity_pairs_file)
        assert raw["total_source_papers"] == 1
        assert raw["total_pairs"] == 1
        assert raw["top_n_per_paper"] == 20
        assert "generated_at" in raw

    def test_generate_summary_includes_similarity_stats(self, tmp_path):
        store = StructuredStore(tmp_path)

        # Create minimal papers and extractions
        store.save_papers([{"paper_id": "p1", "title": "Paper 1", "item_type": "journalArticle"}])
        store.save_extractions({"p1": {"paper_id": "p1", "extraction": {}}})

        # Save similarity pairs
        pairs = {"p1": [{"similar_paper_id": "p2", "similarity_score": 0.8}]}
        store.save_similarity_pairs(pairs)

        summary = store.generate_summary()
        assert "similarity_pairs" in summary
        assert summary["similarity_pairs"]["total_source_papers"] == 1
        assert summary["similarity_pairs"]["total_pairs"] == 1

    def test_generate_summary_no_similarity_file(self, tmp_path):
        store = StructuredStore(tmp_path)
        store.save_papers([{"paper_id": "p1", "title": "Paper 1", "item_type": "journalArticle"}])
        store.save_extractions({"p1": {"paper_id": "p1", "extraction": {}}})

        summary = store.generate_summary()
        assert "similarity_pairs" not in summary


class TestSearchEngineSimilarity:
    """Tests for pre-computed similarity in SearchEngine."""

    def _make_engine(self, tmp_path, pairs=None):
        """Create a SearchEngine with mocked components."""
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

        if pairs is not None:
            store.save_similarity_pairs(pairs)

        engine = SearchEngine.__new__(SearchEngine)
        engine.structured_store = store
        engine.vector_store = MagicMock()
        engine.embedding_generator = MagicMock()

        return engine

    def test_uses_precomputed_when_available(self, tmp_path):
        pairs = {
            "p1": [
                {"similar_paper_id": "p2", "similarity_score": 0.92},
                {"similar_paper_id": "p3", "similarity_score": 0.78},
            ],
        }
        engine = self._make_engine(tmp_path, pairs=pairs)

        results = engine.search_similar_papers("p1", top_k=10)

        assert len(results) == 2
        assert results[0].paper_id == "p2"
        assert results[0].score == 0.92
        assert results[1].paper_id == "p3"
        assert results[1].score == 0.78
        # Vector store should NOT be called
        engine.vector_store.get_paper_chunks.assert_not_called()

    def test_falls_back_to_vector_search(self, tmp_path):
        pairs = {
            "p1": [{"similar_paper_id": "p2", "similarity_score": 0.9}],
        }
        engine = self._make_engine(tmp_path, pairs=pairs)

        # Search for p3 which is NOT in pre-computed pairs
        engine.vector_store.get_paper_chunks.return_value = [
            {"text": "Some text", "metadata": {"chunk_type": "raptor_overview", "paper_id": "p3"}}
        ]
        engine.embedding_generator.embed_text.return_value = [0.1] * 384
        engine.vector_store.search.return_value = []

        engine.search_similar_papers("p3", top_k=10)

        # Vector store SHOULD be called since p3 not in pre-computed
        engine.vector_store.get_paper_chunks.assert_called_once_with("p3")

    def test_falls_back_when_no_pairs_file(self, tmp_path):
        engine = self._make_engine(tmp_path, pairs=None)

        engine.vector_store.get_paper_chunks.return_value = []

        engine.search_similar_papers("p1", top_k=10)

        # Should fall back since no pairs file exists
        engine.vector_store.get_paper_chunks.assert_called_once()

    def test_enrich_similarity_pairs(self, tmp_path):
        engine = self._make_engine(tmp_path)
        pairs = [
            {"similar_paper_id": "p2", "similarity_score": 0.85},
            {"similar_paper_id": "p3", "similarity_score": 0.72},
        ]

        results = engine._enrich_similarity_pairs(pairs)

        assert len(results) == 2
        assert isinstance(results[0], EnrichedResult)
        assert results[0].paper_id == "p2"
        assert results[0].title == "Paper Two"
        assert results[0].score == 0.85
        assert results[0].chunk_type == "raptor_overview"

    def test_enrich_skips_missing_papers(self, tmp_path):
        engine = self._make_engine(tmp_path)
        pairs = [
            {"similar_paper_id": "p2", "similarity_score": 0.85},
            {"similar_paper_id": "nonexistent", "similarity_score": 0.5},
        ]

        results = engine._enrich_similarity_pairs(pairs)

        assert len(results) == 1
        assert results[0].paper_id == "p2"

    def test_top_k_limits_precomputed_results(self, tmp_path):
        pairs = {
            "p1": [
                {"similar_paper_id": "p2", "similarity_score": 0.9},
                {"similar_paper_id": "p3", "similarity_score": 0.8},
            ],
        }
        engine = self._make_engine(tmp_path, pairs=pairs)

        results = engine.search_similar_papers("p1", top_k=1)

        assert len(results) == 1
        assert results[0].paper_id == "p2"
