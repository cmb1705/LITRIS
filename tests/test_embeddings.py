"""Tests for embedding generation and vector store."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.analysis.schemas import (
    KeyClaim,
    KeyFinding,
    Methodology,
    PaperExtraction,
)
from src.indexing.embeddings import (
    CHUNK_TYPES,
    EmbeddingChunk,
    EmbeddingGenerator,
)
from src.indexing.vector_store import SearchResult, VectorStore
from src.zotero.models import PaperMetadata


class TestEmbeddingChunk:
    """Tests for EmbeddingChunk dataclass."""

    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = EmbeddingChunk(
            paper_id="paper_001",
            chunk_id="paper_001_abstract",
            chunk_type="abstract",
            text="This is the abstract text.",
            metadata={"title": "Test Paper"},
        )
        assert chunk.paper_id == "paper_001"
        assert chunk.chunk_type == "abstract"
        assert chunk.embedding == []

    def test_chunk_to_dict(self):
        """Test conversion to dictionary."""
        chunk = EmbeddingChunk(
            paper_id="paper_001",
            chunk_id="paper_001_thesis",
            chunk_type="thesis",
            text="Main thesis statement.",
            metadata={"year": 2024},
        )
        d = chunk.to_dict()
        assert d["paper_id"] == "paper_001"
        assert d["chunk_type"] == "thesis"
        assert "text" in d
        assert "embedding" not in d  # Embedding not included in to_dict


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator class."""

    @pytest.fixture
    def mock_model(self):
        """Create mock sentence transformer model."""
        with patch("src.indexing.embeddings.SentenceTransformer") as mock:
            model = MagicMock()
            model.get_sentence_embedding_dimension.return_value = 384
            model.encode.return_value = [[0.1] * 384]
            mock.return_value = model
            yield mock

    @pytest.fixture
    def sample_paper(self):
        """Create sample paper metadata."""
        from datetime import datetime
        return PaperMetadata(
            zotero_key="ABC123",
            zotero_item_id=12345,
            title="Sample Research Paper",
            abstract="This is a sample abstract for testing.",
            item_type="journalArticle",
            publication_year=2024,
            authors=[{"first_name": "John", "last_name": "Doe"}],
            collections=["Research", "ML"],
            pdf_path=Path("sample.pdf"),
            date_added=datetime(2024, 1, 1),
            date_modified=datetime(2024, 1, 2),
        )

    @pytest.fixture
    def sample_extraction(self):
        """Create sample paper extraction."""
        return PaperExtraction(
            thesis_statement="This paper presents a novel approach.",
            research_questions=["RQ1: What is the impact?"],
            methodology=Methodology(
                approach="Quantitative",
                design="Experimental",
                data_sources=["Survey data"],
                analysis_methods=["Regression"],
            ),
            key_findings=[
                KeyFinding(finding="Finding 1", evidence_type="quantitative"),
            ],
            key_claims=[
                KeyClaim(claim="Claim 1", evidence_strength="Strong"),
            ],
            limitations=["Limited sample size"],
            future_directions=["Expand to more domains"],
            conclusions="The results support our hypothesis.",
            contribution_summary="Novel contribution to the field.",
            discipline_tags=["Machine Learning"],
        )

    def test_generator_initialization(self, mock_model):
        """Test generator initialization."""
        gen = EmbeddingGenerator(model_name="test-model")
        assert gen.model_name == "test-model"
        assert gen.embedding_dim == 384
        mock_model.assert_called_once()

    def test_create_chunks_basic(self, mock_model, sample_paper, sample_extraction):
        """Test basic chunk creation."""
        gen = EmbeddingGenerator()
        chunks = gen.create_chunks(sample_paper, sample_extraction)

        assert len(chunks) > 0
        chunk_types = [c.chunk_type for c in chunks]
        assert "abstract" in chunk_types
        assert "thesis" in chunk_types
        assert "contribution" in chunk_types

    def test_create_chunks_all_types(self, mock_model, sample_paper, sample_extraction):
        """Test that all expected chunk types are created."""
        gen = EmbeddingGenerator()
        chunks = gen.create_chunks(sample_paper, sample_extraction)

        chunk_types = {c.chunk_type for c in chunks}
        # Should have most chunk types
        assert "abstract" in chunk_types
        assert "thesis" in chunk_types
        assert "methodology" in chunk_types
        assert "findings" in chunk_types
        assert "claims" in chunk_types
        assert "limitations" in chunk_types
        assert "future_work" in chunk_types
        assert "full_summary" in chunk_types

    def test_create_chunks_metadata(self, mock_model, sample_paper, sample_extraction):
        """Test that chunks have correct metadata."""
        gen = EmbeddingGenerator()
        chunks = gen.create_chunks(sample_paper, sample_extraction)

        for chunk in chunks:
            assert chunk.paper_id == sample_paper.paper_id
            assert chunk.metadata.get("title") == sample_paper.title
            assert chunk.metadata.get("year") == sample_paper.publication_year

    def test_truncate_text_short(self, mock_model):
        """Test that short text is not truncated."""
        gen = EmbeddingGenerator(max_chunk_tokens=512)
        text = "Short text that fits."
        result = gen._truncate_text(text)
        assert result == text

    def test_truncate_text_long(self, mock_model):
        """Test that long text is truncated."""
        gen = EmbeddingGenerator(max_chunk_tokens=50)  # ~200 chars
        text = "A " * 500  # Much longer
        result = gen._truncate_text(text)
        assert len(result) < len(text)
        assert result.endswith("...")

    def test_generate_embeddings(self, mock_model):
        """Test embedding generation."""
        import numpy as np

        mock_model.return_value.encode.return_value = np.array([
            [0.1] * 384,
            [0.2] * 384,
        ])

        gen = EmbeddingGenerator()
        chunks = [
            EmbeddingChunk(
                paper_id="p1", chunk_id="c1", chunk_type="abstract", text="Text 1"
            ),
            EmbeddingChunk(
                paper_id="p1", chunk_id="c2", chunk_type="thesis", text="Text 2"
            ),
        ]

        result = gen.generate_embeddings(chunks, show_progress=False)
        assert len(result) == 2
        assert len(result[0].embedding) == 384
        assert len(result[1].embedding) == 384

    def test_embed_text(self, mock_model):
        """Test single text embedding."""
        import numpy as np

        mock_model.return_value.encode.return_value = np.array([0.5] * 384)

        gen = EmbeddingGenerator()
        embedding = gen.embed_text("Test text")

        assert len(embedding) == 384
        mock_model.return_value.encode.assert_called()

    def test_embed_batch(self, mock_model):
        """Test batch text embedding."""
        import numpy as np

        mock_model.return_value.encode.return_value = np.array([
            [0.1] * 384,
            [0.2] * 384,
            [0.3] * 384,
        ])

        gen = EmbeddingGenerator()
        embeddings = gen.embed_batch(["Text 1", "Text 2", "Text 3"])

        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)

    def test_empty_chunks(self, mock_model):
        """Test handling of empty chunk list."""
        gen = EmbeddingGenerator()
        result = gen.generate_embeddings([])
        assert result == []


class TestVectorStore:
    """Tests for VectorStore class."""

    @pytest.fixture
    def vector_store(self, tmp_path):
        """Create a vector store in temp directory."""
        return VectorStore(persist_directory=tmp_path / "chroma")

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks with embeddings."""
        return [
            EmbeddingChunk(
                paper_id="paper_001",
                chunk_id="paper_001_abstract",
                chunk_type="abstract",
                text="Machine learning approaches for text classification.",
                embedding=[0.1] * 384,
                metadata={"title": "ML Paper", "year": "2024", "item_type": "journalArticle"},
            ),
            EmbeddingChunk(
                paper_id="paper_001",
                chunk_id="paper_001_thesis",
                chunk_type="thesis",
                text="Novel attention mechanisms improve performance.",
                embedding=[0.2] * 384,
                metadata={"title": "ML Paper", "year": "2024", "item_type": "journalArticle"},
            ),
            EmbeddingChunk(
                paper_id="paper_002",
                chunk_id="paper_002_abstract",
                chunk_type="abstract",
                text="Deep learning for image recognition.",
                embedding=[0.3] * 384,
                metadata={"title": "DL Paper", "year": "2023", "item_type": "conferencePaper"},
            ),
        ]

    def test_store_initialization(self, vector_store):
        """Test store initialization."""
        assert vector_store.count() == 0
        assert vector_store.collection is not None

    def test_add_chunks(self, vector_store, sample_chunks):
        """Test adding chunks to store."""
        added = vector_store.add_chunks(sample_chunks)
        assert added == 3
        assert vector_store.count() == 3

    def test_add_chunks_without_embeddings(self, vector_store):
        """Test that chunks without embeddings are skipped."""
        chunks = [
            EmbeddingChunk(
                paper_id="p1",
                chunk_id="c1",
                chunk_type="abstract",
                text="Text",
                embedding=[],  # Empty embedding
            ),
        ]
        added = vector_store.add_chunks(chunks)
        assert added == 0

    def test_search_basic(self, vector_store, sample_chunks):
        """Test basic search."""
        vector_store.add_chunks(sample_chunks)

        results = vector_store.search(
            query_embedding=[0.1] * 384,
            top_k=5,
        )

        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
        assert results[0].score > 0

    def test_search_filter_chunk_types(self, vector_store, sample_chunks):
        """Test search with chunk type filter."""
        vector_store.add_chunks(sample_chunks)

        results = vector_store.search(
            query_embedding=[0.1] * 384,
            top_k=5,
            chunk_types=["abstract"],
        )

        assert all(r.chunk_type == "abstract" for r in results)

    def test_search_filter_item_types(self, vector_store, sample_chunks):
        """Test search with item type filter."""
        vector_store.add_chunks(sample_chunks)

        results = vector_store.search(
            query_embedding=[0.1] * 384,
            top_k=5,
            item_types=["journalArticle"],
        )

        for r in results:
            assert r.metadata.get("item_type") == "journalArticle"

    def test_search_year_filter_operands_are_numeric(self, vector_store, monkeypatch):
        """Ensure year filter operands stay numeric for Chroma queries."""
        captured = {}

        def fake_query(**kwargs):
            captured["where"] = kwargs.get("where")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        monkeypatch.setattr(vector_store.collection, "query", fake_query)

        vector_store.search(
            query_embedding=[0.1] * 384,
            top_k=5,
            year_min=1945,
            year_max=2025,
        )

        where = captured.get("where")
        assert where is not None
        assert where["$and"][0]["year"]["$gte"] == 1945
        assert isinstance(where["$and"][0]["year"]["$gte"], int)
        assert where["$and"][1]["year"]["$lte"] == 2025
        assert isinstance(where["$and"][1]["year"]["$lte"], int)

    def test_delete_paper(self, vector_store, sample_chunks):
        """Test deleting a paper's chunks."""
        vector_store.add_chunks(sample_chunks)
        assert vector_store.count() == 3

        deleted = vector_store.delete_paper("paper_001")
        assert deleted == 2
        assert vector_store.count() == 1

    def test_get_paper_chunks(self, vector_store, sample_chunks):
        """Test retrieving chunks for a paper."""
        vector_store.add_chunks(sample_chunks)

        chunks = vector_store.get_paper_chunks("paper_001")
        assert len(chunks) == 2
        assert all(c["metadata"]["paper_id"] == "paper_001" for c in chunks)

    def test_get_stats(self, vector_store, sample_chunks):
        """Test getting store statistics."""
        vector_store.add_chunks(sample_chunks)

        stats = vector_store.get_stats()
        assert stats["total_chunks"] == 3
        assert stats["unique_papers"] == 2
        assert "chunk_type_distribution" in stats

    def test_clear(self, vector_store, sample_chunks):
        """Test clearing the store."""
        vector_store.add_chunks(sample_chunks)
        assert vector_store.count() == 3

        cleared = vector_store.clear()
        assert cleared == 3
        assert vector_store.count() == 0


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_result_creation(self):
        """Test result creation."""
        result = SearchResult(
            paper_id="p1",
            chunk_id="c1",
            chunk_type="abstract",
            text="Sample text",
            score=0.95,
            metadata={"title": "Test"},
        )
        assert result.paper_id == "p1"
        assert result.score == 0.95

    def test_result_to_dict(self):
        """Test conversion to dictionary."""
        result = SearchResult(
            paper_id="p1",
            chunk_id="c1",
            chunk_type="thesis",
            text="Thesis text",
            score=0.85,
            metadata={"year": 2024},
        )
        d = result.to_dict()
        assert d["paper_id"] == "p1"
        assert d["score"] == 0.85
        assert d["metadata"]["year"] == 2024


class TestChunkTypes:
    """Tests for chunk type definitions."""

    def test_chunk_types_defined(self):
        """Test that all chunk types are defined."""
        expected_types = [
            "abstract",
            "thesis",
            "contribution",
            "methodology",
            "findings",
            "claims",
            "limitations",
            "future_work",
            "full_summary",
        ]
        assert set(CHUNK_TYPES) == set(expected_types)
