"""Tests for the similarity graph module."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.analysis.similarity_graph import (
    SimilarityConfig,
    _compute_jaccard,
    _cosine_similarity_matrix,
    _tokenize_text,
    build_similarity_graph,
)


def _make_mock_vector_store(paper_embeddings: dict[str, list[list[float]]],
                             paper_documents: dict[str, list[str]] | None = None):
    """Create a mock VectorStore with controlled embeddings.

    Args:
        paper_embeddings: Mapping of paper_id to list of chunk embedding vectors.
        paper_documents: Optional mapping of paper_id to list of chunk texts.
    """
    store = MagicMock()

    def mock_get(where=None, include=None):
        paper_id = where.get("paper_id") if where else None
        result = {"ids": [], "embeddings": None, "documents": None}

        if paper_id and paper_id in paper_embeddings:
            embs = paper_embeddings[paper_id]
            n = len(embs)
            result["ids"] = [f"{paper_id}_chunk_{i}" for i in range(n)]

            if include and "embeddings" in include:
                result["embeddings"] = embs
            if include and "documents" in include:
                if paper_documents and paper_id in paper_documents:
                    result["documents"] = paper_documents[paper_id]
                else:
                    result["documents"] = ["text"] * n

        return result

    store.collection.get = mock_get
    return store


class TestBasicSimilarityGraph:
    """test_basic_similarity_graph -- mock 5 papers, verify structure."""

    def test_graph_structure(self):
        # 5 papers with 2D embeddings. Papers 0-2 are similar, 3-4 are similar.
        embs = {
            "paper_a": [[1.0, 0.0]],
            "paper_b": [[0.95, 0.05]],
            "paper_c": [[0.9, 0.1]],
            "paper_d": [[0.0, 1.0]],
            "paper_e": [[0.05, 0.95]],
        }
        store = _make_mock_vector_store(embs)
        config = SimilarityConfig(min_similarity=0.9, max_edges_per_node=10)

        graph = build_similarity_graph(
            paper_ids=list(embs.keys()),
            vector_store=store,
            config=config,
        )

        assert "metadata" in graph
        assert "nodes" in graph
        assert "edges" in graph
        assert graph["metadata"]["node_count"] == 5
        assert graph["metadata"]["papers_analyzed"] == 5
        assert graph["metadata"]["papers_with_embeddings"] == 5
        assert len(graph["nodes"]) == 5
        # At minimum, similar papers should have edges
        assert len(graph["edges"]) > 0

        # Verify edge schema
        for edge in graph["edges"]:
            assert "source" in edge
            assert "target" in edge
            assert edge["type"] == "related_to"
            assert edge["source_type"] == "embedding_similarity"
            assert isinstance(edge["confidence"], float)
            assert edge["confidence"] >= 0.9


class TestThresholdFiltering:
    """test_threshold_filtering -- below min_similarity -> no edge."""

    def test_orthogonal_papers_no_edges(self):
        # Two completely orthogonal papers
        embs = {
            "paper_x": [[1.0, 0.0]],
            "paper_y": [[0.0, 1.0]],
        }
        store = _make_mock_vector_store(embs)
        config = SimilarityConfig(min_similarity=0.9)

        graph = build_similarity_graph(
            paper_ids=list(embs.keys()),
            vector_store=store,
            config=config,
        )

        assert graph["metadata"]["edge_count"] == 0
        assert len(graph["edges"]) == 0

    def test_low_threshold_creates_edges(self):
        embs = {
            "paper_x": [[1.0, 0.0]],
            "paper_y": [[0.7, 0.7]],
        }
        store = _make_mock_vector_store(embs)
        config = SimilarityConfig(min_similarity=0.5)

        graph = build_similarity_graph(
            paper_ids=list(embs.keys()),
            vector_store=store,
            config=config,
        )

        assert graph["metadata"]["edge_count"] > 0


class TestMaxEdgesPerNode:
    """test_max_edges_per_node -- cap enforced."""

    def test_cap_limits_edges(self):
        # Create 6 papers all very similar (all point roughly same direction)
        embs = {}
        for i in range(6):
            angle = i * 0.01  # Very slight variation
            embs[f"paper_{i}"] = [[np.cos(angle), np.sin(angle)]]

        store = _make_mock_vector_store(embs)
        config = SimilarityConfig(min_similarity=0.9, max_edges_per_node=2)

        graph = build_similarity_graph(
            paper_ids=list(embs.keys()),
            vector_store=store,
            config=config,
        )

        # Count edges per node
        edge_counts: dict[str, int] = {}
        for edge in graph["edges"]:
            edge_counts[edge["source"]] = edge_counts.get(edge["source"], 0) + 1
            edge_counts[edge["target"]] = edge_counts.get(edge["target"], 0) + 1

        # No node should exceed the cap
        for pid, count in edge_counts.items():
            assert count <= 2, f"{pid} has {count} edges, expected <= 2"


class TestSymmetricEdges:
    """test_symmetric_edges -- stored once, source < target."""

    def test_canonical_ordering(self):
        embs = {
            "paper_z": [[1.0, 0.0]],
            "paper_a": [[0.99, 0.01]],
        }
        store = _make_mock_vector_store(embs)
        config = SimilarityConfig(min_similarity=0.9)

        graph = build_similarity_graph(
            paper_ids=list(embs.keys()),
            vector_store=store,
            config=config,
        )

        assert len(graph["edges"]) == 1
        edge = graph["edges"][0]
        # Source should be alphabetically before target
        assert edge["source"] < edge["target"]
        assert edge["source"] == "paper_a"
        assert edge["target"] == "paper_z"


class TestEmptyCorpus:
    """test_empty_corpus -- 0 papers -> empty graph."""

    def test_no_papers(self):
        store = _make_mock_vector_store({})
        graph = build_similarity_graph(
            paper_ids=[],
            vector_store=store,
        )

        assert graph["metadata"]["node_count"] == 0
        assert graph["metadata"]["edge_count"] == 0
        assert graph["nodes"] == []
        assert graph["edges"] == []

    def test_single_paper(self):
        embs = {"paper_only": [[1.0, 0.0]]}
        store = _make_mock_vector_store(embs)

        graph = build_similarity_graph(
            paper_ids=["paper_only"],
            vector_store=store,
        )

        assert graph["metadata"]["edge_count"] == 0


class TestPaperCentroidComputation:
    """test_paper_centroid_computation -- 3 chunks -> single centroid."""

    def test_centroid_is_mean(self):
        # Paper with 3 chunks; centroid should be mean
        embs = {
            "paper_multi": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        }
        store = _make_mock_vector_store(embs)

        # Create a second paper near the expected centroid [0.333, 0.333, 0.333]
        centroid_approx = [0.34, 0.33, 0.33]
        norm = np.linalg.norm(centroid_approx)
        near_centroid = [x / norm for x in centroid_approx]
        embs["paper_near"] = [near_centroid]
        store = _make_mock_vector_store(embs)

        config = SimilarityConfig(min_similarity=0.9)
        graph = build_similarity_graph(
            paper_ids=list(embs.keys()),
            vector_store=store,
            config=config,
        )

        # The multi-chunk paper's centroid is [1/3, 1/3, 1/3], which
        # is very similar to paper_near (normalized ~[1/3, 1/3, 1/3])
        assert graph["metadata"]["edge_count"] >= 1


class TestDeterministicOutput:
    """test_deterministic_output -- same inputs same graph."""

    def test_same_inputs_same_graph(self):
        embs = {
            "paper_a": [[1.0, 0.0]],
            "paper_b": [[0.98, 0.02]],
            "paper_c": [[0.0, 1.0]],
        }
        store1 = _make_mock_vector_store(embs)
        store2 = _make_mock_vector_store(embs)
        config = SimilarityConfig(min_similarity=0.9)

        graph1 = build_similarity_graph(
            paper_ids=sorted(embs.keys()),
            vector_store=store1,
            config=config,
        )
        graph2 = build_similarity_graph(
            paper_ids=sorted(embs.keys()),
            vector_store=store2,
            config=config,
        )

        # Edges should be identical (ignore timestamps in metadata)
        assert graph1["edges"] == graph2["edges"]
        assert graph1["nodes"] == graph2["nodes"]
        assert graph1["metadata"]["node_count"] == graph2["metadata"]["node_count"]
        assert graph1["metadata"]["edge_count"] == graph2["metadata"]["edge_count"]


class TestJaccardValidation:
    """Test the optional Jaccard validation layer."""

    def test_jaccard_scores_computed(self):
        embs = {
            "paper_a": [[1.0, 0.0]],
            "paper_b": [[0.99, 0.01]],
        }
        docs = {
            "paper_a": ["network analysis graph theory community detection"],
            "paper_b": ["network analysis centrality metrics graph algorithms"],
        }
        store = _make_mock_vector_store(embs, paper_documents=docs)
        config = SimilarityConfig(min_similarity=0.9)

        graph = build_similarity_graph(
            paper_ids=list(embs.keys()),
            vector_store=store,
            config=config,
            compute_jaccard=True,
        )

        assert len(graph["edges"]) == 1
        assert "jaccard_score" in graph["edges"][0]
        assert graph["edges"][0]["jaccard_score"] > 0.0

    def test_jaccard_not_computed_by_default(self):
        embs = {
            "paper_a": [[1.0, 0.0]],
            "paper_b": [[0.99, 0.01]],
        }
        store = _make_mock_vector_store(embs)
        config = SimilarityConfig(min_similarity=0.9)

        graph = build_similarity_graph(
            paper_ids=list(embs.keys()),
            vector_store=store,
            config=config,
        )

        assert len(graph["edges"]) == 1
        assert "jaccard_score" not in graph["edges"][0]


class TestHelperFunctions:
    """Test internal helper functions."""

    def test_cosine_similarity_matrix_identical(self):
        vecs = np.array([[1.0, 0.0], [1.0, 0.0]])
        sim = _cosine_similarity_matrix(vecs)
        assert sim[0, 1] == pytest.approx(1.0, abs=1e-6)

    def test_cosine_similarity_matrix_orthogonal(self):
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = _cosine_similarity_matrix(vecs)
        assert sim[0, 1] == pytest.approx(0.0, abs=1e-6)

    def test_tokenize_text_filters_common_words(self):
        tokens = _tokenize_text("the network analysis for graph theory")
        assert "the" not in tokens
        assert "for" not in tokens
        assert "network" in tokens
        assert "analysis" in tokens

    def test_compute_jaccard_identical(self):
        s = {"network", "analysis", "graph"}
        assert _compute_jaccard(s, s) == pytest.approx(1.0)

    def test_compute_jaccard_disjoint(self):
        a = {"network", "analysis"}
        b = {"biology", "genetics"}
        assert _compute_jaccard(a, b) == pytest.approx(0.0)

    def test_compute_jaccard_empty(self):
        assert _compute_jaccard(set(), {"network"}) == pytest.approx(0.0)


class TestPaperMetadata:
    """Test that paper metadata is correctly included in nodes."""

    def test_metadata_in_nodes(self):
        embs = {
            "paper_a": [[1.0, 0.0]],
            "paper_b": [[0.99, 0.01]],
        }
        metadata = [
            {
                "paper_id": "paper_a",
                "title": "Network Analysis Methods",
                "authors": "Smith, J.",
                "publication_year": 2023,
                "collections": ["methods"],
            },
            {
                "paper_id": "paper_b",
                "title": "Graph Theory Applications",
                "authors": "Jones, K.",
                "publication_year": 2024,
                "collections": ["theory"],
            },
        ]
        store = _make_mock_vector_store(embs)
        config = SimilarityConfig(min_similarity=0.9)

        graph = build_similarity_graph(
            paper_ids=list(embs.keys()),
            vector_store=store,
            config=config,
            paper_metadata=metadata,
        )

        node_map = {n["id"]: n for n in graph["nodes"]}
        assert node_map["paper_a"]["title"] == "Network Analysis Methods"
        assert node_map["paper_a"]["authors"] == "Smith, J."
        assert node_map["paper_a"]["year"] == 2023
        assert node_map["paper_b"]["collections"] == ["theory"]
