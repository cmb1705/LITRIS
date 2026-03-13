"""Tests for topic clustering with UMAP + HDBSCAN."""

import pytest

umap = pytest.importorskip("umap", reason="umap-learn not installed")
hdbscan = pytest.importorskip("hdbscan", reason="hdbscan not installed")

from unittest.mock import MagicMock  # noqa: E402

import numpy as np  # noqa: E402

from src.analysis.clustering import (  # noqa: E402
    ClusteringResult,
    TopicCluster,
    cluster_papers,
    extract_embeddings,
    reduce_dimensions,
    run_clustering,
    save_clustering,
)


class TestExtractEmbeddings:
    """Tests for embedding extraction from ChromaDB."""

    def test_extracts_embeddings(self):
        mock_store = MagicMock()
        mock_store.collection.get.return_value = {
            "ids": ["c1", "c2"],
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "metadatas": [
                {"paper_id": "p1", "chunk_type": "raptor_overview"},
                {"paper_id": "p2", "chunk_type": "raptor_overview"},
            ],
        }

        paper_ids, embeddings = extract_embeddings(mock_store)

        assert paper_ids == ["p1", "p2"]
        assert embeddings.shape == (2, 3)
        np.testing.assert_array_almost_equal(embeddings[0], [0.1, 0.2, 0.3])

    def test_empty_collection(self):
        mock_store = MagicMock()
        mock_store.collection.get.return_value = {
            "ids": [],
            "embeddings": [],
            "metadatas": [],
        }

        paper_ids, embeddings = extract_embeddings(mock_store)

        assert paper_ids == []
        assert embeddings.size == 0

    def test_filters_by_chunk_type(self):
        mock_store = MagicMock()
        mock_store.collection.get.return_value = {
            "ids": ["c1"],
            "embeddings": [[0.1, 0.2]],
            "metadatas": [{"paper_id": "p1"}],
        }

        extract_embeddings(mock_store, chunk_type="dim_q02")

        call_kwargs = mock_store.collection.get.call_args
        assert call_kwargs[1]["where"] == {"chunk_type": "dim_q02"}


class TestReduceDimensions:
    """Tests for UMAP dimensionality reduction."""

    def test_reduces_to_2d(self):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(20, 50)

        result = reduce_dimensions(embeddings, n_components=2)

        assert result.shape == (20, 2)

    def test_respects_n_components(self):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(20, 50)

        result = reduce_dimensions(embeddings, n_components=3)

        assert result.shape == (20, 3)

    def test_handles_small_dataset(self):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(6, 10)

        result = reduce_dimensions(embeddings, n_neighbors=5)

        assert result.shape == (6, 2)


class TestClusterPapers:
    """Tests for HDBSCAN clustering."""

    def test_finds_clusters(self):
        # Create two clear clusters
        rng = np.random.RandomState(42)
        cluster_1 = rng.randn(15, 2) + np.array([5, 5])
        cluster_2 = rng.randn(15, 2) + np.array([-5, -5])
        embeddings_2d = np.vstack([cluster_1, cluster_2])

        labels = cluster_papers(embeddings_2d, min_cluster_size=5, min_samples=3)

        assert len(labels) == 30
        # Should find at least 2 clusters
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        assert len(unique_labels) >= 2

    def test_noise_label_for_outliers(self):
        # Sparse scattered points should be noise
        rng = np.random.RandomState(42)
        embeddings_2d = rng.randn(10, 2) * 100

        labels = cluster_papers(embeddings_2d, min_cluster_size=5)

        # Most or all should be noise (-1)
        assert -1 in labels


class TestRunClustering:
    """Tests for the full clustering pipeline."""

    def test_full_pipeline(self):
        rng = np.random.RandomState(42)
        n_papers = 30
        n_dims = 50

        # Create embeddings with two clear groups
        group_1 = rng.randn(15, n_dims) + 3
        group_2 = rng.randn(15, n_dims) - 3
        embeddings = np.vstack([group_1, group_2]).tolist()

        paper_ids = [f"p{i}" for i in range(n_papers)]
        papers_dict = {f"p{i}": {"title": f"Paper {i}"} for i in range(n_papers)}

        mock_store = MagicMock()
        mock_store.collection.get.return_value = {
            "ids": [f"c{i}" for i in range(n_papers)],
            "embeddings": embeddings,
            "metadatas": [{"paper_id": pid} for pid in paper_ids],
        }

        result = run_clustering(mock_store, papers_dict=papers_dict, min_cluster_size=5)

        assert isinstance(result, ClusteringResult)
        assert result.n_papers == 30
        assert result.n_clusters >= 1
        assert len(result.paper_clusters) == 30
        assert len(result.umap_coords) == 30

        # Check UMAP coords are 2D tuples
        for coord in result.umap_coords.values():
            assert len(coord) == 2

    def test_too_few_papers(self):
        mock_store = MagicMock()
        mock_store.collection.get.return_value = {
            "ids": ["c1", "c2"],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "metadatas": [{"paper_id": "p1"}, {"paper_id": "p2"}],
        }

        result = run_clustering(mock_store)

        assert result.n_clusters == 0
        assert result.n_papers == 2

    def test_to_dict_serializable(self):
        result = ClusteringResult(
            n_papers=10,
            n_clusters=2,
            n_noise=1,
            clusters=[
                TopicCluster(
                    cluster_id=0,
                    paper_ids=["p1", "p2"],
                    size=2,
                    representative_titles=["Paper 1", "Paper 2"],
                ),
            ],
            paper_clusters={"p1": 0, "p2": 0, "p3": -1},
            umap_coords={"p1": (1.0, 2.0), "p2": (1.5, 2.5)},
        )

        d = result.to_dict()

        assert d["n_papers"] == 10
        assert d["n_clusters"] == 2
        assert len(d["clusters"]) == 1
        assert d["clusters"][0]["size"] == 2


class TestSaveClustering:
    """Tests for saving clustering results."""

    def test_saves_to_json(self, tmp_path):
        result = ClusteringResult(
            n_papers=5,
            n_clusters=1,
            n_noise=0,
            clusters=[
                TopicCluster(
                    cluster_id=0,
                    paper_ids=["p1"],
                    size=1,
                    representative_titles=["Paper 1"],
                ),
            ],
            paper_clusters={"p1": 0},
            umap_coords={"p1": (1.0, 2.0)},
        )

        path = save_clustering(result, tmp_path)

        assert path.exists()
        assert path.name == "topic_clusters.json"

        import json

        data = json.loads(path.read_text())
        assert data["n_clusters"] == 1
