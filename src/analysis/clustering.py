"""Topic clustering with UMAP dimensionality reduction and HDBSCAN.

Clusters paper embeddings to identify topic groups and micro-topics.
Uses full_summary embeddings (one per paper) for paper-level clustering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TopicCluster:
    """A topic cluster with member papers."""

    cluster_id: int
    paper_ids: list[str]
    size: int
    representative_titles: list[str]
    label: str = ""


@dataclass
class ClusteringResult:
    """Complete result of topic clustering."""

    n_papers: int
    n_clusters: int
    n_noise: int
    clusters: list[TopicCluster]
    paper_clusters: dict[str, int]
    umap_coords: dict[str, tuple[float, float]]

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "n_papers": self.n_papers,
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "size": c.size,
                    "label": c.label,
                    "representative_titles": c.representative_titles[:5],
                    "paper_ids": c.paper_ids,
                }
                for c in self.clusters
            ],
            "paper_clusters": self.paper_clusters,
        }


def extract_embeddings(
    vector_store,
    chunk_type: str = "full_summary",
) -> tuple[list[str], np.ndarray]:
    """Extract paper embeddings from ChromaDB.

    Args:
        vector_store: VectorStore instance.
        chunk_type: Chunk type to extract (default: full_summary).

    Returns:
        Tuple of (paper_ids, embeddings_matrix).
    """
    results = vector_store.collection.get(
        where={"chunk_type": chunk_type},
        include=["embeddings", "metadatas"],
    )

    if not results["ids"]:
        return [], np.array([])

    paper_ids = []
    embeddings = []

    for i, _chunk_id in enumerate(results["ids"]):
        meta = results["metadatas"][i] if results["metadatas"] is not None else {}
        paper_id = meta.get("paper_id", "")
        embedding = results["embeddings"][i] if results["embeddings"] is not None else None

        if paper_id and embedding is not None:
            paper_ids.append(paper_id)
            embeddings.append(embedding)

    if not embeddings:
        return [], np.array([])

    return paper_ids, np.array(embeddings)


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """Reduce embedding dimensions with UMAP.

    Args:
        embeddings: High-dimensional embedding matrix (n_papers x n_dims).
        n_components: Target dimensions (default: 2 for visualization).
        n_neighbors: UMAP locality parameter.
        min_dist: UMAP minimum distance between points.
        metric: Distance metric.
        random_state: Reproducibility seed.

    Returns:
        Low-dimensional projection (n_papers x n_components).
    """
    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(n_neighbors, len(embeddings) - 1),
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    return reducer.fit_transform(embeddings)


def cluster_papers(
    embeddings_2d: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> np.ndarray:
    """Cluster papers using HDBSCAN on UMAP projections.

    Args:
        embeddings_2d: 2D UMAP projections.
        min_cluster_size: Minimum cluster size for HDBSCAN.
        min_samples: Minimum samples for core points.

    Returns:
        Cluster labels array (-1 = noise).
    """
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min(min_samples, min_cluster_size),
        metric="euclidean",
    )

    return clusterer.fit_predict(embeddings_2d)


def run_clustering(
    vector_store,
    papers_dict: dict | None = None,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    n_neighbors: int = 15,
) -> ClusteringResult:
    """Run the full clustering pipeline.

    Args:
        vector_store: VectorStore instance with paper embeddings.
        papers_dict: Optional dict of paper_id -> paper data for titles.
        min_cluster_size: HDBSCAN minimum cluster size.
        min_samples: HDBSCAN minimum samples.
        n_neighbors: UMAP locality parameter.

    Returns:
        ClusteringResult with cluster assignments and coordinates.
    """
    logger.info("Extracting full_summary embeddings...")
    paper_ids, embeddings = extract_embeddings(vector_store)

    if len(paper_ids) < 5:
        logger.warning(f"Only {len(paper_ids)} papers with embeddings, too few to cluster")
        return ClusteringResult(
            n_papers=len(paper_ids),
            n_clusters=0,
            n_noise=len(paper_ids),
            clusters=[],
            paper_clusters=dict.fromkeys(paper_ids, -1),
            umap_coords={},
        )

    logger.info(f"Reducing {embeddings.shape[1]}D -> 2D with UMAP ({len(paper_ids)} papers)...")
    coords_2d = reduce_dimensions(
        embeddings,
        n_neighbors=n_neighbors,
    )

    logger.info("Clustering with HDBSCAN...")
    labels = cluster_papers(
        coords_2d,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    # Build cluster objects
    paper_clusters = {}
    cluster_members: dict[int, list[str]] = {}

    for i, paper_id in enumerate(paper_ids):
        label = int(labels[i])
        paper_clusters[paper_id] = label
        if label not in cluster_members:
            cluster_members[label] = []
        cluster_members[label].append(paper_id)

    # Build UMAP coordinate map
    umap_coords = {}
    for i, paper_id in enumerate(paper_ids):
        umap_coords[paper_id] = (float(coords_2d[i, 0]), float(coords_2d[i, 1]))

    # Build TopicCluster objects (exclude noise cluster -1)
    clusters = []
    for cluster_id in sorted(cluster_members.keys()):
        if cluster_id == -1:
            continue

        members = cluster_members[cluster_id]
        titles = []
        if papers_dict:
            for pid in members[:5]:
                paper = papers_dict.get(pid, {})
                title = paper.get("title", pid)
                titles.append(title)
        else:
            titles = members[:5]

        clusters.append(TopicCluster(
            cluster_id=cluster_id,
            paper_ids=members,
            size=len(members),
            representative_titles=titles,
        ))

    n_noise = len(cluster_members.get(-1, []))

    result = ClusteringResult(
        n_papers=len(paper_ids),
        n_clusters=len(clusters),
        n_noise=n_noise,
        clusters=clusters,
        paper_clusters=paper_clusters,
        umap_coords=umap_coords,
    )

    logger.info(
        f"Clustering complete: {result.n_clusters} clusters, "
        f"{result.n_noise} noise papers from {result.n_papers} total"
    )
    return result


def save_clustering(result: ClusteringResult, output_dir: Path | str) -> Path:
    """Save clustering results to JSON.

    Args:
        result: ClusteringResult to save.
        output_dir: Directory to save results.

    Returns:
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "topic_clusters.json"
    output_path.write_text(
        json.dumps(result.to_dict(), indent=2),
        encoding="utf-8",
    )

    logger.info(f"Clustering results saved to {output_path}")
    return output_path
