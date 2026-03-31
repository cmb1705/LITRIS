"""Pre-computed topical similarity graph for relationship discovery.

Builds a persistent similarity graph using cosine similarity on paper
centroid embeddings. Unlike citation edges (A cites B), these are
symmetric 'related_to' edges for shared domain/methodology/concepts.

Output: data/index/similarity_graph.json
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from src.utils.file_utils import safe_read_json, safe_write_json

logger = logging.getLogger(__name__)

# Words too common to be meaningful for Jaccard validation
_COMMON_WORDS = {
    "the", "and", "for", "from", "with", "that", "this", "are",
    "was", "were", "been", "have", "has", "had", "not", "but",
    "can", "will", "may", "use", "its", "our", "all", "new",
    "also", "than", "more", "which", "their", "other", "these",
    "some", "such", "into", "they", "between", "most", "about",
}


@dataclass
class SimilarityConfig:
    """Configuration for similarity graph generation."""

    min_similarity: float = 0.75
    max_edges_per_node: int = 10
    use_paper_centroid: bool = True
    batch_size: int = 100


@dataclass
class SimilarityEdge:
    """An edge in the similarity graph."""

    source: str
    target: str
    edge_type: str = "related_to"
    confidence: float = 0.0
    source_type: str = "embedding_similarity"
    jaccard_score: float | None = None


def _cosine_similarity_matrix(centroids: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between all centroid vectors.

    Args:
        centroids: (N, D) array of centroid vectors.

    Returns:
        (N, N) similarity matrix.
    """
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    normalized = centroids / norms
    return normalized @ normalized.T


def _tokenize_text(text: str) -> set[str]:
    """Tokenize text into significant words for Jaccard computation."""
    words = re.findall(r"\b\w{3,}\b", text.lower())
    return {w for w in words if w not in _COMMON_WORDS}


def _compute_jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _get_paper_embeddings(
    vector_store,
    paper_ids: list[str],
    batch_size: int = 100,
) -> dict[str, np.ndarray]:
    """Retrieve embeddings from ChromaDB and compute paper centroids.

    Args:
        vector_store: VectorStore instance with ChromaDB collection.
        paper_ids: List of paper IDs to retrieve.
        batch_size: Number of papers to process per batch.

    Returns:
        Mapping of paper_id to centroid vector (mean of chunk embeddings).
    """
    centroids: dict[str, np.ndarray] = {}

    for i in range(0, len(paper_ids), batch_size):
        batch_ids = paper_ids[i: i + batch_size]

        for paper_id in batch_ids:
            results = vector_store.collection.get(
                where={"paper_id": paper_id},
                include=["embeddings"],
            )

            if not results["ids"] or not results["embeddings"]:
                continue

            embeddings = np.array(results["embeddings"])
            if embeddings.ndim == 1:
                centroids[paper_id] = embeddings
            elif len(embeddings) > 0:
                centroids[paper_id] = embeddings.mean(axis=0)

    return centroids


def _get_paper_texts(
    vector_store,
    paper_ids: list[str],
) -> dict[str, set[str]]:
    """Retrieve document texts from ChromaDB and tokenize for Jaccard.

    Args:
        vector_store: VectorStore instance.
        paper_ids: Papers to retrieve text for.

    Returns:
        Mapping of paper_id to token sets.
    """
    paper_tokens: dict[str, set[str]] = {}

    for paper_id in paper_ids:
        results = vector_store.collection.get(
            where={"paper_id": paper_id},
            include=["documents"],
        )

        if not results["ids"] or not results["documents"]:
            continue

        combined_text = " ".join(
            doc for doc in results["documents"] if doc
        )
        paper_tokens[paper_id] = _tokenize_text(combined_text)

    return paper_tokens


def build_similarity_graph(
    paper_ids: list[str],
    vector_store,
    config: SimilarityConfig | None = None,
    paper_metadata: list[dict] | None = None,
    compute_jaccard: bool = False,
) -> dict:
    """Build a similarity graph from paper embeddings.

    Args:
        paper_ids: List of paper IDs to include.
        vector_store: VectorStore instance with ChromaDB collection.
        config: Optional configuration.
        paper_metadata: Optional list of paper metadata dicts for node info.
        compute_jaccard: Whether to compute Jaccard validation scores.

    Returns:
        Similarity graph dictionary with nodes, edges, and metadata.
    """
    if config is None:
        config = SimilarityConfig()

    logger.info("Building similarity graph for %d papers", len(paper_ids))

    # Step 1: Get paper centroids
    centroids = _get_paper_embeddings(
        vector_store, paper_ids, batch_size=config.batch_size
    )
    logger.info(
        "Computed centroids for %d/%d papers", len(centroids), len(paper_ids)
    )

    if len(centroids) < 2:
        logger.warning("Fewer than 2 papers with embeddings, returning empty graph")
        return _empty_graph(len(paper_ids))

    # Step 2: Compute pairwise cosine similarity
    ordered_ids = sorted(centroids.keys())
    centroid_matrix = np.array([centroids[pid] for pid in ordered_ids])
    sim_matrix = _cosine_similarity_matrix(centroid_matrix)

    # Step 3: Build edges with threshold and cap
    edges: list[SimilarityEdge] = []
    edge_counts: dict[str, int] = defaultdict(int)

    # Collect all candidate edges above threshold
    candidates: list[tuple[float, int, int]] = []
    n = len(ordered_ids)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i, j])
            if score >= config.min_similarity:
                candidates.append((score, i, j))

    # Sort by descending similarity so highest-quality edges are added first
    candidates.sort(key=lambda x: x[0], reverse=True)

    for score, i, j in candidates:
        id_a = ordered_ids[i]
        id_b = ordered_ids[j]

        # Enforce canonical ordering: source < target alphabetically
        source, target = (id_a, id_b) if id_a < id_b else (id_b, id_a)

        # Check cap for both nodes
        if (edge_counts[source] >= config.max_edges_per_node
                or edge_counts[target] >= config.max_edges_per_node):
            continue

        edges.append(SimilarityEdge(
            source=source,
            target=target,
            confidence=round(score, 4),
        ))
        edge_counts[source] += 1
        edge_counts[target] += 1

    logger.info("Found %d similarity edges above threshold %.2f",
                len(edges), config.min_similarity)

    # Step 4: Optional Jaccard validation
    if compute_jaccard and edges:
        # Get paper IDs that have edges
        edge_paper_ids = set()
        for e in edges:
            edge_paper_ids.add(e.source)
            edge_paper_ids.add(e.target)

        paper_tokens = _get_paper_texts(vector_store, list(edge_paper_ids))

        for edge in edges:
            tokens_a = paper_tokens.get(edge.source, set())
            tokens_b = paper_tokens.get(edge.target, set())
            edge.jaccard_score = round(
                _compute_jaccard(tokens_a, tokens_b), 4
            )

    # Step 5: Build metadata lookup
    meta_lookup: dict[str, dict] = {}
    if paper_metadata:
        for pm in paper_metadata:
            pid = pm.get("paper_id")
            if pid:
                meta_lookup[pid] = pm

    # Step 6: Build output
    nodes = []
    for pid in ordered_ids:
        pm = meta_lookup.get(pid, {})
        title = pm.get("title") or pid
        nodes.append({
            "id": pid,
            "type": "paper",
            "label": title[:50] + ("..." if len(title) > 50 else ""),
            "title": title,
            "authors": pm.get("authors") or "",
            "year": pm.get("publication_year"),
            "collections": list(pm.get("collections") or []),
            "in_library": True,
            "edge_count": edge_counts.get(pid, 0),
        })

    edge_dicts = []
    for e in edges:
        d = {
            "source": e.source,
            "target": e.target,
            "type": e.edge_type,
            "confidence": e.confidence,
            "source_type": e.source_type,
        }
        if e.jaccard_score is not None:
            d["jaccard_score"] = e.jaccard_score
        edge_dicts.append(d)

    graph = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "papers_analyzed": len(paper_ids),
            "papers_with_embeddings": len(centroids),
            "config": {
                "min_similarity": config.min_similarity,
                "max_edges_per_node": config.max_edges_per_node,
                "use_paper_centroid": config.use_paper_centroid,
            },
        },
        "nodes": nodes,
        "edges": edge_dicts,
    }

    return graph


def _empty_graph(papers_analyzed: int = 0) -> dict:
    """Return an empty graph structure."""
    return {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "node_count": 0,
            "edge_count": 0,
            "papers_analyzed": papers_analyzed,
            "papers_with_embeddings": 0,
            "config": {},
        },
        "nodes": [],
        "edges": [],
    }


def save_similarity_graph(graph: dict, output_path: Path) -> Path:
    """Save similarity graph to JSON file.

    Args:
        graph: Similarity graph dictionary.
        output_path: Path to save the graph.

    Returns:
        Path to the saved file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_write_json(output_path, graph)
    logger.info(
        "Saved similarity graph: %d nodes, %d edges -> %s",
        graph["metadata"]["node_count"],
        graph["metadata"]["edge_count"],
        output_path,
    )
    return output_path


def load_and_build_similarity_graph(
    index_dir: Path,
    config: SimilarityConfig | None = None,
    compute_jaccard: bool = False,
) -> dict:
    """Load index data and build similarity graph.

    Args:
        index_dir: Path to the index directory (containing chroma/ and papers.json).
        config: Optional configuration.
        compute_jaccard: Whether to compute Jaccard validation.

    Returns:
        Similarity graph dictionary.
    """
    from src.indexing.vector_store import VectorStore

    # Load paper IDs from papers.json
    papers_data = safe_read_json(index_dir / "papers.json", default={"papers": []})
    if isinstance(papers_data, dict) and "papers" in papers_data:
        papers = papers_data["papers"]
    elif isinstance(papers_data, list):
        papers = papers_data
    else:
        papers = []

    paper_ids = [p.get("paper_id") for p in papers if p.get("paper_id")]

    if not paper_ids:
        logger.warning("No papers found in %s", index_dir / "papers.json")
        return _empty_graph()

    # Open ChromaDB
    chroma_dir = index_dir / "chroma"
    if not chroma_dir.exists():
        logger.warning("ChromaDB directory not found: %s", chroma_dir)
        return _empty_graph(len(paper_ids))

    vector_store = VectorStore(persist_directory=chroma_dir)
    try:
        return build_similarity_graph(
            paper_ids=paper_ids,
            vector_store=vector_store,
            config=config,
            paper_metadata=papers,
            compute_jaccard=compute_jaccard,
        )
    finally:
        vector_store.close()
