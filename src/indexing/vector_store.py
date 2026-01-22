"""Vector store using ChromaDB for semantic search."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from src.indexing.embeddings import CHUNK_TYPES, ChunkType, EmbeddingChunk
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "paper_chunks"


@dataclass
class SearchResult:
    """A single search result."""

    paper_id: str
    chunk_id: str
    chunk_type: ChunkType
    text: str
    score: float
    metadata: dict

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "paper_id": self.paper_id,
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }


class VectorStore:
    """ChromaDB-based vector store for paper embeddings."""

    def __init__(
        self,
        persist_directory: Path | str,
        collection_name: str = COLLECTION_NAME,
    ):
        """Initialize vector store.

        Args:
            persist_directory: Directory for persistent storage.
            collection_name: Name of the ChromaDB collection.
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        logger.info(f"Initializing ChromaDB at {self.persist_directory}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self._get_or_create_collection()
        logger.info(f"Collection '{collection_name}' ready with {self.collection.count()} documents")

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create new one."""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Paper chunk embeddings for semantic search"},
        )

    def add_chunks(
        self,
        chunks: list[EmbeddingChunk],
        batch_size: int = 100,
    ) -> int:
        """Add chunks to the vector store.

        Args:
            chunks: List of chunks with embeddings.
            batch_size: Batch size for insertion.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        # Filter out chunks without embeddings
        valid_chunks = [c for c in chunks if c.embedding]
        if len(valid_chunks) < len(chunks):
            logger.warning(
                f"Skipping {len(chunks) - len(valid_chunks)} chunks without embeddings"
            )

        if not valid_chunks:
            return 0

        # Validate embedding dimensions on first chunk
        first_embedding = valid_chunks[0].embedding
        if first_embedding:
            dim = len(first_embedding)
            # Check if collection already has embeddings with different dimension
            existing_count = self.collection.count()
            if existing_count > 0:
                # ChromaDB will error if dimensions mismatch, but log for clarity
                logger.debug(f"Adding embeddings with dimension {dim} to collection with {existing_count} existing chunks")
            else:
                logger.info(f"Initializing collection with embedding dimension {dim}")

        logger.info(f"Adding {len(valid_chunks)} chunks to vector store")

        # Process in batches
        added = 0
        for i in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[i : i + batch_size]

            ids = [c.chunk_id for c in batch]
            embeddings = [c.embedding for c in batch]
            documents = [c.text for c in batch]
            metadatas = [
                {
                    "paper_id": c.paper_id,
                    "chunk_type": c.chunk_type,
                    **{k: str(v) for k, v in c.metadata.items()},
                }
                for c in batch
            ]

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            added += len(batch)

        logger.info(f"Added {added} chunks to collection")
        return added

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        chunk_types: list[ChunkType] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        collections: list[str] | None = None,
        item_types: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            chunk_types: Filter by chunk types.
            year_min: Minimum publication year.
            year_max: Maximum publication year.
            collections: Filter by collection names.
            item_types: Filter by item types.

        Returns:
            List of SearchResult objects sorted by score.
        """
        where_clauses = []

        if chunk_types:
            where_clauses.append({"chunk_type": {"$in": chunk_types}})

        if year_min is not None:
            where_clauses.append({"year": {"$gte": year_min}})

        if year_max is not None:
            where_clauses.append({"year": {"$lte": year_max}})

        if item_types:
            where_clauses.append({"item_type": {"$in": item_types}})

        # Build where filter
        where = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to SearchResult objects
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0

                # Convert distance to similarity score (ChromaDB uses L2 distance)
                # Lower distance = higher similarity
                score = 1.0 / (1.0 + distance)

                search_results.append(
                    SearchResult(
                        paper_id=metadata.get("paper_id", ""),
                        chunk_id=chunk_id,
                        chunk_type=metadata.get("chunk_type", "full_summary"),
                        text=results["documents"][0][i] if results["documents"] else "",
                        score=score,
                        metadata=metadata,
                    )
                )

        # Filter by collections if specified (post-filter since ChromaDB doesn't support substring matching)
        if collections:
            search_results = [
                r
                for r in search_results
                if any(c in r.metadata.get("collections", "") for c in collections)
            ]

        return search_results

    def search_by_text(
        self,
        query_text: str,
        embedding_generator,
        top_k: int = 10,
        **filter_kwargs,
    ) -> list[SearchResult]:
        """Search using text query (generates embedding automatically).

        Args:
            query_text: Text query to search for.
            embedding_generator: EmbeddingGenerator instance.
            top_k: Number of results to return.
            **filter_kwargs: Additional filter arguments for search().

        Returns:
            List of SearchResult objects.
        """
        query_embedding = embedding_generator.embed_text(query_text)
        return self.search(query_embedding, top_k=top_k, **filter_kwargs)

    def delete_paper(self, paper_id: str) -> int:
        """Delete all chunks for a paper.

        Args:
            paper_id: Paper ID to delete.

        Returns:
            Number of chunks deleted.
        """
        # Get all chunk IDs for this paper
        results = self.collection.get(
            where={"paper_id": paper_id},
            include=[],
        )

        if not results["ids"]:
            return 0

        count = len(results["ids"])
        self.collection.delete(ids=results["ids"])
        logger.info(f"Deleted {count} chunks for paper {paper_id}")
        return count

    def delete_papers(self, paper_ids: list[str]) -> int:
        """Delete chunks for multiple papers.

        Args:
            paper_ids: List of paper IDs to delete.

        Returns:
            Total number of chunks deleted.
        """
        total = 0
        for paper_id in paper_ids:
            total += self.delete_paper(paper_id)
        return total

    def get_paper_chunks(self, paper_id: str) -> list[dict]:
        """Get all chunks for a specific paper.

        Args:
            paper_id: Paper ID to retrieve.

        Returns:
            List of chunk dictionaries.
        """
        results = self.collection.get(
            where={"paper_id": paper_id},
            include=["documents", "metadatas"],
        )

        chunks = []
        if results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": results["documents"][i] if results["documents"] else "",
                        "metadata": results["metadatas"][i] if results["metadatas"] else {},
                    }
                )

        return chunks

    def count(self) -> int:
        """Get total number of documents in the collection."""
        return self.collection.count()

    def get_stats(self) -> dict:
        """Get statistics about the vector store.

        Returns:
            Dictionary with store statistics.
        """
        total_count = self.collection.count()

        # Get chunk type distribution
        chunk_type_counts = {}
        for chunk_type in CHUNK_TYPES:
            try:
                results = self.collection.get(
                    where={"chunk_type": chunk_type},
                    include=[],
                )
                chunk_type_counts[chunk_type] = len(results["ids"]) if results["ids"] else 0
            except Exception:
                chunk_type_counts[chunk_type] = 0

        # Get unique paper count using metadatas only (no documents/embeddings)
        try:
            all_results = self.collection.get(include=["metadatas"])
            paper_ids = set()
            metadatas = all_results.get("metadatas") if isinstance(all_results, dict) else None
            if metadatas:
                for meta in metadatas:
                    if meta and "paper_id" in meta:
                        paper_ids.add(meta["paper_id"])
            unique_papers = len(paper_ids)
        except Exception:
            unique_papers = 0

        return {
            "total_chunks": total_count,
            "unique_papers": unique_papers,
            "chunk_type_distribution": chunk_type_counts,
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
        }

    def clear(self) -> int:
        """Clear all documents from the collection.

        Returns:
            Number of documents deleted.
        """
        count = self.collection.count()
        if count > 0:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection()
            logger.info(f"Cleared {count} documents from collection")
        return count
