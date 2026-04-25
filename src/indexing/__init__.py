"""Embedding generation and vector storage module."""

from src.indexing.embeddings import (
    CHUNK_TYPES,
    ChunkType,
    EmbeddingChunk,
    EmbeddingGenerator,
)
from src.indexing.structured_store import StructuredStore
from src.indexing.update_state import UpdateRecord, UpdateState

__all__ = [
    "CHUNK_TYPES",
    "ChunkType",
    "EmbeddingChunk",
    "EmbeddingGenerator",
    "SearchResult",
    "StructuredStore",
    "UpdateRecord",
    "UpdateState",
    "VectorStore",
]


def __getattr__(name: str):
    """Lazily import vector-store types so non-Chroma paths avoid heavy side effects."""
    if name in {"SearchResult", "VectorStore"}:
        from src.indexing.vector_store import SearchResult, VectorStore

        return {"SearchResult": SearchResult, "VectorStore": VectorStore}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
