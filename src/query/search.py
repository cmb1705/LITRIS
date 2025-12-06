"""Search engine combining semantic and metadata search."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.indexing.embeddings import CHUNK_TYPES, ChunkType, EmbeddingGenerator
from src.indexing.structured_store import StructuredStore
from src.indexing.vector_store import SearchResult, VectorStore
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class EnrichedResult:
    """Search result enriched with paper metadata and extraction."""

    paper_id: str
    title: str
    authors: str
    year: int | None
    collections: list[str]
    item_type: str
    chunk_type: ChunkType
    matched_text: str
    score: float
    paper_data: dict = field(default_factory=dict)
    extraction_data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "collections": self.collections,
            "item_type": self.item_type,
            "chunk_type": self.chunk_type,
            "matched_text": self.matched_text,
            "score": self.score,
            "paper": self.paper_data,
            "extraction": self.extraction_data,
        }


class SearchEngine:
    """Combined semantic and metadata search engine."""

    def __init__(
        self,
        index_dir: Path | str,
        chroma_dir: Path | str | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """Initialize search engine.

        Args:
            index_dir: Directory containing index JSON files.
            chroma_dir: Directory for ChromaDB storage. Defaults to index_dir/chroma.
            embedding_model: Name of sentence-transformers model.
        """
        self.index_dir = Path(index_dir)
        self.chroma_dir = Path(chroma_dir) if chroma_dir else self.index_dir / "chroma"

        logger.info(f"Initializing search engine with index at {self.index_dir}")

        # Initialize components
        self.structured_store = StructuredStore(self.index_dir)
        self.vector_store = VectorStore(self.chroma_dir)
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)

    def search(
        self,
        query: str,
        top_k: int = 10,
        chunk_types: list[ChunkType] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        collections: list[str] | None = None,
        item_types: list[str] | None = None,
        include_paper_data: bool = True,
        include_extraction: bool = False,
        deduplicate_papers: bool = True,
    ) -> list[EnrichedResult]:
        """Perform semantic search with optional metadata filtering.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            chunk_types: Filter by chunk types (abstract, thesis, etc.).
            year_min: Minimum publication year.
            year_max: Maximum publication year.
            collections: Filter by collection names.
            item_types: Filter by item types (journalArticle, book, etc.).
            include_paper_data: Include full paper metadata in results.
            include_extraction: Include extraction data in results.
            deduplicate_papers: Return only best match per paper.

        Returns:
            List of EnrichedResult objects sorted by score.
        """
        logger.info(f"Searching for: {query[:50]}...")

        # Generate query embedding
        query_embedding = self.embedding_generator.embed_text(query)

        # Search vector store
        # Request more results if deduplicating
        search_k = top_k * 3 if deduplicate_papers else top_k

        raw_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=search_k,
            chunk_types=chunk_types,
            year_min=year_min,
            year_max=year_max,
            collections=collections,
            item_types=item_types,
        )

        # Enrich results with paper metadata
        enriched = []
        seen_papers = set()

        for result in raw_results:
            # Deduplicate by paper if requested
            if deduplicate_papers and result.paper_id in seen_papers:
                continue

            seen_papers.add(result.paper_id)

            # Get paper metadata
            paper_data = {}
            extraction_data = {}

            if include_paper_data or include_extraction:
                combined = self.structured_store.get_paper_with_extraction(result.paper_id)
                if combined:
                    if include_paper_data:
                        paper_data = combined.get("paper", {})
                    if include_extraction:
                        extraction_data = combined.get("extraction", {})

            # Extract fields from metadata
            title = result.metadata.get("title", paper_data.get("title", "Unknown"))
            authors = result.metadata.get("authors", paper_data.get("author_string", ""))
            year_str = result.metadata.get("year", paper_data.get("publication_year"))
            year = int(year_str) if year_str and str(year_str).isdigit() else None
            collections_str = result.metadata.get("collections", "")
            collections_list = (
                collections_str.split(",") if collections_str else
                paper_data.get("collections", [])
            )
            item_type = result.metadata.get("item_type", paper_data.get("item_type", ""))

            enriched.append(
                EnrichedResult(
                    paper_id=result.paper_id,
                    title=title,
                    authors=authors,
                    year=year,
                    collections=[c.strip() for c in collections_list if c.strip()],
                    item_type=item_type,
                    chunk_type=result.chunk_type,
                    matched_text=result.text,
                    score=result.score,
                    paper_data=paper_data,
                    extraction_data=extraction_data,
                )
            )

            if len(enriched) >= top_k:
                break

        logger.info(f"Found {len(enriched)} results")
        return enriched

    def search_similar_papers(
        self,
        paper_id: str,
        top_k: int = 10,
        exclude_self: bool = True,
    ) -> list[EnrichedResult]:
        """Find papers similar to a given paper.

        Args:
            paper_id: Paper ID to find similar papers for.
            top_k: Number of similar papers to return.
            exclude_self: Whether to exclude the source paper.

        Returns:
            List of similar papers as EnrichedResult objects.
        """
        # Get the paper's full_summary chunk
        chunks = self.vector_store.get_paper_chunks(paper_id)
        summary_chunk = None

        for chunk in chunks:
            if chunk.get("metadata", {}).get("chunk_type") == "full_summary":
                summary_chunk = chunk
                break

        if not summary_chunk:
            # Fall back to any chunk
            if chunks:
                summary_chunk = chunks[0]
            else:
                logger.warning(f"No chunks found for paper {paper_id}")
                return []

        # Search using the summary text
        results = self.search(
            query=summary_chunk.get("text", ""),
            top_k=top_k + (1 if exclude_self else 0),
            chunk_types=["full_summary", "thesis", "contribution"],
            deduplicate_papers=True,
        )

        if exclude_self:
            results = [r for r in results if r.paper_id != paper_id]

        return results[:top_k]

    def search_by_metadata(
        self,
        title_contains: str | None = None,
        author_contains: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        collection: str | None = None,
        item_type: str | None = None,
    ) -> list[dict]:
        """Search papers by metadata fields only (no semantic search).

        Args:
            title_contains: Filter by title substring.
            author_contains: Filter by author name substring.
            year_min: Minimum publication year.
            year_max: Maximum publication year.
            collection: Filter by collection name.
            item_type: Filter by item type.

        Returns:
            List of matching paper dictionaries.
        """
        return self.structured_store.search_papers(
            title_contains=title_contains,
            author_contains=author_contains,
            year_min=year_min,
            year_max=year_max,
            collection=collection,
            item_type=item_type,
        )

    def get_paper(self, paper_id: str) -> dict | None:
        """Get a paper with its extraction.

        Args:
            paper_id: Paper ID to retrieve.

        Returns:
            Combined paper and extraction data, or None.
        """
        return self.structured_store.get_paper_with_extraction(paper_id)

    def get_summary(self) -> dict:
        """Get index summary statistics.

        Returns:
            Summary dictionary.
        """
        summary = self.structured_store.load_summary()
        if not summary:
            summary = self.structured_store.generate_summary()

        # Add vector store stats
        vector_stats = self.vector_store.get_stats()
        summary["vector_store"] = vector_stats

        return summary

    def get_collections(self) -> list[str]:
        """Get list of all collections in the index.

        Returns:
            List of collection names.
        """
        summary = self.structured_store.generate_summary()
        return list(summary.get("papers_by_collection", {}).keys())

    def get_item_types(self) -> list[str]:
        """Get list of all item types in the index.

        Returns:
            List of item type names.
        """
        summary = self.structured_store.generate_summary()
        return list(summary.get("papers_by_type", {}).keys())

    def get_year_range(self) -> tuple[int | None, int | None]:
        """Get the range of publication years in the index.

        Returns:
            Tuple of (min_year, max_year).
        """
        summary = self.structured_store.generate_summary()
        years = summary.get("papers_by_year", {})
        if not years:
            return None, None

        year_ints = [int(y) for y in years.keys() if y.isdigit()]
        if not year_ints:
            return None, None

        return min(year_ints), max(year_ints)
