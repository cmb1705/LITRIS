"""Adapter layer connecting MCP tools to LITRIS SearchEngine."""

from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import Config
from src.query.search import SearchEngine
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class LitrisAdapter:
    """Adapter that wraps SearchEngine for MCP tool access."""

    def __init__(self, config: Config | None = None):
        """Initialize the adapter.

        Args:
            config: Optional Config instance. If None, loads from config.yaml.
        """
        if config is None:
            config = Config.load()

        self.config = config
        self._engine: SearchEngine | None = None

    @property
    def engine(self) -> SearchEngine:
        """Lazy-initialize and return the SearchEngine."""
        if self._engine is None:
            logger.info("Initializing SearchEngine...")
            index_dir = self._get_index_dir()
            # ChromaDB is stored inside the index directory
            chroma_dir = index_dir / "chroma"

            self._engine = SearchEngine(
                index_dir=index_dir,
                chroma_dir=chroma_dir,
                embedding_model=self.config.embeddings.model,
            )
            logger.info("SearchEngine initialized successfully")

        return self._engine

    def _get_index_dir(self) -> Path:
        """Get the index directory path."""
        # Index is stored in data/index relative to project root
        if self.config._project_root:
            return self.config._project_root / "data" / "index"
        return Path("data/index")

    def search(
        self,
        query: str,
        top_k: int = 10,
        chunk_types: list[str] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        collections: list[str] | None = None,
        item_types: list[str] | None = None,
        include_extraction: bool = True,
        recency_boost: float = 0.0,
    ) -> dict[str, Any]:
        """Perform semantic search and format results for MCP.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            chunk_types: Filter by extraction section types.
            year_min: Minimum publication year.
            year_max: Maximum publication year.
            collections: Filter by Zotero collection names.
            item_types: Filter by item type.
            include_extraction: Include full extraction data.
            recency_boost: Boost factor for recent papers (0.0-1.0).

        Returns:
            Formatted search results dictionary.
        """
        logger.info(f"Searching: '{query[:50]}...' top_k={top_k}")

        results = self.engine.search(
            query=query,
            top_k=top_k,
            chunk_types=chunk_types,
            year_min=year_min,
            year_max=year_max,
            collections=collections,
            item_types=item_types,
            include_paper_data=True,
            include_extraction=include_extraction,
            deduplicate_papers=True,
        )

        # Apply recency boost if specified
        if recency_boost > 0.0 and results:
            results = self._apply_recency_boost(results, recency_boost)

        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted = {
                "rank": i,
                "score": round(result.score, 4),
                "paper_id": result.paper_id,
                "title": result.title,
                "authors": result.authors,
                "year": result.year,
                "collections": result.collections,
                "item_type": result.item_type,
                "chunk_type": result.chunk_type,
                "matched_text": result.matched_text[:500] if result.matched_text else "",
            }

            if include_extraction and result.extraction_data:
                formatted["extraction"] = self._format_extraction(result.extraction_data)

            # Include PDF path if available
            pdf_path = result.paper_data.get("pdf_path")
            if pdf_path:
                formatted["pdf_path"] = pdf_path

            formatted_results.append(formatted)

        return {
            "query": query,
            "result_count": len(formatted_results),
            "results": formatted_results,
        }

    def _apply_recency_boost(
        self, results: list, recency_boost: float
    ) -> list:
        """Apply recency boost to search results.

        More recent papers get a score bonus based on their publication year.

        Args:
            results: List of EnrichedResult objects.
            recency_boost: Boost factor (0.0-1.0).

        Returns:
            Re-sorted results with boosted scores.
        """
        current_year = datetime.now().year
        boosted = []

        for result in results:
            if result.year:
                # Calculate age-based boost (papers within last 5 years get full boost)
                age = current_year - result.year
                if age <= 0:
                    boost_factor = recency_boost
                elif age <= 5:
                    boost_factor = recency_boost * (1 - age / 10)
                elif age <= 10:
                    boost_factor = recency_boost * (0.5 - (age - 5) / 20)
                else:
                    boost_factor = 0.0

                # Apply boost to score
                boosted_score = result.score * (1 + boost_factor)
            else:
                boosted_score = result.score

            boosted.append((boosted_score, result))

        # Sort by boosted score
        boosted.sort(key=lambda x: x[0], reverse=True)

        # Update original scores are preserved, just re-ordered
        return [r for _, r in boosted]

    def _format_extraction(self, extraction: dict) -> dict[str, Any]:
        """Format extraction data for MCP response.

        Args:
            extraction: Raw extraction dictionary.

        Returns:
            Formatted extraction data.
        """
        return {
            "thesis_statement": extraction.get("thesis_statement", ""),
            "research_questions": extraction.get("research_questions", []),
            "methodology": extraction.get("methodology", {}),
            "key_findings": extraction.get("key_findings", []),
            "conclusions": extraction.get("conclusions", ""),
            "limitations": extraction.get("limitations", []),
            "future_directions": extraction.get("future_directions", []),
            "contribution_summary": extraction.get("contribution_summary", ""),
            "discipline_tags": extraction.get("discipline_tags", []),
            "extraction_confidence": extraction.get("extraction_confidence", 0.0),
        }

    def get_paper(self, paper_id: str) -> dict[str, Any]:
        """Get full details for a specific paper.

        Args:
            paper_id: LITRIS paper identifier.

        Returns:
            Paper data with extraction, or not found response.
        """
        logger.info(f"Getting paper: {paper_id}")

        combined = self.engine.get_paper(paper_id)

        if not combined:
            return {
                "paper_id": paper_id,
                "found": False,
                "error": f"Paper not found: {paper_id}",
            }

        paper = combined.get("paper", {})
        extraction = combined.get("extraction", {})

        return {
            "paper_id": paper_id,
            "found": True,
            "paper": {
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "author_string": paper.get("author_string", ""),
                "publication_year": paper.get("publication_year"),
                "publication_date": paper.get("publication_date"),
                "journal": paper.get("journal"),
                "doi": paper.get("doi"),
                "abstract": paper.get("abstract"),
                "collections": paper.get("collections", []),
                "tags": paper.get("tags", []),
                "item_type": paper.get("item_type", ""),
                "pdf_path": paper.get("pdf_path"),
                "zotero_key": paper.get("zotero_key", ""),
            },
            "extraction": self._format_extraction(extraction) if extraction else None,
        }

    def find_similar(self, paper_id: str, top_k: int = 10) -> dict[str, Any]:
        """Find papers similar to a given paper.

        Args:
            paper_id: Source paper identifier.
            top_k: Number of similar papers to return.

        Returns:
            Similar papers with similarity scores.
        """
        logger.info(f"Finding similar to: {paper_id} (top_k={top_k})")

        # First get the source paper for title
        source = self.engine.get_paper(paper_id)
        if not source:
            return {
                "source_paper_id": paper_id,
                "found": False,
                "error": f"Source paper not found: {paper_id}",
            }

        source_title = source.get("paper", {}).get("title", "Unknown")

        results = self.engine.search_similar_papers(
            paper_id=paper_id,
            top_k=top_k,
            exclude_self=True,
        )

        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append({
                "rank": i,
                "score": round(result.score, 4),
                "paper_id": result.paper_id,
                "title": result.title,
                "authors": result.authors,
                "year": result.year,
                "matched_on": result.chunk_type,
                "extraction": self._format_extraction(result.extraction_data)
                if result.extraction_data else None,
            })

        return {
            "source_paper_id": paper_id,
            "source_title": source_title,
            "result_count": len(formatted_results),
            "similar_papers": formatted_results,
        }

    def get_summary(self) -> dict[str, Any]:
        """Get index summary statistics.

        Returns:
            Summary statistics dictionary.
        """
        logger.info("Getting index summary")

        summary = self.engine.get_summary()

        return {
            "generated_at": datetime.now().isoformat(),
            "total_papers": summary.get("total_papers", 0),
            "total_extractions": summary.get("total_extractions", 0),
            "papers_by_type": summary.get("papers_by_type", {}),
            "papers_by_year": summary.get("papers_by_year", {}),
            "papers_by_collection": summary.get("papers_by_collection", {}),
            "top_disciplines": summary.get("top_disciplines", {}),
            "vector_store": summary.get("vector_store", {}),
            "recent_papers": summary.get("recent_papers", [])[:10],
        }

    def get_collections(self) -> dict[str, Any]:
        """List all collections in the index.

        Returns:
            Collections with paper counts.
        """
        logger.info("Getting collections list")

        summary = self.engine.get_summary()
        collection_counts = summary.get("papers_by_collection", {})

        return {
            "collections": list(collection_counts.keys()),
            "collection_counts": collection_counts,
        }
