"""Adapter layer connecting MCP tools to LITRIS SearchEngine."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from src.utils.logging_config import get_logger

if TYPE_CHECKING:
    from src.config import Config
    from src.query.search import SearchEngine
else:

    class Config:
        """Lazy proxy preserving the historical ``src.mcp.adapters.Config`` patch target."""

        @classmethod
        def load(cls, *args: Any, **kwargs: Any) -> Any:
            from src.config import Config as RealConfig

            return RealConfig.load(*args, **kwargs)

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
            from src.query.search import SearchEngine

            logger.info("Initializing SearchEngine...")
            index_dir = self._get_index_dir()
            chroma_dir = index_dir / "chroma"

            # Validate paths exist
            if not index_dir.exists():
                logger.error(f"Index directory not found: {index_dir}")
                raise FileNotFoundError(
                    f"Index directory not found: {index_dir}. "
                    "Run the /build command to create the literature index."
                )

            papers_index = index_dir / "papers.json"
            if not papers_index.exists():
                logger.error(f"Papers index not found: {papers_index}")
                raise FileNotFoundError(
                    f"Papers index not found: {papers_index}. "
                    "Run the /build command to create the literature index."
                )

            if not chroma_dir.exists():
                logger.warning(
                    f"ChromaDB directory not found: {chroma_dir}. Vector search may fail."
                )

            self._engine = SearchEngine(
                index_dir=index_dir,
                chroma_dir=chroma_dir,
                embedding_model=self.config.embeddings.model,
                embedding_backend=self.config.embeddings.backend,
                embedding_device=self.config.embeddings.device,
                ollama_base_url=self.config.embeddings.ollama_base_url,
                embedding_ollama_concurrency=self.config.embeddings.ollama_concurrency,
                query_prefix=self.config.embeddings.query_prefix,
                document_prefix=self.config.embeddings.document_prefix,
            )
            logger.info("SearchEngine initialized successfully")

        return self._engine

    def _get_index_dir(self) -> Path:
        """Get the index directory path."""
        # Index is stored in data/index relative to project root
        if self.config._project_root:
            return cast(Path, self.config._project_root / "data" / "index")
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
        quality_min: int | None = None,
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
            quality_min=quality_min,
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

    def _apply_recency_boost(self, results: list, recency_boost: float) -> list:
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
        """Format SemanticAnalysis data for MCP response.

        Handles both flat extraction dicts and wrapper records that nest
        the actual fields under an ``"extraction"`` key (the format used
        in ``semantic_analyses.json``).

        Dimensions are grouped by analysis pass for readability.

        Args:
            extraction: Raw extraction/analysis dictionary (flat or wrapped).

        Returns:
            Formatted SemanticAnalysis data grouped by pass.
        """
        # Unwrap if the record contains a nested "extraction" dict
        if "extraction" in extraction and isinstance(extraction["extraction"], dict):
            extraction = extraction["extraction"]

        groups = {}
        engine = getattr(self, "_engine", None)
        registry = getattr(engine, "dimension_registry", None)
        if registry is None:
            from src.analysis.dimensions import get_default_dimension_registry

            registry = get_default_dimension_registry()
        dimensions = extraction.get("dimensions", {})
        if not isinstance(dimensions, dict):
            dimensions = {}
        for group_name, dimension_ids in registry.get_dimension_groups().items():
            group_data = {}
            for dimension_id in dimension_ids:
                dimension = registry.resolve_optional_dimension(dimension_id)
                output_key = (
                    dimension.legacy_field_name
                    if dimension and dimension.legacy_field_name
                    else dimension_id
                )
                value = dimensions.get(dimension_id)
                if value is None:
                    value = extraction.get(output_key)
                if value is None:
                    value = extraction.get(dimension_id)
                if value is not None:
                    group_data[output_key] = value
            if group_data:
                groups[group_name] = group_data

        return {
            **groups,
            "dimension_coverage": extraction.get("dimension_coverage", 0.0),
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
        fulltext = combined.get("fulltext")

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
            "fulltext": fulltext,
            "fulltext_available": bool(fulltext),
        }

    def get_fulltext_context(
        self,
        paper_id: str,
        query: str,
        max_hits: int = 3,
        context_chars: int = 400,
    ) -> dict[str, Any]:
        """Get verbatim context windows from a paper's canonical full-text snapshot."""

        logger.info(
            "Getting full-text context: %s query='%s...' max_hits=%d",
            paper_id,
            query[:40],
            max_hits,
        )

        combined = self.engine.get_paper(paper_id)
        if not combined:
            return {
                "paper_id": paper_id,
                "found": False,
                "error": f"Paper not found: {paper_id}",
            }

        context = self.engine.get_fulltext_context(
            paper_id=paper_id,
            query=query,
            max_hits=max_hits,
            context_chars=context_chars,
        )
        paper = combined.get("paper", {})
        context["paper"] = {
            "title": paper.get("title", ""),
            "author_string": paper.get("author_string", ""),
            "publication_year": paper.get("publication_year"),
        }
        return cast(dict[str, Any], context)

    def search_rrf(
        self,
        query: str,
        top_k: int = 10,
        n_variants: int = 4,
        chunk_types: list[str] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        collections: list[str] | None = None,
        item_types: list[str] | None = None,
        include_extraction: bool = True,
        recency_boost: float = 0.0,
        quality_min: int | None = None,
    ) -> dict[str, Any]:
        """Multi-query search using Reciprocal Rank Fusion.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            n_variants: Number of query reformulations.
            chunk_types: Filter by extraction section types.
            year_min: Minimum publication year.
            year_max: Maximum publication year.
            collections: Filter by Zotero collection names.
            item_types: Filter by item type.
            include_extraction: Include full extraction data.
            recency_boost: Boost factor for recent papers (0.0-1.0).
            quality_min: Minimum paper quality rating (1-5).

        Returns:
            Formatted search results with query variants.
        """
        logger.info(f"RRF search: '{query[:50]}...' top_k={top_k} variants={n_variants}")

        results, query_variants = self.engine.search_rrf(
            query=query,
            top_k=top_k,
            n_variants=n_variants,
            chunk_types=chunk_types,
            year_min=year_min,
            year_max=year_max,
            collections=collections,
            item_types=item_types,
            include_paper_data=True,
            include_extraction=include_extraction,
            quality_min=quality_min,
        )

        # Apply recency boost if specified
        if recency_boost > 0.0 and results:
            results = self._apply_recency_boost(results, recency_boost)

        # Format results (same structure as search())
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted = {
                "rank": i,
                "score": round(result.score, 6),
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

            pdf_path = result.paper_data.get("pdf_path")
            if pdf_path:
                formatted["pdf_path"] = pdf_path

            formatted_results.append(formatted)

        return {
            "query": query,
            "query_variants": query_variants,
            "result_count": len(formatted_results),
            "results": formatted_results,
        }

    def search_agentic(
        self,
        query: str,
        top_k: int = 10,
        max_rounds: int = 2,
        chunk_types: list[str] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        collections: list[str] | None = None,
        item_types: list[str] | None = None,
        include_extraction: bool = True,
        recency_boost: float = 0.0,
        quality_min: int | None = None,
    ) -> dict[str, Any]:
        """Multi-round agentic search with gap analysis.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            max_rounds: Maximum gap-analysis rounds (1-5).
            chunk_types: Filter by extraction section types.
            year_min: Minimum publication year.
            year_max: Maximum publication year.
            collections: Filter by Zotero collection names.
            item_types: Filter by item type.
            include_extraction: Include full extraction data.
            recency_boost: Boost factor for recent papers (0.0-1.0).
            quality_min: Minimum paper quality rating (1-5).

        Returns:
            Formatted search results with round-by-round metadata.
        """
        logger.info(f"Agentic search: '{query[:50]}...' top_k={top_k} rounds={max_rounds}")

        results, metadata = self.engine.search_agentic(
            query=query,
            top_k=top_k,
            max_rounds=max_rounds,
            chunk_types=chunk_types,
            year_min=year_min,
            year_max=year_max,
            collections=collections,
            item_types=item_types,
            include_paper_data=True,
            include_extraction=include_extraction,
            quality_min=quality_min,
        )

        # Apply recency boost if specified
        if recency_boost > 0.0 and results:
            results = self._apply_recency_boost(results, recency_boost)

        # Format results (same structure as search())
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

            pdf_path = result.paper_data.get("pdf_path")
            if pdf_path:
                formatted["pdf_path"] = pdf_path

            formatted_results.append(formatted)

        # Format round metadata
        rounds_info = []
        for r in metadata.rounds:
            round_dict = {
                "round": r.round_number,
                "queries_used": r.queries_used,
                "papers_found": r.papers_found,
                "new_papers": r.new_papers,
            }
            if r.gap_analysis:
                round_dict["gaps_identified"] = r.gap_analysis.gaps
                round_dict["follow_up_queries"] = r.gap_analysis.follow_up_queries
            rounds_info.append(round_dict)

        return {
            "query": query,
            "result_count": len(formatted_results),
            "total_papers_explored": metadata.total_papers,
            "rounds_completed": len(metadata.rounds),
            "rounds": rounds_info,
            "results": formatted_results,
        }

    def deep_review(
        self,
        topic: str,
        top_k: int = 20,
        max_rounds: int = 2,
        verify: bool = True,
    ) -> dict[str, Any]:
        """Execute the deep review literature synthesis pipeline.

        Args:
            topic: Research topic or question for the review.
            top_k: Number of papers to include in synthesis.
            max_rounds: Max gap-analysis rounds for discovery.
            verify: Whether to run QA citation verification.

        Returns:
            Deep review result with literature review and metadata.
        """
        from src.query.deep_review import deep_review

        logger.info(f"Deep review: '{topic[:50]}...' top_k={top_k}")

        result = deep_review(
            topic=topic,
            engine=self.engine,
            adapter=self,
            top_k=top_k,
            max_rounds=max_rounds,
            verify=verify,
        )

        response = {
            "topic": result.topic,
            "papers_discovered": result.papers_discovered,
            "papers_used": result.papers_used,
            "review": result.review_text,
            "generated_at": result.generated_at,
            "source_papers": [
                {
                    "paper_id": r.paper_id,
                    "title": r.title,
                    "authors": r.authors,
                    "year": r.year,
                }
                for r in result.paper_readings
            ],
        }

        if result.qa_result:
            response["qa_verification"] = {
                "verified": result.qa_result.verified,
                "citation_count": result.qa_result.citation_count,
                "issues": result.qa_result.issues,
                "uncited_papers": result.qa_result.uncited_papers,
            }

        return response

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
            formatted_results.append(
                {
                    "rank": i,
                    "score": round(result.score, 4),
                    "paper_id": result.paper_id,
                    "title": result.title,
                    "authors": result.authors,
                    "year": result.year,
                    "matched_on": result.chunk_type,
                    "extraction": self._format_extraction(result.extraction_data)
                    if result.extraction_data
                    else None,
                }
            )

        return {
            "source_paper_id": paper_id,
            "source_title": source_title,
            "result_count": len(formatted_results),
            "similar_papers": formatted_results,
        }

    def get_clusters(
        self,
        min_cluster_size: int = 5,
    ) -> dict[str, Any]:
        """Run topic clustering on paper embeddings.

        Args:
            min_cluster_size: Minimum papers per cluster.

        Returns:
            Clustering results with topic assignments.
        """
        from src.analysis.clustering import run_clustering

        logger.info("Running topic clustering...")

        papers_dict = self.engine.structured_store.load_papers()
        result = run_clustering(
            vector_store=self.engine.vector_store,
            papers_dict=papers_dict,
            min_cluster_size=min_cluster_size,
        )

        return cast(dict[str, Any], result.to_dict())

    def get_summary(self) -> dict[str, Any]:
        """Get index summary statistics.

        Returns:
            Summary statistics dictionary.
        """
        logger.info("Getting index summary")

        summary = self.engine.get_summary()

        result = {
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

        if "similarity_pairs" in summary:
            result["similarity_pairs"] = summary["similarity_pairs"]

        return result

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
