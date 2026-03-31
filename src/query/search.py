"""Search engine combining semantic and metadata search."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.indexing.embeddings import ChunkType, EmbeddingGenerator

if TYPE_CHECKING:
    from src.query.agentic import AgenticSearchResult
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
        embedding_backend: str = "sentence-transformers",
        ollama_base_url: str = "http://localhost:11434",
        query_prefix: str | None = None,
        document_prefix: str | None = None,
    ):
        """Initialize search engine.

        Args:
            index_dir: Directory containing index JSON files.
            chroma_dir: Directory for ChromaDB storage. Defaults to index_dir/chroma.
            embedding_model: Name of embedding model.
            embedding_backend: Backend ('sentence-transformers' or 'ollama').
            ollama_base_url: Base URL for Ollama server.
            query_prefix: Prefix for query texts (e.g., instruction prefix for Qwen3).
            document_prefix: Prefix for document texts during embedding.
        """
        self.index_dir = Path(index_dir)
        self.chroma_dir = Path(chroma_dir) if chroma_dir else self.index_dir / "chroma"

        logger.info(f"Initializing search engine with index at {self.index_dir}")

        # Initialize components
        self.structured_store = StructuredStore(self.index_dir)
        self.vector_store = VectorStore(self.chroma_dir)
        self.embedding_generator = EmbeddingGenerator(
            model_name=embedding_model,
            backend=embedding_backend,
            ollama_base_url=ollama_base_url,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
        )

    def close(self) -> None:
        """Release vector-store resources owned by this engine."""
        self.vector_store.close()

    def __enter__(self) -> "SearchEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

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
        quality_min: int | None = None,
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
            quality_min: Minimum paper quality rating (1-5).

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
            quality_min=quality_min,
        )

        # Enrich results with paper metadata
        enriched = []
        seen_papers = set()

        # Pre-load papers and extractions to avoid repeated method calls
        papers_dict = {}
        extractions_dict = {}
        if include_paper_data or include_extraction:
            if include_paper_data:
                papers_dict = self.structured_store.load_papers()
            if include_extraction:
                extractions_dict = self.structured_store.load_extractions()

        for result in raw_results:
            # Deduplicate by paper if requested
            if deduplicate_papers and result.paper_id in seen_papers:
                continue

            seen_papers.add(result.paper_id)

            # Get paper metadata via direct dictionary access
            paper_data = papers_dict.get(result.paper_id, {}) if include_paper_data else {}
            extraction_data = extractions_dict.get(result.paper_id, {}) if include_extraction else {}

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

        Uses pre-computed similarity pairs when available,
        falling back to query-time vector search otherwise.

        Args:
            paper_id: Paper ID to find similar papers for.
            top_k: Number of similar papers to return.
            exclude_self: Whether to exclude the source paper.

        Returns:
            List of similar papers as EnrichedResult objects.
        """
        # Try pre-computed pairs first
        pairs = self.structured_store.load_similarity_pairs()
        if paper_id in pairs:
            logger.info(f"Using pre-computed similarity for {paper_id}")
            return self._enrich_similarity_pairs(pairs[paper_id][:top_k])

        # Fall back to query-time computation
        logger.info(f"No pre-computed pairs for {paper_id}, using vector search")
        chunks = self.vector_store.get_paper_chunks(paper_id)
        summary_chunk = None

        for chunk in chunks:
            if chunk.get("metadata", {}).get("chunk_type") == "raptor_overview":
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
            chunk_types=["raptor_overview", "dim_q02", "dim_q22"],
            deduplicate_papers=True,
        )

        if exclude_self:
            results = [r for r in results if r.paper_id != paper_id]

        return results[:top_k]

    def _enrich_similarity_pairs(
        self,
        pairs: list[dict],
    ) -> list[EnrichedResult]:
        """Convert pre-computed similarity pairs to EnrichedResult objects.

        Args:
            pairs: List of dicts with similar_paper_id and similarity_score.

        Returns:
            List of EnrichedResult objects.
        """
        papers_dict = self.structured_store.load_papers()
        extractions_dict = self.structured_store.load_extractions()
        enriched = []

        for pair in pairs:
            similar_id = pair["similar_paper_id"]
            score = pair["similarity_score"]
            paper_data = papers_dict.get(similar_id, {})

            if not paper_data:
                continue

            year_val = paper_data.get("publication_year")
            year = int(year_val) if year_val and str(year_val).isdigit() else None

            enriched.append(
                EnrichedResult(
                    paper_id=similar_id,
                    title=paper_data.get("title", "Unknown"),
                    authors=paper_data.get("author_string", ""),
                    year=year,
                    collections=paper_data.get("collections", []),
                    item_type=paper_data.get("item_type", ""),
                    chunk_type="raptor_overview",
                    matched_text="",
                    score=score,
                    paper_data=paper_data,
                    extraction_data=extractions_dict.get(similar_id, {}),
                )
            )

        return enriched

    def search_rrf(
        self,
        query: str,
        top_k: int = 10,
        n_variants: int = 4,
        rrf_k: int = 60,
        chunk_types: list[ChunkType] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        collections: list[str] | None = None,
        item_types: list[str] | None = None,
        include_paper_data: bool = True,
        include_extraction: bool = False,
        quality_min: int | None = None,
        rrf_provider: str = "anthropic",
        rrf_model: str | None = None,
    ) -> tuple[list[EnrichedResult], list[str]]:
        """Multi-query search using Reciprocal Rank Fusion.

        Generates query reformulations, runs each through vector search,
        and fuses results using RRF scoring for improved recall.

        Args:
            query: Natural language search query.
            top_k: Number of final results to return.
            n_variants: Number of query reformulations to generate.
            rrf_k: RRF constant (default 60).
            chunk_types: Filter by chunk types.
            year_min: Minimum publication year.
            year_max: Maximum publication year.
            collections: Filter by collection names.
            item_types: Filter by item types.
            include_paper_data: Include full paper metadata.
            include_extraction: Include extraction data.
            quality_min: Minimum paper quality rating (1-5).
            rrf_provider: LLM provider for query reformulation.
            rrf_model: LLM model for query reformulation.

        Returns:
            Tuple of (results, query_variants) where query_variants
            includes the original query and all reformulations.
        """
        from src.query.rrf import generate_query_variants, rrf_score

        logger.info(f"RRF search: '{query[:50]}...' with {n_variants} variants")

        # Generate query variants
        query_variants = generate_query_variants(
            query=query,
            n_variants=n_variants,
            provider=rrf_provider,
            model=rrf_model,
        )
        logger.info(f"Generated {len(query_variants)} query variants (incl. original)")

        # Run vector search for each variant and collect rankings
        # Request more results per variant to improve fusion quality
        per_query_k = top_k * 3
        rankings: list[list[str]] = []
        all_results: dict[str, SearchResult] = {}  # paper_id -> best raw result

        for variant in query_variants:
            query_embedding = self.embedding_generator.embed_text(variant)
            raw_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=per_query_k,
                chunk_types=chunk_types,
                year_min=year_min,
                year_max=year_max,
                collections=collections,
                item_types=item_types,
                quality_min=quality_min,
            )

            # Deduplicate within this variant's results by paper_id
            seen = set()
            ranking = []
            for result in raw_results:
                if result.paper_id not in seen:
                    seen.add(result.paper_id)
                    ranking.append(result.paper_id)
                    # Keep best result per paper for enrichment later
                    if result.paper_id not in all_results:
                        all_results[result.paper_id] = result

            rankings.append(ranking)

        # Compute RRF scores
        rrf_ranked = rrf_score(rankings, k=rrf_k)

        # Enrich top_k results
        papers_dict = {}
        extractions_dict = {}
        if include_paper_data:
            papers_dict = self.structured_store.load_papers()
        if include_extraction:
            extractions_dict = self.structured_store.load_extractions()

        enriched = []
        for paper_id, score in rrf_ranked[:top_k]:
            raw = all_results.get(paper_id)
            if not raw:
                continue

            paper_data = papers_dict.get(paper_id, {}) if include_paper_data else {}
            extraction_data = extractions_dict.get(paper_id, {}) if include_extraction else {}

            title = raw.metadata.get("title", paper_data.get("title", "Unknown"))
            authors = raw.metadata.get("authors", paper_data.get("author_string", ""))
            year_str = raw.metadata.get("year", paper_data.get("publication_year"))
            year = int(year_str) if year_str and str(year_str).isdigit() else None
            collections_str = raw.metadata.get("collections", "")
            collections_list = (
                collections_str.split(",") if collections_str else
                paper_data.get("collections", [])
            )
            item_type = raw.metadata.get("item_type", paper_data.get("item_type", ""))

            enriched.append(
                EnrichedResult(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    year=year,
                    collections=[c.strip() for c in collections_list if c.strip()],
                    item_type=item_type,
                    chunk_type=raw.chunk_type,
                    matched_text=raw.text,
                    score=score,
                    paper_data=paper_data,
                    extraction_data=extraction_data,
                )
            )

        logger.info(f"RRF search returned {len(enriched)} results from {len(all_results)} unique papers")
        return enriched, query_variants

    def search_agentic(
        self,
        query: str,
        top_k: int = 10,
        max_rounds: int = 2,
        chunk_types: list[ChunkType] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        collections: list[str] | None = None,
        item_types: list[str] | None = None,
        include_paper_data: bool = True,
        include_extraction: bool = True,
        quality_min: int | None = None,
        provider: str = "anthropic",
        model: str | None = None,
    ) -> tuple[list[EnrichedResult], AgenticSearchResult]:
        """Multi-round search with LLM gap analysis between rounds.

        Performs an initial search, then iteratively analyzes results for
        topical gaps and runs follow-up searches to fill them.

        Args:
            query: Natural language search query.
            top_k: Number of final results to return.
            max_rounds: Maximum gap-analysis rounds (default 2).
            chunk_types: Filter by chunk types.
            year_min: Minimum publication year.
            year_max: Maximum publication year.
            collections: Filter by collection names.
            item_types: Filter by item types.
            include_paper_data: Include full paper metadata.
            include_extraction: Include extraction data.
            quality_min: Minimum paper quality rating (1-5).
            provider: LLM provider for gap analysis.
            model: LLM model for gap analysis.

        Returns:
            Tuple of (enriched_results, agentic_metadata) where metadata
            contains round-by-round details and gap analyses.
        """
        from src.query.agentic import AgenticRound, AgenticSearchResult, analyze_gaps

        logger.info(
            f"Agentic search: '{query[:50]}...' max_rounds={max_rounds}"
        )

        search_kwargs = {
            "chunk_types": chunk_types,
            "year_min": year_min,
            "year_max": year_max,
            "collections": collections,
            "item_types": item_types,
            "include_paper_data": include_paper_data,
            "include_extraction": include_extraction,
            "deduplicate_papers": True,
            "quality_min": quality_min,
        }

        # Fetch more per round to give the gap analyzer good coverage
        per_round_k = max(top_k, 10)

        # Accumulate unique papers across all rounds
        all_papers: dict[str, EnrichedResult] = {}
        metadata = AgenticSearchResult(original_query=query)

        # Round 0: Initial search
        initial_results = self.search(query=query, top_k=per_round_k, **search_kwargs)
        for r in initial_results:
            all_papers[r.paper_id] = r

        round_0 = AgenticRound(
            round_number=0,
            queries_used=[query],
            papers_found=len(initial_results),
            new_papers=len(initial_results),
        )
        metadata.rounds.append(round_0)

        # Gap-analysis rounds
        for round_num in range(1, max_rounds + 1):
            # Format current results for gap analysis
            current_results_dicts = [r.to_dict() for r in all_papers.values()]

            gap = analyze_gaps(
                query=query,
                results=current_results_dicts,
                provider=provider,
                model=model,
            )

            if not gap.follow_up_queries:
                logger.info(
                    f"Round {round_num}: No gaps identified, stopping early"
                )
                round_info = AgenticRound(
                    round_number=round_num,
                    queries_used=[],
                    papers_found=0,
                    new_papers=0,
                    gap_analysis=gap,
                )
                metadata.rounds.append(round_info)
                break

            logger.info(
                f"Round {round_num}: {len(gap.gaps)} gaps, "
                f"{len(gap.follow_up_queries)} follow-up queries"
            )

            # Run follow-up searches
            round_queries = gap.follow_up_queries
            round_new = 0
            round_found = 0

            for follow_up_query in round_queries:
                follow_results = self.search(
                    query=follow_up_query,
                    top_k=per_round_k,
                    **search_kwargs,
                )
                round_found += len(follow_results)
                for r in follow_results:
                    if r.paper_id not in all_papers:
                        all_papers[r.paper_id] = r
                        round_new += 1

            round_info = AgenticRound(
                round_number=round_num,
                queries_used=round_queries,
                papers_found=round_found,
                new_papers=round_new,
                gap_analysis=gap,
            )
            metadata.rounds.append(round_info)

            if round_new == 0:
                logger.info(
                    f"Round {round_num}: No new papers found, stopping"
                )
                break

        # Sort all accumulated papers by score descending, take top_k
        sorted_results = sorted(
            all_papers.values(),
            key=lambda r: r.score,
            reverse=True,
        )[:top_k]

        metadata.total_papers = len(all_papers)

        logger.info(
            f"Agentic search complete: {len(metadata.rounds)} rounds, "
            f"{metadata.total_papers} unique papers, returning {len(sorted_results)}"
        )
        return sorted_results, metadata

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
