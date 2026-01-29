"""Federated search across multiple LITRIS indexes."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from src.config import FederatedSearchConfig
from src.indexing.embeddings import ChunkType
from src.query.search import EnrichedResult, SearchEngine
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FederatedResult(EnrichedResult):
    """Search result with federation metadata.

    Extends EnrichedResult to track which index the result came from,
    enabling provenance tracking and weighted scoring.
    """

    source_index: str = "primary"
    source_weight: float = 1.0
    weighted_score: float = 0.0

    def __post_init__(self) -> None:
        """Calculate weighted score after initialization."""
        self.weighted_score = self.score * self.source_weight

    def to_dict(self) -> dict:
        """Convert to dictionary with federation metadata."""
        result = super().to_dict()
        result["source_index"] = self.source_index
        result["source_weight"] = self.source_weight
        result["weighted_score"] = self.weighted_score
        return result


def _title_similarity(title1: str, title2: str) -> float:
    """Calculate title similarity using sequence matching.

    Args:
        title1: First title string.
        title2: Second title string.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if not title1 or not title2:
        return 0.0
    # Normalize: lowercase and strip whitespace
    t1 = title1.lower().strip()
    t2 = title2.lower().strip()
    return SequenceMatcher(None, t1, t2).ratio()


def _extract_doi(paper_data: dict) -> str | None:
    """Extract DOI from paper metadata.

    Args:
        paper_data: Paper metadata dictionary.

    Returns:
        DOI string or None.
    """
    # Check common DOI locations
    if doi := paper_data.get("doi"):
        return doi.lower().strip()
    if doi := paper_data.get("DOI"):
        return doi.lower().strip()
    # Check identifiers array
    for ident in paper_data.get("identifiers", []):
        if ident.get("type") == "doi":
            return ident.get("value", "").lower().strip()
    return None


class FederatedSearchEngine:
    """Search engine that queries multiple indexes and merges results.

    Supports three merge strategies:
    - interleave: Round-robin selection by weighted score
    - concat: Primary index results first, then federated indexes
    - rerank: Combined re-ranking across all results
    """

    def __init__(
        self,
        primary_index_dir: Path | str,
        config: FederatedSearchConfig,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """Initialize federated search engine.

        Args:
            primary_index_dir: Path to primary index directory.
            config: Federated search configuration.
            embedding_model: Sentence-transformers model name.
        """
        self.primary_index_dir = Path(primary_index_dir)
        self.config = config
        self.embedding_model = embedding_model

        # Initialize primary search engine
        self.primary_engine = SearchEngine(
            index_dir=self.primary_index_dir,
            embedding_model=embedding_model,
        )

        # Initialize federated search engines
        self.federated_engines: dict[str, tuple[SearchEngine, float]] = {}
        if config.enabled:
            self._load_federated_indexes()

    def _load_federated_indexes(self) -> None:
        """Load all enabled federated indexes."""
        for idx_config in self.config.indexes:
            if not idx_config.enabled:
                logger.debug(f"Skipping disabled index: {idx_config.label}")
                continue

            index_path = idx_config.path
            if not index_path.exists():
                logger.warning(f"Federated index not found: {index_path}")
                continue

            try:
                engine = SearchEngine(
                    index_dir=index_path,
                    embedding_model=self.embedding_model,
                )
                self.federated_engines[idx_config.label] = (engine, idx_config.weight)
                logger.info(f"Loaded federated index: {idx_config.label} (weight={idx_config.weight})")
            except Exception as e:
                logger.error(f"Failed to load federated index {idx_config.label}: {e}")

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
    ) -> list[FederatedResult]:
        """Execute federated search across all indexes.

        Args:
            query: Natural language search query.
            top_k: Number of results to return after merging.
            chunk_types: Filter by chunk types.
            year_min: Minimum publication year.
            year_max: Maximum publication year.
            collections: Filter by collection names.
            item_types: Filter by item types.
            include_paper_data: Include full paper metadata.
            include_extraction: Include extraction data.

        Returns:
            Merged and deduplicated list of FederatedResult objects.
        """
        # Collect results from all indexes
        all_results: list[FederatedResult] = []

        # Search primary index
        primary_results = self._search_single_index(
            engine=self.primary_engine,
            source_label="primary",
            weight=1.0,
            query=query,
            top_k=self.config.max_results_per_index,
            chunk_types=chunk_types,
            year_min=year_min,
            year_max=year_max,
            collections=collections,
            item_types=item_types,
            include_paper_data=include_paper_data,
            include_extraction=include_extraction,
        )
        all_results.extend(primary_results)

        # Search federated indexes in parallel
        if self.federated_engines:
            federated_results = self._search_federated_parallel(
                query=query,
                top_k=self.config.max_results_per_index,
                chunk_types=chunk_types,
                year_min=year_min,
                year_max=year_max,
                collections=collections,
                item_types=item_types,
                include_paper_data=include_paper_data,
                include_extraction=include_extraction,
            )
            all_results.extend(federated_results)

        # Deduplicate results
        deduplicated = self._deduplicate_results(all_results)

        # Merge using configured strategy
        merged = self._merge_results(deduplicated, top_k)

        logger.info(
            f"Federated search: {len(all_results)} raw -> {len(deduplicated)} deduped -> {len(merged)} final"
        )

        return merged

    def _search_single_index(
        self,
        engine: SearchEngine,
        source_label: str,
        weight: float,
        query: str,
        top_k: int,
        chunk_types: list[ChunkType] | None,
        year_min: int | None,
        year_max: int | None,
        collections: list[str] | None,
        item_types: list[str] | None,
        include_paper_data: bool,
        include_extraction: bool,
    ) -> list[FederatedResult]:
        """Search a single index and convert results to FederatedResult.

        Args:
            engine: SearchEngine instance.
            source_label: Label for this index.
            weight: Weight multiplier for this index.
            query: Search query.
            top_k: Number of results.
            chunk_types: Chunk type filter.
            year_min: Minimum year filter.
            year_max: Maximum year filter.
            collections: Collection filter.
            item_types: Item type filter.
            include_paper_data: Include paper metadata.
            include_extraction: Include extraction data.

        Returns:
            List of FederatedResult objects.
        """
        try:
            results = engine.search(
                query=query,
                top_k=top_k,
                chunk_types=chunk_types,
                year_min=year_min,
                year_max=year_max,
                collections=collections,
                item_types=item_types,
                include_paper_data=include_paper_data,
                include_extraction=include_extraction,
                deduplicate_papers=True,
            )
        except Exception as e:
            logger.error(f"Search failed for index {source_label}: {e}")
            return []

        # Convert to FederatedResult
        federated_results = []
        for result in results:
            federated_results.append(
                FederatedResult(
                    paper_id=result.paper_id,
                    title=result.title,
                    authors=result.authors,
                    year=result.year,
                    collections=result.collections,
                    item_type=result.item_type,
                    chunk_type=result.chunk_type,
                    matched_text=result.matched_text,
                    score=result.score,
                    paper_data=result.paper_data,
                    extraction_data=result.extraction_data,
                    source_index=source_label,
                    source_weight=weight,
                )
            )

        return federated_results

    def _search_federated_parallel(
        self,
        query: str,
        top_k: int,
        chunk_types: list[ChunkType] | None,
        year_min: int | None,
        year_max: int | None,
        collections: list[str] | None,
        item_types: list[str] | None,
        include_paper_data: bool,
        include_extraction: bool,
    ) -> list[FederatedResult]:
        """Search all federated indexes in parallel.

        Args:
            query: Search query.
            top_k: Number of results per index.
            chunk_types: Chunk type filter.
            year_min: Minimum year filter.
            year_max: Maximum year filter.
            collections: Collection filter.
            item_types: Item type filter.
            include_paper_data: Include paper metadata.
            include_extraction: Include extraction data.

        Returns:
            Combined results from all federated indexes.
        """
        all_federated: list[FederatedResult] = []

        with ThreadPoolExecutor(max_workers=len(self.federated_engines)) as executor:
            futures = {}
            for label, (engine, weight) in self.federated_engines.items():
                future = executor.submit(
                    self._search_single_index,
                    engine=engine,
                    source_label=label,
                    weight=weight,
                    query=query,
                    top_k=top_k,
                    chunk_types=chunk_types,
                    year_min=year_min,
                    year_max=year_max,
                    collections=collections,
                    item_types=item_types,
                    include_paper_data=include_paper_data,
                    include_extraction=include_extraction,
                )
                futures[future] = label

            for future in as_completed(futures):
                label = futures[future]
                try:
                    results = future.result()
                    all_federated.extend(results)
                    logger.debug(f"Got {len(results)} results from {label}")
                except Exception as e:
                    logger.error(f"Federated search failed for {label}: {e}")

        return all_federated

    def _deduplicate_results(
        self,
        results: list[FederatedResult],
    ) -> list[FederatedResult]:
        """Deduplicate results by DOI or title similarity.

        When duplicates are found, keeps the result from the highest-weighted index.

        Args:
            results: List of results to deduplicate.

        Returns:
            Deduplicated list of results.
        """
        if not results:
            return []

        # Group by DOI first
        doi_groups: dict[str, list[FederatedResult]] = {}
        no_doi_results: list[FederatedResult] = []

        for result in results:
            doi = _extract_doi(result.paper_data)
            if doi:
                if doi not in doi_groups:
                    doi_groups[doi] = []
                doi_groups[doi].append(result)
            else:
                no_doi_results.append(result)

        # Select best from each DOI group (highest weighted score)
        deduplicated: list[FederatedResult] = []
        for doi_results in doi_groups.values():
            best = max(doi_results, key=lambda r: r.weighted_score)
            deduplicated.append(best)

        # Deduplicate no-DOI results by title similarity
        for result in no_doi_results:
            is_duplicate = False
            for existing in deduplicated:
                similarity = _title_similarity(result.title, existing.title)
                if similarity >= self.config.dedup_threshold:
                    # Duplicate found - keep higher weighted one
                    if result.weighted_score > existing.weighted_score:
                        deduplicated.remove(existing)
                        deduplicated.append(result)
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(result)

        return deduplicated

    def _merge_results(
        self,
        results: list[FederatedResult],
        top_k: int,
    ) -> list[FederatedResult]:
        """Merge results using configured strategy.

        Args:
            results: Deduplicated results to merge.
            top_k: Number of results to return.

        Returns:
            Merged and ranked list of results.
        """
        if not results:
            return []

        strategy = self.config.merge_strategy

        if strategy == "interleave":
            return self._merge_interleave(results, top_k)
        elif strategy == "concat":
            return self._merge_concat(results, top_k)
        elif strategy == "rerank":
            return self._merge_rerank(results, top_k)
        else:
            logger.warning(f"Unknown merge strategy '{strategy}', using interleave")
            return self._merge_interleave(results, top_k)

    def _merge_interleave(
        self,
        results: list[FederatedResult],
        top_k: int,
    ) -> list[FederatedResult]:
        """Interleave results by weighted score.

        Takes results in order of weighted score, creating a balanced mix
        from all indexes.

        Args:
            results: Results to merge.
            top_k: Number of results to return.

        Returns:
            Interleaved results sorted by weighted score.
        """
        # Sort by weighted score descending
        sorted_results = sorted(results, key=lambda r: r.weighted_score, reverse=True)
        return sorted_results[:top_k]

    def _merge_concat(
        self,
        results: list[FederatedResult],
        top_k: int,
    ) -> list[FederatedResult]:
        """Concatenate results with primary index first.

        Shows all primary results before federated results, each group
        sorted by score.

        Args:
            results: Results to merge.
            top_k: Number of results to return.

        Returns:
            Concatenated results with primary first.
        """
        primary = [r for r in results if r.source_index == "primary"]
        federated = [r for r in results if r.source_index != "primary"]

        # Sort each group by score
        primary.sort(key=lambda r: r.score, reverse=True)
        federated.sort(key=lambda r: r.weighted_score, reverse=True)

        merged = primary + federated
        return merged[:top_k]

    def _merge_rerank(
        self,
        results: list[FederatedResult],
        top_k: int,
    ) -> list[FederatedResult]:
        """Rerank all results with combined scoring.

        Normalizes scores across indexes and re-ranks globally.

        Args:
            results: Results to merge.
            top_k: Number of results to return.

        Returns:
            Reranked results with normalized combined scores.
        """
        if not results:
            return []

        # Find score range for normalization
        all_scores = [r.score for r in results]
        min_score = min(all_scores)
        max_score = max(all_scores)
        score_range = max_score - min_score if max_score > min_score else 1.0

        # Normalize and apply weights
        for result in results:
            normalized = (result.score - min_score) / score_range
            result.weighted_score = normalized * result.source_weight

        # Sort by weighted score
        sorted_results = sorted(results, key=lambda r: r.weighted_score, reverse=True)
        return sorted_results[:top_k]

    def get_index_info(self) -> dict:
        """Get information about all configured indexes.

        Returns:
            Dictionary with index details.
        """
        info = {
            "primary": {
                "path": str(self.primary_index_dir),
                "enabled": True,
                "weight": 1.0,
            },
            "federated_enabled": self.config.enabled,
            "merge_strategy": self.config.merge_strategy,
            "dedup_threshold": self.config.dedup_threshold,
            "indexes": [],
        }

        for idx_config in self.config.indexes:
            info["indexes"].append({
                "label": idx_config.label,
                "path": str(idx_config.path),
                "enabled": idx_config.enabled,
                "weight": idx_config.weight,
                "loaded": idx_config.label in self.federated_engines,
            })

        return info
