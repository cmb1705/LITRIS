"""Structured storage for papers and extractions using JSON files."""

from collections import Counter
from datetime import datetime
from pathlib import Path

from src.utils.file_utils import safe_read_json, safe_write_json
from src.utils.logging_config import get_logger

# Sentinel for "file did not exist at last check"
_NO_MTIME = -1.0

logger = get_logger(__name__)

SCHEMA_VERSION = "1.0"


class StructuredStore:
    """JSON-based storage for papers, extractions, and metadata."""

    def __init__(self, index_dir: Path | str):
        """Initialize structured store.

        Args:
            index_dir: Directory for index files.
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.papers_file = self.index_dir / "papers.json"
        self.extractions_file = self.index_dir / "extractions.json"
        self.metadata_file = self.index_dir / "metadata.json"
        self.summary_file = self.index_dir / "summary.json"

        # Cache loaded data with file modification tracking
        self._papers_cache: dict[str, dict] | None = None
        self._papers_mtime: float = _NO_MTIME
        self._extractions_cache: dict[str, dict] | None = None
        self._extractions_mtime: float = _NO_MTIME

    def _file_mtime(self, path: Path) -> float:
        """Get file modification time, or sentinel if file does not exist."""
        try:
            return path.stat().st_mtime
        except OSError:
            return _NO_MTIME

    def _cache_stale(self, path: Path, cached_mtime: float) -> bool:
        """Check whether a cached file has been modified on disk."""
        return self._file_mtime(path) != cached_mtime

    def load_papers(self) -> dict[str, dict]:
        """Load papers from JSON file.

        Automatically reloads from disk if the file has been modified
        since the last read (e.g. by a concurrent build process).

        Returns:
            Dictionary mapping paper_id to paper data.
        """
        if self._papers_cache is not None and not self._cache_stale(
            self.papers_file, self._papers_mtime
        ):
            return self._papers_cache

        if self._papers_cache is not None:
            logger.debug("papers.json changed on disk, reloading cache")

        data = safe_read_json(self.papers_file, default={"papers": []})
        papers_list = data.get("papers", data) if isinstance(data, dict) else data

        # Convert to dictionary by paper_id
        if isinstance(papers_list, list):
            self._papers_cache = {p["paper_id"]: p for p in papers_list if "paper_id" in p}
        else:
            self._papers_cache = papers_list

        self._papers_mtime = self._file_mtime(self.papers_file)
        return self._papers_cache

    def load_extractions(self) -> dict[str, dict]:
        """Load extractions from JSON file.

        Automatically reloads from disk if the file has been modified
        since the last read (e.g. by a concurrent build process).

        Returns:
            Dictionary mapping paper_id to extraction data.
        """
        if self._extractions_cache is not None and not self._cache_stale(
            self.extractions_file, self._extractions_mtime
        ):
            return self._extractions_cache

        if self._extractions_cache is not None:
            logger.debug("extractions.json changed on disk, reloading cache")

        data = safe_read_json(self.extractions_file, default={})

        # Handle both formats: dict or list
        if isinstance(data, dict) and "extractions" in data:
            extractions_list = data["extractions"]
            if isinstance(extractions_list, list):
                self._extractions_cache = {
                    e["paper_id"]: e for e in extractions_list if "paper_id" in e
                }
            else:
                self._extractions_cache = extractions_list
        elif isinstance(data, list):
            self._extractions_cache = {
                e["paper_id"]: e for e in data if "paper_id" in e
            }
        else:
            self._extractions_cache = data

        self._extractions_mtime = self._file_mtime(self.extractions_file)
        return self._extractions_cache

    def save_papers(self, papers: list[dict] | dict[str, dict]) -> None:
        """Save papers to JSON file.

        Args:
            papers: List of paper dicts or dict mapping paper_id to paper data.
        """
        if isinstance(papers, dict):
            papers_list = list(papers.values())
        else:
            papers_list = papers

        data = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(),
            "paper_count": len(papers_list),
            "papers": papers_list,
        }

        safe_write_json(self.papers_file, data)
        self._papers_cache = {p["paper_id"]: p for p in papers_list if "paper_id" in p}
        self._papers_mtime = self._file_mtime(self.papers_file)
        logger.info(f"Saved {len(papers_list)} papers to {self.papers_file}")

    def save_extractions(self, extractions: dict[str, dict]) -> None:
        """Save extractions to JSON file.

        Args:
            extractions: Dictionary mapping paper_id to extraction data.
        """
        data = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(),
            "extraction_count": len(extractions),
            "extractions": extractions,
        }

        safe_write_json(self.extractions_file, data)
        self._extractions_cache = extractions
        self._extractions_mtime = self._file_mtime(self.extractions_file)
        logger.info(f"Saved {len(extractions)} extractions to {self.extractions_file}")

    def get_paper(self, paper_id: str) -> dict | None:
        """Get a single paper by ID.

        Args:
            paper_id: Paper ID to retrieve.

        Returns:
            Paper data dictionary or None if not found.
        """
        papers = self.load_papers()
        return papers.get(paper_id)

    def get_extraction(self, paper_id: str) -> dict | None:
        """Get extraction for a paper.

        Args:
            paper_id: Paper ID to retrieve extraction for.

        Returns:
            Extraction data dictionary or None if not found.
        """
        extractions = self.load_extractions()
        return extractions.get(paper_id)

    def get_paper_with_extraction(self, paper_id: str) -> dict | None:
        """Get paper with its extraction combined.

        Args:
            paper_id: Paper ID to retrieve.

        Returns:
            Combined paper and extraction data, or None if not found.
        """
        paper = self.get_paper(paper_id)
        if not paper:
            return None

        extraction = self.get_extraction(paper_id)
        return {
            "paper": paper,
            "extraction": extraction,
        }

    def search_papers(
        self,
        title_contains: str | None = None,
        author_contains: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        collection: str | None = None,
        item_type: str | None = None,
    ) -> list[dict]:
        """Search papers by metadata fields.

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
        papers = self.load_papers()
        results = []

        for paper in papers.values():
            # Title filter
            if title_contains:
                title = paper.get("title", "").lower()
                if title_contains.lower() not in title:
                    continue

            # Author filter
            if author_contains:
                authors = paper.get("author_string", "") or ""
                if isinstance(paper.get("authors"), list):
                    authors = " ".join(
                        a.get("full_name", "") for a in paper["authors"]
                    )
                if author_contains.lower() not in authors.lower():
                    continue

            # Year filter
            year = paper.get("publication_year")
            if year_min is not None and year and int(year) < year_min:
                continue
            if year_max is not None and year and int(year) > year_max:
                continue

            # Collection filter
            if collection:
                collections = paper.get("collections", [])
                if collection not in collections:
                    continue

            # Item type filter
            if item_type:
                if paper.get("item_type") != item_type:
                    continue

            results.append(paper)

        return results

    def generate_summary(self) -> dict:
        """Generate summary statistics for the index.

        Returns:
            Summary statistics dictionary.
        """
        papers = self.load_papers()
        extractions = self.load_extractions()

        # Count by type
        type_counts = Counter(p.get("item_type", "unknown") for p in papers.values())

        # Count by year
        year_counts = Counter()
        for p in papers.values():
            year = p.get("publication_year")
            if year:
                year_counts[str(year)] += 1

        # Count by collection
        collection_counts = Counter()
        for p in papers.values():
            for coll in p.get("collections", []):
                collection_counts[coll] += 1

        # Get recent papers (last 10 by date_added)
        sorted_papers = sorted(
            papers.values(),
            key=lambda p: p.get("date_added", ""),
            reverse=True,
        )
        recent_papers = [
            {
                "paper_id": p["paper_id"],
                "title": p.get("title", ""),
                "authors": p.get("author_string", ""),
                "year": p.get("publication_year"),
            }
            for p in sorted_papers[:10]
        ]

        # Extract discipline tags
        discipline_counts = Counter()
        for ext in extractions.values():
            ext_data = ext.get("extraction", ext)
            for tag in ext_data.get("discipline_tags", []):
                discipline_counts[tag] += 1

        summary = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(),
            "total_papers": len(papers),
            "total_extractions": len(extractions),
            "papers_by_type": dict(type_counts),
            "papers_by_year": dict(sorted(year_counts.items())),
            "papers_by_collection": dict(collection_counts.most_common(20)),
            "top_disciplines": dict(discipline_counts.most_common(20)),
            "recent_papers": recent_papers,
        }

        return summary

    def save_summary(self, summary: dict | None = None) -> dict:
        """Generate and save summary statistics.

        Args:
            summary: Pre-computed summary, or None to generate.

        Returns:
            The saved summary dictionary.
        """
        if summary is None:
            summary = self.generate_summary()

        safe_write_json(self.summary_file, summary)
        logger.info(f"Saved summary to {self.summary_file}")
        return summary

    def load_summary(self) -> dict:
        """Load existing summary file.

        Returns:
            Summary dictionary.
        """
        return safe_read_json(self.summary_file, default={})

    def save_metadata(
        self,
        extraction_mode: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
        total_input_tokens: int = 0,
        total_output_tokens: int = 0,
        estimated_cost: float = 0.0,
        failed_extractions: list[dict] | None = None,
    ) -> dict:
        """Save index metadata.

        Args:
            extraction_mode: CLI or batch_api mode.
            model: LLM model used.
            embedding_model: Embedding model used.
            total_input_tokens: Total input tokens used.
            total_output_tokens: Total output tokens used.
            estimated_cost: Estimated cost in USD.
            failed_extractions: List of failed extraction records.

        Returns:
            The saved metadata dictionary.
        """
        papers = self.load_papers()
        extractions = self.load_extractions()

        # Generate statistics
        type_counts = Counter(p.get("item_type", "unknown") for p in papers.values())
        year_counts = Counter()
        for p in papers.values():
            year = p.get("publication_year")
            if year:
                year_counts[str(year)] += 1

        collection_counts = Counter()
        for p in papers.values():
            for coll in p.get("collections", []):
                collection_counts[coll] += 1

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "index_name": "LITRIS_Index",
            "created_at": datetime.now().isoformat(),
            "last_full_build": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "statistics": {
                "total_papers": len(papers),
                "total_extractions": len(extractions),
                "papers_by_type": dict(type_counts),
                "papers_by_year": dict(sorted(year_counts.items())),
                "papers_by_collection": dict(collection_counts),
            },
            "processing": {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "estimated_cost_usd": estimated_cost,
                "extraction_model": model,
                "extraction_mode": extraction_mode,
                "embedding_model": embedding_model,
            },
            "failed_extractions": failed_extractions or [],
        }

        safe_write_json(self.metadata_file, metadata)
        logger.info(f"Saved metadata to {self.metadata_file}")
        return metadata

    def load_metadata(self) -> dict:
        """Load existing metadata file.

        Returns:
            Metadata dictionary.
        """
        return safe_read_json(self.metadata_file, default={})

    def clear_cache(self) -> None:
        """Clear in-memory caches, forcing reload on next access."""
        self._papers_cache = None
        self._papers_mtime = _NO_MTIME
        self._extractions_cache = None
        self._extractions_mtime = _NO_MTIME

    def get_paper_ids(self) -> set[str]:
        """Get set of all paper IDs in the store.

        Returns:
            Set of paper IDs.
        """
        papers = self.load_papers()
        return set(papers.keys())

    def get_extracted_paper_ids(self) -> set[str]:
        """Get set of paper IDs that have extractions.

        Returns:
            Set of paper IDs with extractions.
        """
        extractions = self.load_extractions()
        return set(extractions.keys())

    def get_missing_extractions(self) -> list[str]:
        """Get paper IDs that are missing extractions.

        Returns:
            List of paper IDs without extractions.
        """
        paper_ids = self.get_paper_ids()
        extracted_ids = self.get_extracted_paper_ids()
        return list(paper_ids - extracted_ids)
