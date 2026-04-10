"""Structured storage for papers, extractions, and full-text snapshots."""

import hashlib
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.analysis.dimensions import build_legacy_dimension_profile, get_dimension_value
from src.utils.file_utils import safe_read_json, safe_write_json
from src.utils.logging_config import get_logger

# Sentinel for "file did not exist at last check"
_NO_MTIME = -1.0

logger = get_logger(__name__)

SCHEMA_VERSION = "1.0"
EXTRACTION_STORE_SCHEMA_VERSION = "2.0.0"
FULLTEXT_STORE_SCHEMA_VERSION = "1.0.0"


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
        self.extractions_file = self.index_dir / "semantic_analyses.json"
        self.dimension_profile_file = self.index_dir / "dimension_profile.json"
        self.metadata_file = self.index_dir / "metadata.json"
        self.summary_file = self.index_dir / "summary.json"
        self.similarity_pairs_file = self.index_dir / "similarity_pairs.json"
        self.fulltext_dir = self.index_dir / "fulltext"
        self.fulltext_manifest_file = self.index_dir / "fulltext_manifest.json"

        # Cache loaded data with file modification tracking
        self._papers_cache: dict[str, dict] | None = None
        self._papers_mtime: float = _NO_MTIME
        self._extractions_cache: dict[str, dict] | None = None
        self._extractions_mtime: float = _NO_MTIME
        self._dimension_profile_cache: dict | None = None
        self._dimension_profile_mtime: float = _NO_MTIME
        self._fulltext_manifest_cache: dict[str, dict] | None = None
        self._fulltext_manifest_mtime: float = _NO_MTIME

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
            logger.debug("semantic_analyses.json changed on disk, reloading cache")

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
            "schema_version": EXTRACTION_STORE_SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(),
            "extraction_count": len(extractions),
            "extractions": extractions,
        }

        safe_write_json(self.extractions_file, data)
        self._extractions_cache = extractions
        self._extractions_mtime = self._file_mtime(self.extractions_file)
        logger.info(f"Saved {len(extractions)} extractions to {self.extractions_file}")

    def load_dimension_profile(self) -> dict:
        """Load the active dimension profile snapshot for the index."""

        if self._dimension_profile_cache is not None and not self._cache_stale(
            self.dimension_profile_file,
            self._dimension_profile_mtime,
        ):
            return self._dimension_profile_cache

        if self.dimension_profile_file.exists():
            data = safe_read_json(self.dimension_profile_file, default={})
        else:
            data = build_legacy_dimension_profile().model_dump(mode="json")

        self._dimension_profile_cache = data
        self._dimension_profile_mtime = self._file_mtime(self.dimension_profile_file)
        return self._dimension_profile_cache

    def save_dimension_profile(self, profile: dict) -> None:
        """Save the active dimension profile snapshot for the index."""

        safe_write_json(self.dimension_profile_file, profile)
        self._dimension_profile_cache = profile
        self._dimension_profile_mtime = self._file_mtime(self.dimension_profile_file)
        logger.info("Saved dimension profile snapshot to %s", self.dimension_profile_file)

    def _fulltext_path(self, paper_id: str) -> Path:
        """Return the canonical full-text snapshot path for a paper."""

        return self.fulltext_dir / f"{paper_id}.txt"

    def load_fulltext_manifest(self) -> dict[str, dict]:
        """Load per-paper full-text snapshot metadata."""

        if self._fulltext_manifest_cache is not None and not self._cache_stale(
            self.fulltext_manifest_file,
            self._fulltext_manifest_mtime,
        ):
            return self._fulltext_manifest_cache

        if (
            not self.fulltext_manifest_file.exists()
            and self.fulltext_dir.exists()
            and any(self.fulltext_dir.glob("*.txt"))
        ):
            logger.warning(
                "Full-text manifest missing; rebuilding from snapshot files in %s",
                self.fulltext_dir,
            )
            rebuilt = self.rebuild_fulltext_manifest()
            self._fulltext_manifest_cache = rebuilt
            self._fulltext_manifest_mtime = self._file_mtime(self.fulltext_manifest_file)
            return self._fulltext_manifest_cache

        data = safe_read_json(self.fulltext_manifest_file, default={})
        if isinstance(data, dict) and "snapshots" in data:
            snapshots = data["snapshots"]
        elif isinstance(data, dict):
            snapshots = data
        else:
            snapshots = {}

        self._fulltext_manifest_cache = snapshots
        self._fulltext_manifest_mtime = self._file_mtime(self.fulltext_manifest_file)
        return self._fulltext_manifest_cache

    def save_fulltext_manifest(self, snapshots: dict[str, dict]) -> None:
        """Persist full-text snapshot metadata."""

        data = {
            "schema_version": FULLTEXT_STORE_SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(),
            "snapshot_count": len(snapshots),
            "snapshots": snapshots,
        }
        safe_write_json(self.fulltext_manifest_file, data)
        self._fulltext_manifest_cache = snapshots
        self._fulltext_manifest_mtime = self._file_mtime(self.fulltext_manifest_file)
        logger.info(
            "Saved full-text manifest for %d papers to %s",
            len(snapshots),
            self.fulltext_manifest_file,
        )

    def flush_fulltext_manifest(self) -> None:
        """Flush in-memory full-text manifest changes to disk.

        If the on-disk manifest has been modified since this store last
        loaded it (for example by another ``StructuredStore`` instance
        such as the extraction pipeline subroutine writing fresh text
        snapshots), the disk state is reloaded and merged with the local
        cache before saving. The cache wins on key collisions, so any
        local additions or rewrites are preserved while entries that
        only exist on disk are not clobbered.
        """

        if self._fulltext_manifest_cache is None:
            return

        if self._cache_stale(
            self.fulltext_manifest_file,
            self._fulltext_manifest_mtime,
        ):
            data = safe_read_json(self.fulltext_manifest_file, default={})
            if isinstance(data, dict) and "snapshots" in data:
                disk_snapshots = data["snapshots"]
            elif isinstance(data, dict):
                disk_snapshots = data
            else:
                disk_snapshots = {}

            merged: dict[str, dict] = dict(disk_snapshots)
            merged.update(self._fulltext_manifest_cache)
            added_from_disk = set(disk_snapshots) - set(self._fulltext_manifest_cache)
            if added_from_disk:
                logger.info(
                    "Merging %d full-text manifest entries from disk during "
                    "flush (cache was stale)",
                    len(added_from_disk),
                )
            self._fulltext_manifest_cache = merged

        self.save_fulltext_manifest(self._fulltext_manifest_cache)

    def rebuild_fulltext_manifest(self) -> dict[str, dict]:
        """Rebuild the full-text manifest from snapshot files on disk.

        Returns:
            Reconstructed snapshot metadata keyed by ``paper_id``.
        """

        snapshots: dict[str, dict] = {}
        if not self.fulltext_dir.exists():
            self._fulltext_manifest_cache = snapshots
            self.save_fulltext_manifest(snapshots)
            return snapshots

        for snapshot_path in sorted(self.fulltext_dir.glob("*.txt")):
            paper_id = snapshot_path.stem
            try:
                text = snapshot_path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning(
                    "Failed to rebuild full-text manifest entry for %s: %s",
                    paper_id,
                    exc,
                )
                continue

            stat = snapshot_path.stat()
            snapshots[paper_id] = {
                "paper_id": paper_id,
                "path": str(snapshot_path.relative_to(self.index_dir)),
                "char_count": len(text),
                "word_count": len(text.split()),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "captured_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_cleaned": True,
                "is_truncated_for_llm": False,
                "source": "manifest_rebuild",
                "reconstructed": True,
            }

        self.save_fulltext_manifest(snapshots)
        return snapshots

    def save_text_snapshot(
        self,
        paper_id: str,
        text: str,
        metadata: dict | None = None,
        persist_manifest: bool = True,
    ) -> dict:
        """Save a canonical cleaned full-text snapshot for one paper.

        Args:
            paper_id: Paper identifier.
            text: Full cleaned text to persist verbatim.
            metadata: Optional sidecar metadata to merge into the manifest entry.
            persist_manifest: Persist the updated manifest immediately.

        Returns:
            Metadata entry recorded in the full-text manifest.
        """

        self.fulltext_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = self._fulltext_path(paper_id)
        snapshot_path.write_text(text, encoding="utf-8")

        entry = {
            "paper_id": paper_id,
            "path": str(snapshot_path.relative_to(self.index_dir)),
            "char_count": len(text),
            "word_count": len(text.split()),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "captured_at": datetime.now().isoformat(),
            "is_cleaned": True,
            "is_truncated_for_llm": False,
        }
        if metadata:
            entry.update(metadata)

        snapshots = dict(self.load_fulltext_manifest())
        snapshots[paper_id] = entry
        self._fulltext_manifest_cache = snapshots
        if persist_manifest:
            self.save_fulltext_manifest(snapshots)
        return entry

    def load_text_snapshot(self, paper_id: str) -> dict | None:
        """Load a full-text snapshot plus its metadata for one paper."""

        manifest = self.load_fulltext_manifest()
        entry = manifest.get(paper_id)
        if not entry:
            return None

        snapshot_path = self.index_dir / entry.get("path", "")
        if not snapshot_path.exists():
            logger.warning("Full-text snapshot missing on disk for %s", paper_id)
            return None

        try:
            text = snapshot_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read full-text snapshot for %s: %s", paper_id, exc)
            return None

        return {**entry, "text": text}

    def delete_text_snapshot(self, paper_id: str, persist_manifest: bool = True) -> None:
        """Delete a paper's canonical full-text snapshot and manifest entry."""

        snapshot_path = self._fulltext_path(paper_id)
        if snapshot_path.exists():
            try:
                snapshot_path.unlink()
            except OSError as exc:
                logger.warning("Failed to remove full-text snapshot for %s: %s", paper_id, exc)

        snapshots = dict(self.load_fulltext_manifest())
        if paper_id in snapshots:
            snapshots.pop(paper_id, None)
            self._fulltext_manifest_cache = snapshots
            if persist_manifest:
                self.save_fulltext_manifest(snapshots)

    def get_fulltext_context(
        self,
        paper_id: str,
        query: str,
        *,
        max_hits: int = 3,
        context_chars: int = 400,
        case_sensitive: bool = False,
    ) -> dict:
        """Return verbatim context windows for a query within a paper snapshot."""

        snapshot = self.load_text_snapshot(paper_id)
        if snapshot is None:
            return {
                "paper_id": paper_id,
                "found": False,
                "error": f"No full-text snapshot available for {paper_id}",
                "query": query,
                "matches": [],
            }

        text = snapshot["text"]
        flags = 0 if case_sensitive else re.IGNORECASE
        matches = []
        for match in re.finditer(re.escape(query), text, flags):
            start = max(0, match.start() - context_chars)
            end = min(len(text), match.end() + context_chars)
            matches.append(
                {
                    "match_text": text[match.start():match.end()],
                    "context": text[start:end],
                    "start_char": match.start(),
                    "end_char": match.end(),
                    "context_start_char": start,
                    "context_end_char": end,
                }
            )
            if len(matches) >= max_hits:
                break

        return {
            "paper_id": paper_id,
            "found": True,
            "query": query,
            "match_count": len(matches),
            "matches": matches,
            "fulltext_metadata": {
                key: value
                for key, value in snapshot.items()
                if key != "text"
            },
        }

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
        fulltext = self.load_text_snapshot(paper_id)
        return {
            "paper": paper,
            "extraction": extraction,
            "fulltext": (
                {key: value for key, value in fulltext.items() if key != "text"}
                if fulltext
                else None
            ),
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

        # Extract field/discipline from q17_field
        discipline_counts = Counter()
        for ext in extractions.values():
            field_val = get_dimension_value(ext, "field") or ""
            if field_val:
                discipline_counts[field_val] += 1

        # Similarity pairs stats
        similarity_stats = {}
        if self.similarity_pairs_file.exists():
            pairs_data = safe_read_json(self.similarity_pairs_file, default={})
            similarity_stats = {
                "total_source_papers": pairs_data.get("total_source_papers", 0),
                "total_pairs": pairs_data.get("total_pairs", 0),
                "generated_at": pairs_data.get("generated_at"),
            }

        fulltext_manifest = self.load_fulltext_manifest()
        fulltext_stats = {}
        if fulltext_manifest:
            fulltext_stats = {
                "snapshot_count": len(fulltext_manifest),
                "total_characters": sum(
                    int(entry.get("char_count", 0) or 0)
                    for entry in fulltext_manifest.values()
                ),
                "generated_at": safe_read_json(
                    self.fulltext_manifest_file,
                    default={},
                ).get("generated_at"),
            }

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

        if similarity_stats:
            summary["similarity_pairs"] = similarity_stats
        if fulltext_stats:
            summary["fulltext"] = fulltext_stats

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

    def load_similarity_pairs(self) -> dict[str, list[dict]]:
        """Load pre-computed similarity pairs.

        Returns:
            Dictionary mapping paper_id to list of similar paper entries,
            each containing: similar_paper_id, similarity_score.
        """
        data = safe_read_json(self.similarity_pairs_file, default={})
        if isinstance(data, dict) and "pairs" in data:
            return data["pairs"]
        return data if isinstance(data, dict) else {}

    def save_similarity_pairs(
        self,
        pairs: dict[str, list[dict]],
        metadata: dict | None = None,
    ) -> None:
        """Save pre-computed similarity pairs.

        Args:
            pairs: Dictionary mapping paper_id to list of similar papers.
            metadata: Optional metadata about the computation.
        """
        data = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(),
            "total_source_papers": len(pairs),
            "total_pairs": sum(len(v) for v in pairs.values()),
            **(metadata or {}),
            "pairs": pairs,
        }
        safe_write_json(self.similarity_pairs_file, data)
        logger.info(
            f"Saved {data['total_pairs']} similarity pairs "
            f"for {len(pairs)} papers to {self.similarity_pairs_file}"
        )

    def clear_cache(self) -> None:
        """Clear in-memory caches, forcing reload on next access."""
        self._papers_cache = None
        self._papers_mtime = _NO_MTIME
        self._extractions_cache = None
        self._extractions_mtime = _NO_MTIME
        self._dimension_profile_cache = None
        self._dimension_profile_mtime = _NO_MTIME
        self._fulltext_manifest_cache = None
        self._fulltext_manifest_mtime = _NO_MTIME

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
