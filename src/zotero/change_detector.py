"""Detect changes in Zotero library for incremental updates."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from src.indexing.structured_store import StructuredStore
from src.utils.logging_config import get_logger
from src.zotero.database import ZoteroDatabase

logger = get_logger(__name__)


class ChangeSet(NamedTuple):
    """Detected changes between Zotero library and index."""

    new_items: list[str]
    modified_items: list[str]
    deleted_items: list[str]
    unchanged_items: list[str]

    @property
    def has_changes(self) -> bool:
        """Check if any changes were detected."""
        return bool(self.new_items or self.modified_items or self.deleted_items)

    @property
    def total_changes(self) -> int:
        """Total number of items that need processing."""
        return len(self.new_items) + len(self.modified_items) + len(self.deleted_items)

    def summary(self) -> str:
        """Get human-readable summary of changes."""
        parts = []
        if self.new_items:
            parts.append(f"{len(self.new_items)} new")
        if self.modified_items:
            parts.append(f"{len(self.modified_items)} modified")
        if self.deleted_items:
            parts.append(f"{len(self.deleted_items)} deleted")
        if not parts:
            return "No changes detected"
        return ", ".join(parts)


@dataclass
class IndexState:
    """State of an indexed paper for change detection."""

    paper_id: str
    zotero_key: str
    date_modified: datetime
    date_indexed: datetime | None = None


@dataclass
class ChangeDetector:
    """Detect changes between Zotero library and indexed papers.

    Compares the Zotero database against the structured store to identify:
    - New papers: In Zotero but not in index
    - Modified papers: In both but Zotero has newer modification date
    - Deleted papers: In index but not in Zotero
    """

    zotero_db: ZoteroDatabase
    index_dir: Path
    _store: StructuredStore = field(init=False)
    _indexed_state: dict[str, IndexState] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Initialize the structured store."""
        self._store = StructuredStore(self.index_dir)

    def _load_indexed_state(self) -> dict[str, IndexState]:
        """Load the current state of indexed papers.

        Returns:
            Dictionary mapping zotero_key to IndexState.
        """
        papers = self._store.load_papers()
        state = {}

        for paper_data in papers.values():
            zotero_key = paper_data.get("zotero_key")
            if not zotero_key:
                continue

            # Parse dates
            date_modified = None
            if dm := paper_data.get("date_modified"):
                try:
                    date_modified = datetime.fromisoformat(dm.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            date_indexed = None
            if di := paper_data.get("indexed_at") or paper_data.get("date_added"):
                try:
                    date_indexed = datetime.fromisoformat(di.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            state[zotero_key] = IndexState(
                paper_id=paper_data["paper_id"],
                zotero_key=zotero_key,
                date_modified=date_modified or datetime.min,
                date_indexed=date_indexed,
            )

        return state

    def _get_zotero_items(self) -> dict[str, dict]:
        """Get all items from Zotero database.

        Returns:
            Dictionary mapping zotero_key to item info dict.
        """
        items = self.zotero_db.get_all_items_with_pdfs()
        return {item["key"]: item for item in items}

    def detect_changes(self) -> ChangeSet:
        """Detect all changes between Zotero and index.

        Returns:
            ChangeSet with categorized items.
        """
        logger.info("Detecting changes between Zotero and index...")

        # Load current states
        indexed_state = self._load_indexed_state()
        zotero_items = self._get_zotero_items()

        logger.info(f"Index contains {len(indexed_state)} papers")
        logger.info(f"Zotero contains {len(zotero_items)} papers with PDFs")

        indexed_keys = set(indexed_state.keys())
        zotero_keys = set(zotero_items.keys())

        # Detect new items (in Zotero but not in index)
        new_keys = zotero_keys - indexed_keys
        new_items = list(new_keys)

        # Detect deleted items (in index but not in Zotero)
        deleted_keys = indexed_keys - zotero_keys
        deleted_items = [indexed_state[k].paper_id for k in deleted_keys]

        # Detect modified items (in both, but Zotero is newer)
        common_keys = indexed_keys & zotero_keys
        modified_items = []
        unchanged_items = []

        for key in common_keys:
            indexed = indexed_state[key]
            zotero_item = zotero_items[key]

            # Parse Zotero modification date
            try:
                zotero_modified = datetime.fromisoformat(
                    zotero_item["date_modified"].replace(" ", "T")
                )
            except (ValueError, TypeError, KeyError):
                # If we can't parse the date, assume unchanged
                unchanged_items.append(indexed.paper_id)
                continue

            # Compare modification dates
            if zotero_modified > indexed.date_modified:
                modified_items.append(indexed.paper_id)
                logger.debug(
                    f"Modified: {key} (indexed: {indexed.date_modified}, "
                    f"zotero: {zotero_modified})"
                )
            else:
                unchanged_items.append(indexed.paper_id)

        changes = ChangeSet(
            new_items=new_items,
            modified_items=modified_items,
            deleted_items=deleted_items,
            unchanged_items=unchanged_items,
        )

        logger.info(f"Change detection complete: {changes.summary()}")
        return changes

    def detect_new_items(self) -> list[str]:
        """Detect only new items (not in index).

        Returns:
            List of Zotero keys for new items.
        """
        indexed_state = self._load_indexed_state()
        zotero_items = self._get_zotero_items()

        new_keys = set(zotero_items.keys()) - set(indexed_state.keys())
        logger.info(f"Found {len(new_keys)} new items")
        return list(new_keys)

    def detect_modified_items(self) -> list[str]:
        """Detect only modified items.

        Returns:
            List of paper_ids for modified items.
        """
        indexed_state = self._load_indexed_state()
        zotero_items = self._get_zotero_items()

        common_keys = set(indexed_state.keys()) & set(zotero_items.keys())
        modified = []

        for key in common_keys:
            indexed = indexed_state[key]
            zotero_item = zotero_items[key]

            try:
                zotero_modified = datetime.fromisoformat(
                    zotero_item["date_modified"].replace(" ", "T")
                )
                if zotero_modified > indexed.date_modified:
                    modified.append(indexed.paper_id)
            except (ValueError, TypeError, KeyError):
                continue

        logger.info(f"Found {len(modified)} modified items")
        return modified

    def detect_deleted_items(self) -> list[str]:
        """Detect only deleted items (in index but not in Zotero).

        Returns:
            List of paper_ids for deleted items.
        """
        indexed_state = self._load_indexed_state()
        zotero_items = self._get_zotero_items()

        deleted_keys = set(indexed_state.keys()) - set(zotero_items.keys())
        deleted = [indexed_state[k].paper_id for k in deleted_keys]
        logger.info(f"Found {len(deleted)} deleted items")
        return deleted

    def get_paper_id_for_key(self, zotero_key: str) -> str | None:
        """Get the paper_id for a Zotero key if it exists in the index.

        Args:
            zotero_key: Zotero item key.

        Returns:
            paper_id if found, None otherwise.
        """
        indexed_state = self._load_indexed_state()
        if zotero_key in indexed_state:
            return indexed_state[zotero_key].paper_id
        return None

    def get_stats(self) -> dict:
        """Get statistics about current state.

        Returns:
            Dictionary with counts and change summary.
        """
        indexed_state = self._load_indexed_state()
        zotero_items = self._get_zotero_items()
        changes = self.detect_changes()

        return {
            "indexed_papers": len(indexed_state),
            "zotero_papers": len(zotero_items),
            "new_items": len(changes.new_items),
            "modified_items": len(changes.modified_items),
            "deleted_items": len(changes.deleted_items),
            "unchanged_items": len(changes.unchanged_items),
            "has_changes": changes.has_changes,
            "summary": changes.summary(),
        }
