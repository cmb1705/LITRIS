"""State management for incremental index updates."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from src.utils.file_utils import safe_read_json, safe_write_json
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

STATE_FILE = "update_state.json"
SCHEMA_VERSION = "1.0"


class UpdateRecord(NamedTuple):
    """Record of a single update operation."""

    timestamp: str
    update_type: str  # "full", "incremental", "manual"
    new_added: int
    modified_updated: int
    deleted_removed: int
    failed: int
    duration_seconds: float


@dataclass
class UpdateState:
    """Manages state for incremental updates.

    Tracks:
    - Last update timestamp
    - Update history
    - Paper version tracking
    - Build type (full vs incremental)
    """

    index_dir: Path
    _state: dict = field(default_factory=dict, init=False)
    _state_file: Path = field(init=False)

    def __post_init__(self):
        """Initialize state file path."""
        self._state_file = self.index_dir / STATE_FILE
        self._load()

    def _load(self) -> None:
        """Load state from file."""
        if self._state_file.exists():
            self._state = safe_read_json(self._state_file, default={})
        else:
            self._state = self._default_state()

    def _default_state(self) -> dict:
        """Create default state structure."""
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now().isoformat(),
            "last_full_build": None,
            "last_update": None,
            "update_count": 0,
            "update_history": [],
            "paper_versions": {},
        }

    def save(self) -> None:
        """Save state to file."""
        safe_write_json(self._state_file, self._state)
        logger.debug(f"Saved update state to {self._state_file}")

    def get_last_update(self) -> datetime | None:
        """Get timestamp of last update.

        Returns:
            Datetime of last update, or None if never updated.
        """
        last = self._state.get("last_update")
        if last:
            try:
                return datetime.fromisoformat(last)
            except (ValueError, TypeError):
                return None
        return None

    def get_last_full_build(self) -> datetime | None:
        """Get timestamp of last full build.

        Returns:
            Datetime of last full build, or None if never built.
        """
        last = self._state.get("last_full_build")
        if last:
            try:
                return datetime.fromisoformat(last)
            except (ValueError, TypeError):
                return None
        return None

    def record_full_build(
        self,
        papers_count: int,
        failed_count: int = 0,
        duration_seconds: float = 0.0,
    ) -> None:
        """Record a full index build.

        Args:
            papers_count: Number of papers processed.
            failed_count: Number of failed extractions.
            duration_seconds: Build duration in seconds.
        """
        now = datetime.now().isoformat()

        self._state["last_full_build"] = now
        self._state["last_update"] = now
        self._state["update_count"] = self._state.get("update_count", 0) + 1

        record = UpdateRecord(
            timestamp=now,
            update_type="full",
            new_added=papers_count,
            modified_updated=0,
            deleted_removed=0,
            failed=failed_count,
            duration_seconds=duration_seconds,
        )

        self._add_history_record(record)
        self.save()
        logger.info(f"Recorded full build: {papers_count} papers")

    def record_incremental_update(
        self,
        new_added: int = 0,
        modified_updated: int = 0,
        deleted_removed: int = 0,
        failed: int = 0,
        duration_seconds: float = 0.0,
    ) -> None:
        """Record an incremental update.

        Args:
            new_added: Number of new papers added.
            modified_updated: Number of modified papers updated.
            deleted_removed: Number of deleted papers removed.
            failed: Number of failed operations.
            duration_seconds: Update duration in seconds.
        """
        now = datetime.now().isoformat()

        self._state["last_update"] = now
        self._state["update_count"] = self._state.get("update_count", 0) + 1

        record = UpdateRecord(
            timestamp=now,
            update_type="incremental",
            new_added=new_added,
            modified_updated=modified_updated,
            deleted_removed=deleted_removed,
            failed=failed,
            duration_seconds=duration_seconds,
        )

        self._add_history_record(record)
        self.save()
        logger.info(
            f"Recorded incremental update: +{new_added}, ~{modified_updated}, -{deleted_removed}"
        )

    def _add_history_record(self, record: UpdateRecord) -> None:
        """Add a record to update history.

        Args:
            record: UpdateRecord to add.
        """
        history = self._state.get("update_history", [])
        history.append(record._asdict())

        # Keep only last 100 records
        if len(history) > 100:
            history = history[-100:]

        self._state["update_history"] = history

    def get_update_history(self, limit: int = 10) -> list[UpdateRecord]:
        """Get recent update history.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of UpdateRecord objects, most recent first.
        """
        history = self._state.get("update_history", [])
        records = []

        for item in reversed(history[-limit:]):
            try:
                records.append(UpdateRecord(**item))
            except (TypeError, KeyError):
                continue

        return records

    def set_paper_version(self, paper_id: str, zotero_key: str, date_modified: str) -> None:
        """Track a paper's version.

        Args:
            paper_id: Paper ID.
            zotero_key: Zotero item key.
            date_modified: Modification date string.
        """
        versions = self._state.get("paper_versions", {})
        versions[paper_id] = {
            "zotero_key": zotero_key,
            "date_modified": date_modified,
            "indexed_at": datetime.now().isoformat(),
        }
        self._state["paper_versions"] = versions
        self.save()

    def get_paper_version(self, paper_id: str) -> dict | None:
        """Get tracked version for a paper.

        Args:
            paper_id: Paper ID.

        Returns:
            Version dict with zotero_key, date_modified, indexed_at.
        """
        versions = self._state.get("paper_versions", {})
        return versions.get(paper_id)

    def remove_paper_version(self, paper_id: str) -> None:
        """Remove version tracking for a paper.

        Args:
            paper_id: Paper ID to remove.
        """
        versions = self._state.get("paper_versions", {})
        if paper_id in versions:
            del versions[paper_id]
            self._state["paper_versions"] = versions
            self.save()

    def needs_full_rebuild(self) -> bool:
        """Check if a full rebuild is recommended.

        Returns:
            True if no full build has ever been done or state is corrupted.
        """
        return self._state.get("last_full_build") is None

    def get_stats(self) -> dict:
        """Get state statistics.

        Returns:
            Dictionary with state statistics.
        """
        history = self._state.get("update_history", [])
        versions = self._state.get("paper_versions", {})

        # Calculate totals from history
        total_added = sum(r.get("new_added", 0) for r in history)
        total_modified = sum(r.get("modified_updated", 0) for r in history)
        total_deleted = sum(r.get("deleted_removed", 0) for r in history)
        total_failed = sum(r.get("failed", 0) for r in history)

        return {
            "created_at": self._state.get("created_at"),
            "last_full_build": self._state.get("last_full_build"),
            "last_update": self._state.get("last_update"),
            "update_count": self._state.get("update_count", 0),
            "tracked_papers": len(versions),
            "history_records": len(history),
            "total_added": total_added,
            "total_modified": total_modified,
            "total_deleted": total_deleted,
            "total_failed": total_failed,
        }

    def reset(self) -> None:
        """Reset state to defaults."""
        self._state = self._default_state()
        self.save()
        logger.info("Update state reset to defaults")
