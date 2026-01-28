"""Checkpoint system for resumable pipeline operations."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from src.utils.file_utils import safe_read_json, safe_write_json
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ItemStatus(str, Enum):
    """Status of a processed item."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class FailedItem:
    """Record of a failed item."""

    item_id: str
    error_message: str
    error_type: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    retry_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FailedItem":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CheckpointState:
    """State of a checkpoint."""

    checkpoint_id: str
    created_at: str
    last_updated: str
    total_items: int
    processed_count: int
    success_count: int
    failed_count: int
    skipped_count: int
    current_item: str | None = None
    processed_ids: list[str] = field(default_factory=list)
    failed_items: list[FailedItem] = field(default_factory=list)
    skipped_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        data["failed_items"] = [f if isinstance(f, dict) else f.to_dict() for f in self.failed_items]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointState":
        """Create from dictionary."""
        if "failed_items" in data:
            data["failed_items"] = [
                FailedItem.from_dict(f) if isinstance(f, dict) else f
                for f in data["failed_items"]
            ]
        return cls(**data)


class CheckpointManager:
    """Manage checkpoint state for resumable operations."""

    CHECKPOINT_FILE = "checkpoint.json"

    def __init__(self, checkpoint_dir: Path, checkpoint_id: str = "default"):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files.
            checkpoint_id: Unique identifier for this checkpoint set.
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_id = checkpoint_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._state: CheckpointState | None = None

    @property
    def checkpoint_file(self) -> Path:
        """Get checkpoint file path."""
        return self.checkpoint_dir / self.CHECKPOINT_FILE

    def initialize(self, total_items: int, metadata: dict | None = None) -> CheckpointState:
        """Initialize a new checkpoint.

        Args:
            total_items: Total number of items to process.
            metadata: Optional metadata to store.

        Returns:
            New checkpoint state.
        """
        now = datetime.now().isoformat()
        self._state = CheckpointState(
            checkpoint_id=self.checkpoint_id,
            created_at=now,
            last_updated=now,
            total_items=total_items,
            processed_count=0,
            success_count=0,
            failed_count=0,
            skipped_count=0,
            metadata=metadata or {},
        )
        self.save()
        logger.info(f"Initialized checkpoint for {total_items} items")
        return self._state

    def load(self) -> CheckpointState | None:
        """Load existing checkpoint state.

        Returns:
            Checkpoint state or None if not found.
        """
        if not self.checkpoint_file.exists():
            return None

        try:
            data = safe_read_json(self.checkpoint_file, default=None)
            if data:
                self._state = CheckpointState.from_dict(data)
                logger.info(
                    f"Loaded checkpoint: {self._state.processed_count}/{self._state.total_items} processed"
                )
                return self._state
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

        return None

    def save(self) -> None:
        """Save current checkpoint state."""
        if not self._state:
            return

        self._state.last_updated = datetime.now().isoformat()
        safe_write_json(self.checkpoint_file, self._state.to_dict())

    @property
    def state(self) -> CheckpointState | None:
        """Get current state."""
        return self._state

    def start_item(self, item_id: str) -> None:
        """Mark an item as in progress.

        Args:
            item_id: Item identifier.
        """
        if self._state:
            self._state.current_item = item_id

    def complete_item(self, item_id: str, success: bool, error: Exception | None = None) -> None:
        """Mark an item as complete.

        Args:
            item_id: Item identifier.
            success: Whether processing succeeded.
            error: Optional error if failed.
        """
        if not self._state:
            return

        self._state.processed_count += 1
        self._state.current_item = None

        if success:
            self._state.success_count += 1
            if item_id not in self._state.processed_ids:
                self._state.processed_ids.append(item_id)
        else:
            self._state.failed_count += 1
            # Check if already failed before (retry)
            existing = next(
                (f for f in self._state.failed_items if f.item_id == item_id), None
            )
            if existing:
                existing.retry_count += 1
                existing.timestamp = datetime.now().isoformat()
                if error:
                    existing.error_message = str(error)
                    existing.error_type = type(error).__name__
            else:
                failed_item = FailedItem(
                    item_id=item_id,
                    error_message=str(error) if error else "Unknown error",
                    error_type=type(error).__name__ if error else "Unknown",
                )
                self._state.failed_items.append(failed_item)

    def skip_item(self, item_id: str, reason: str | None = None) -> None:
        """Mark an item as skipped.

        Args:
            item_id: Item identifier.
            reason: Optional reason for skipping.
        """
        if not self._state:
            return

        self._state.processed_count += 1
        self._state.skipped_count += 1
        if item_id not in self._state.skipped_ids:
            self._state.skipped_ids.append(item_id)

        if reason:
            logger.debug(f"Skipped {item_id}: {reason}")

    def is_processed(self, item_id: str) -> bool:
        """Check if an item has been processed.

        Args:
            item_id: Item identifier.

        Returns:
            True if item was processed successfully.
        """
        if not self._state:
            return False
        return item_id in self._state.processed_ids

    def is_failed(self, item_id: str) -> bool:
        """Check if an item has failed.

        Args:
            item_id: Item identifier.

        Returns:
            True if item failed.
        """
        if not self._state:
            return False
        return any(f.item_id == item_id for f in self._state.failed_items)

    def is_skipped(self, item_id: str) -> bool:
        """Check if an item was skipped.

        Args:
            item_id: Item identifier.

        Returns:
            True if item was skipped.
        """
        if not self._state:
            return False
        return item_id in self._state.skipped_ids

    def get_failed_ids(self) -> list[str]:
        """Get list of failed item IDs.

        Returns:
            List of failed item IDs.
        """
        if not self._state:
            return []
        return [f.item_id for f in self._state.failed_items]

    def get_pending_ids(self, all_ids: list[str]) -> list[str]:
        """Get list of IDs that haven't been processed.

        Args:
            all_ids: All item IDs.

        Returns:
            List of pending item IDs.
        """
        if not self._state:
            return all_ids

        processed = set(self._state.processed_ids)
        failed = set(self.get_failed_ids())
        skipped = set(self._state.skipped_ids)

        return [
            item_id
            for item_id in all_ids
            if item_id not in processed and item_id not in failed and item_id not in skipped
        ]

    def clear_failed(self, item_ids: list[str] | None = None) -> int:
        """Clear failed items to allow retry.

        Args:
            item_ids: Specific IDs to clear, or None for all.

        Returns:
            Number of items cleared.
        """
        if not self._state:
            return 0

        if item_ids is None:
            count = len(self._state.failed_items)
            self._state.failed_items = []
            self._state.failed_count = 0
        else:
            id_set = set(item_ids)
            original_count = len(self._state.failed_items)
            self._state.failed_items = [
                f for f in self._state.failed_items if f.item_id not in id_set
            ]
            count = original_count - len(self._state.failed_items)
            self._state.failed_count = len(self._state.failed_items)

        self.save()
        logger.info(f"Cleared {count} failed items")
        return count

    def get_progress(self) -> dict:
        """Get progress summary.

        Returns:
            Dictionary with progress info.
        """
        if not self._state:
            return {"status": "not_started"}

        return {
            "status": "in_progress" if self._state.current_item else "paused",
            "total": self._state.total_items,
            "processed": self._state.processed_count,
            "success": self._state.success_count,
            "failed": self._state.failed_count,
            "skipped": self._state.skipped_count,
            "remaining": self._state.total_items - self._state.processed_count,
            "progress_pct": (
                self._state.processed_count / self._state.total_items * 100
                if self._state.total_items > 0
                else 0
            ),
            "current_item": self._state.current_item,
        }

    def reset(self) -> None:
        """Reset checkpoint to start fresh."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        self._state = None
        logger.info("Checkpoint reset")
