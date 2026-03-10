"""Progress tracker for CLI extraction with pause/resume capability."""

import json
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class FailedPaper(NamedTuple):
    """Record of a failed paper extraction."""

    paper_id: str
    error: str
    timestamp: str


class ProgressState(NamedTuple):
    """Current progress state."""

    started_at: str
    last_updated: str
    total_papers: int
    completed: list[str]
    failed: list[FailedPaper]
    current_session_start: str | None
    requests_this_session: int


class ProgressTracker:
    """Track CLI extraction progress for pause/resume capability.

    Saves progress to JSON file, allowing extraction to be paused
    (e.g., due to rate limits) and resumed later without reprocessing
    already-extracted papers.
    """

    DEFAULT_FILE = "cli_progress.json"

    def __init__(self, cache_dir: Path):
        """Initialize progress tracker.

        Args:
            cache_dir: Directory for progress file.
        """
        self.cache_dir = cache_dir
        self.progress_file = cache_dir / self.DEFAULT_FILE
        self._state: dict | None = None

        # Ensure directory exists
        cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> ProgressState | None:
        """Load existing progress or return None.

        Returns:
            ProgressState if file exists, else None.
        """
        if not self.progress_file.exists():
            return None

        try:
            with open(self.progress_file, encoding="utf-8") as f:
                data = json.load(f)

            self._state = data

            # Parse failed papers
            failed = [
                FailedPaper(
                    paper_id=fp["paper_id"],
                    error=fp["error"],
                    timestamp=fp["timestamp"],
                )
                for fp in data.get("failed", [])
            ]

            current_session = data.get("current_session", {})

            return ProgressState(
                started_at=data.get("started_at", ""),
                last_updated=data.get("last_updated", ""),
                total_papers=data.get("total_papers", 0),
                completed=data.get("completed", []),
                failed=failed,
                current_session_start=current_session.get("started_at"),
                requests_this_session=current_session.get("requests_this_session", 0),
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load progress file: {e}")
            # Backup corrupted file
            backup_path = self.progress_file.with_suffix(".json.bak")
            self.progress_file.rename(backup_path)
            logger.info(f"Backed up corrupted progress file to {backup_path}")
            return None

    def initialize(self, total_papers: int) -> None:
        """Initialize new progress tracking.

        Args:
            total_papers: Total number of papers to process.
        """
        now = datetime.now().isoformat()
        self._state = {
            "started_at": now,
            "last_updated": now,
            "total_papers": total_papers,
            "completed": [],
            "failed": [],
            "current_session": {
                "started_at": now,
                "requests_this_session": 0,
            },
        }
        self.save()
        logger.info(f"Initialized progress tracking for {total_papers} papers")

    def start_session(self) -> None:
        """Start a new extraction session."""
        if not self._state:
            raise RuntimeError("Progress not initialized. Call initialize() first.")

        self._state["current_session"] = {
            "started_at": datetime.now().isoformat(),
            "requests_this_session": 0,
        }
        self.save()

    def mark_completed(self, paper_id: str) -> None:
        """Mark a paper as successfully extracted.

        Args:
            paper_id: ID of completed paper.
        """
        if not self._state:
            raise RuntimeError("Progress not initialized")

        if paper_id not in self._state["completed"]:
            self._state["completed"].append(paper_id)

        # Remove from failed if previously failed
        self._state["failed"] = [
            fp for fp in self._state["failed"]
            if fp.get("paper_id") != paper_id
        ]

        # Increment session counter
        if "current_session" in self._state:
            self._state["current_session"]["requests_this_session"] += 1

        self._state["last_updated"] = datetime.now().isoformat()
        self.save()

    def mark_failed(self, paper_id: str, error: str) -> None:
        """Mark a paper as failed.

        Args:
            paper_id: ID of failed paper.
            error: Error message.
        """
        if not self._state:
            raise RuntimeError("Progress not initialized")

        # Remove existing failure record if any
        self._state["failed"] = [
            fp for fp in self._state["failed"]
            if fp.get("paper_id") != paper_id
        ]

        # Add new failure record
        self._state["failed"].append({
            "paper_id": paper_id,
            "error": str(error),
            "timestamp": datetime.now().isoformat(),
        })

        self._state["last_updated"] = datetime.now().isoformat()
        self.save()

    def get_pending_papers(self, all_paper_ids: list[str]) -> list[str]:
        """Get papers that haven't been processed yet.

        Args:
            all_paper_ids: List of all paper IDs.

        Returns:
            List of paper IDs not yet completed or failed.
        """
        if not self._state:
            return all_paper_ids

        completed = set(self._state.get("completed", []))
        failed = {fp.get("paper_id") for fp in self._state.get("failed", [])}
        processed = completed | failed

        return [pid for pid in all_paper_ids if pid not in processed]

    def get_failed_papers(self) -> list[FailedPaper]:
        """Get list of failed papers.

        Returns:
            List of FailedPaper records.
        """
        if not self._state:
            return []

        return [
            FailedPaper(
                paper_id=fp["paper_id"],
                error=fp["error"],
                timestamp=fp["timestamp"],
            )
            for fp in self._state.get("failed", [])
        ]

    def get_completed_papers(self) -> list[str]:
        """Get list of completed paper IDs.

        Returns:
            List of completed paper IDs.
        """
        if not self._state:
            return []
        return list(self._state.get("completed", []))

    def get_session_request_count(self) -> int:
        """Get number of requests in current session.

        Returns:
            Request count.
        """
        if not self._state or "current_session" not in self._state:
            return 0
        return self._state["current_session"].get("requests_this_session", 0)

    def save(self) -> None:
        """Save current progress to file."""
        if not self._state:
            return

        try:
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    def reset(self) -> None:
        """Reset progress (delete progress file)."""
        if self.progress_file.exists():
            self.progress_file.unlink()
            logger.info("Progress reset")
        self._state = None

    def get_summary(self) -> dict:
        """Get progress summary.

        Returns:
            Dict with progress statistics.
        """
        if not self._state:
            return {
                "initialized": False,
                "completed": 0,
                "failed": 0,
                "pending": 0,
                "total": 0,
                "progress_percent": 0.0,
            }

        total = self._state.get("total_papers", 0)
        completed = len(self._state.get("completed", []))
        failed = len(self._state.get("failed", []))
        pending = total - completed - failed

        return {
            "initialized": True,
            "started_at": self._state.get("started_at"),
            "last_updated": self._state.get("last_updated"),
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "total": total,
            "progress_percent": (completed / total * 100) if total > 0 else 0.0,
            "session_requests": self.get_session_request_count(),
        }
