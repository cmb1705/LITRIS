"""Tests for incremental update functionality."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.indexing.update_state import UpdateRecord, UpdateState
from src.zotero.change_detector import ChangeDetector, ChangeSet, IndexState


class TestChangeSet:
    """Test ChangeSet data structure."""

    def test_has_changes_true_with_new(self):
        """Test has_changes returns True when new items exist."""
        changes = ChangeSet(
            new_items=["key1"],
            modified_items=[],
            deleted_items=[],
            unchanged_items=[],
        )
        assert changes.has_changes is True

    def test_has_changes_true_with_modified(self):
        """Test has_changes returns True when modified items exist."""
        changes = ChangeSet(
            new_items=[],
            modified_items=["paper1"],
            deleted_items=[],
            unchanged_items=[],
        )
        assert changes.has_changes is True

    def test_has_changes_true_with_deleted(self):
        """Test has_changes returns True when deleted items exist."""
        changes = ChangeSet(
            new_items=[],
            modified_items=[],
            deleted_items=["paper1"],
            unchanged_items=[],
        )
        assert changes.has_changes is True

    def test_has_changes_false_when_empty(self):
        """Test has_changes returns False when no changes."""
        changes = ChangeSet(
            new_items=[],
            modified_items=[],
            deleted_items=[],
            unchanged_items=["paper1"],
        )
        assert changes.has_changes is False

    def test_total_changes(self):
        """Test total_changes calculation."""
        changes = ChangeSet(
            new_items=["key1", "key2"],
            modified_items=["paper1"],
            deleted_items=["paper2", "paper3", "paper4"],
            unchanged_items=["paper5"],
        )
        assert changes.total_changes == 6

    def test_summary_with_changes(self):
        """Test summary string with changes."""
        changes = ChangeSet(
            new_items=["key1"],
            modified_items=["paper1", "paper2"],
            deleted_items=["paper3"],
            unchanged_items=[],
        )
        summary = changes.summary()
        assert "1 new" in summary
        assert "2 modified" in summary
        assert "1 deleted" in summary

    def test_summary_no_changes(self):
        """Test summary string with no changes."""
        changes = ChangeSet(
            new_items=[],
            modified_items=[],
            deleted_items=[],
            unchanged_items=["paper1"],
        )
        assert changes.summary() == "No changes detected"


class TestIndexState:
    """Test IndexState data structure."""

    def test_index_state_creation(self):
        """Test IndexState creation."""
        now = datetime.now()
        state = IndexState(
            paper_id="paper1",
            zotero_key="ABC12345",
            date_modified=now,
            date_indexed=now,
        )
        assert state.paper_id == "paper1"
        assert state.zotero_key == "ABC12345"
        assert state.date_modified == now


class TestChangeDetector:
    """Test ChangeDetector functionality."""

    @pytest.fixture
    def mock_zotero_db(self):
        """Create a mock Zotero database."""
        mock_db = MagicMock()
        mock_db.get_all_items_with_pdfs.return_value = [
            {
                "key": "ABC12345",
                "item_id": 1,
                "date_added": "2024-01-01 00:00:00",
                "date_modified": "2024-01-15 00:00:00",
                "item_type": "journalArticle",
                "attachment_id": 100,
                "attachment_key": "ATT12345",
                "attachment_path": "storage:file.pdf",
            },
            {
                "key": "DEF67890",
                "item_id": 2,
                "date_added": "2024-01-02 00:00:00",
                "date_modified": "2024-01-16 00:00:00",
                "item_type": "book",
                "attachment_id": 101,
                "attachment_key": "ATT67890",
                "attachment_path": "storage:book.pdf",
            },
        ]
        return mock_db

    @pytest.fixture
    def mock_index_dir(self, tmp_path):
        """Create a mock index directory with papers."""
        index_dir = tmp_path / "index"
        index_dir.mkdir()

        # Create papers.json with one existing paper
        papers_data = {
            "schema_version": "1.0",
            "papers": [
                {
                    "paper_id": "paper-abc",
                    "zotero_key": "ABC12345",
                    "title": "Test Paper",
                    "date_modified": "2024-01-10T00:00:00",
                    "date_added": "2024-01-01T00:00:00",
                }
            ],
        }
        with open(index_dir / "papers.json", "w") as f:
            json.dump(papers_data, f)

        return index_dir

    def test_detect_new_items(self, mock_zotero_db, mock_index_dir):
        """Test detection of new items."""
        detector = ChangeDetector(
            zotero_db=mock_zotero_db,
            index_dir=mock_index_dir,
        )

        changes = detector.detect_changes()

        # DEF67890 is new (not in index)
        assert "DEF67890" in changes.new_items
        assert len(changes.new_items) == 1

    def test_detect_modified_items(self, mock_zotero_db, mock_index_dir):
        """Test detection of modified items."""
        detector = ChangeDetector(
            zotero_db=mock_zotero_db,
            index_dir=mock_index_dir,
        )

        changes = detector.detect_changes()

        # ABC12345 was modified (2024-01-15 > 2024-01-10 in index)
        assert "paper-abc" in changes.modified_items

    def test_detect_deleted_items(self, mock_zotero_db, mock_index_dir):
        """Test detection of deleted items."""
        # Add a paper to the index that's not in Zotero
        papers_file = mock_index_dir / "papers.json"
        with open(papers_file) as f:
            data = json.load(f)

        data["papers"].append({
            "paper_id": "paper-deleted",
            "zotero_key": "DELETED123",
            "title": "Deleted Paper",
            "date_modified": "2024-01-05T00:00:00",
        })

        with open(papers_file, "w") as f:
            json.dump(data, f)

        detector = ChangeDetector(
            zotero_db=mock_zotero_db,
            index_dir=mock_index_dir,
        )

        changes = detector.detect_changes()

        assert "paper-deleted" in changes.deleted_items

    def test_detect_unchanged_items(self, mock_zotero_db, mock_index_dir):
        """Test detection of unchanged items."""
        # Update the index to have same modification date
        papers_file = mock_index_dir / "papers.json"
        with open(papers_file) as f:
            data = json.load(f)

        # Set date to same as Zotero
        data["papers"][0]["date_modified"] = "2024-01-15T00:00:00"

        with open(papers_file, "w") as f:
            json.dump(data, f)

        detector = ChangeDetector(
            zotero_db=mock_zotero_db,
            index_dir=mock_index_dir,
        )

        changes = detector.detect_changes()

        assert "paper-abc" in changes.unchanged_items

    def test_get_stats(self, mock_zotero_db, mock_index_dir):
        """Test stats retrieval."""
        detector = ChangeDetector(
            zotero_db=mock_zotero_db,
            index_dir=mock_index_dir,
        )

        stats = detector.get_stats()

        assert stats["indexed_papers"] == 1
        assert stats["zotero_papers"] == 2
        assert stats["has_changes"] is True
        assert "summary" in stats

    def test_empty_index(self, mock_zotero_db, tmp_path):
        """Test detection with empty index."""
        index_dir = tmp_path / "empty_index"
        index_dir.mkdir()

        detector = ChangeDetector(
            zotero_db=mock_zotero_db,
            index_dir=index_dir,
        )

        changes = detector.detect_changes()

        # All items should be new
        assert len(changes.new_items) == 2
        assert len(changes.modified_items) == 0
        assert len(changes.deleted_items) == 0


class TestUpdateState:
    """Test UpdateState functionality."""

    def test_initialization(self, tmp_path):
        """Test state initialization."""
        state = UpdateState(index_dir=tmp_path)

        assert state.get_last_update() is None
        assert state.get_last_full_build() is None
        assert state.needs_full_rebuild() is True

    def test_record_full_build(self, tmp_path):
        """Test recording a full build."""
        state = UpdateState(index_dir=tmp_path)

        state.record_full_build(
            papers_count=100,
            failed_count=5,
            duration_seconds=300.0,
        )

        assert state.get_last_full_build() is not None
        assert state.get_last_update() is not None
        assert state.needs_full_rebuild() is False

        # Check state file was created
        state_file = tmp_path / "update_state.json"
        assert state_file.exists()

    def test_record_incremental_update(self, tmp_path):
        """Test recording an incremental update."""
        state = UpdateState(index_dir=tmp_path)

        # First do a full build
        state.record_full_build(papers_count=100)

        # Then an incremental update
        state.record_incremental_update(
            new_added=5,
            modified_updated=3,
            deleted_removed=2,
            failed=1,
            duration_seconds=60.0,
        )

        stats = state.get_stats()
        assert stats["update_count"] == 2

    def test_update_history(self, tmp_path):
        """Test update history retrieval."""
        state = UpdateState(index_dir=tmp_path)

        state.record_full_build(papers_count=100)
        state.record_incremental_update(new_added=5)
        state.record_incremental_update(new_added=3)

        history = state.get_update_history(limit=10)

        assert len(history) == 3
        # Most recent first
        assert history[0].new_added == 3
        assert history[1].new_added == 5
        assert history[2].update_type == "full"

    def test_paper_version_tracking(self, tmp_path):
        """Test paper version tracking."""
        state = UpdateState(index_dir=tmp_path)

        state.set_paper_version(
            paper_id="paper1",
            zotero_key="ABC123",
            date_modified="2024-01-15T00:00:00",
        )

        version = state.get_paper_version("paper1")

        assert version is not None
        assert version["zotero_key"] == "ABC123"
        assert version["date_modified"] == "2024-01-15T00:00:00"
        assert "indexed_at" in version

    def test_remove_paper_version(self, tmp_path):
        """Test removing paper version tracking."""
        state = UpdateState(index_dir=tmp_path)

        state.set_paper_version("paper1", "ABC123", "2024-01-15T00:00:00")
        state.remove_paper_version("paper1")

        assert state.get_paper_version("paper1") is None

    def test_persistence(self, tmp_path):
        """Test state persistence across instances."""
        state1 = UpdateState(index_dir=tmp_path)
        state1.record_full_build(papers_count=50)
        state1.set_paper_version("paper1", "ABC123", "2024-01-15T00:00:00")

        # Create new instance - should load saved state
        state2 = UpdateState(index_dir=tmp_path)

        assert state2.get_last_full_build() is not None
        assert state2.get_paper_version("paper1") is not None
        assert state2.needs_full_rebuild() is False

    def test_reset(self, tmp_path):
        """Test state reset."""
        state = UpdateState(index_dir=tmp_path)
        state.record_full_build(papers_count=100)

        state.reset()

        assert state.get_last_full_build() is None
        assert state.needs_full_rebuild() is True

    def test_get_stats(self, tmp_path):
        """Test stats retrieval."""
        state = UpdateState(index_dir=tmp_path)

        state.record_full_build(papers_count=100)
        state.record_incremental_update(new_added=10, modified_updated=5, deleted_removed=2)

        stats = state.get_stats()

        assert stats["update_count"] == 2
        assert stats["total_added"] == 110  # 100 + 10
        assert stats["total_modified"] == 5
        assert stats["total_deleted"] == 2


class TestUpdateRecord:
    """Test UpdateRecord data structure."""

    def test_update_record_creation(self):
        """Test UpdateRecord creation."""
        record = UpdateRecord(
            timestamp="2024-01-15T10:00:00",
            update_type="incremental",
            new_added=5,
            modified_updated=3,
            deleted_removed=2,
            failed=1,
            duration_seconds=60.0,
        )

        assert record.update_type == "incremental"
        assert record.new_added == 5
        assert record.duration_seconds == 60.0

    def test_update_record_to_dict(self):
        """Test UpdateRecord conversion to dict."""
        record = UpdateRecord(
            timestamp="2024-01-15T10:00:00",
            update_type="full",
            new_added=100,
            modified_updated=0,
            deleted_removed=0,
            failed=5,
            duration_seconds=300.0,
        )

        d = record._asdict()

        assert d["update_type"] == "full"
        assert d["new_added"] == 100
        assert d["failed"] == 5
