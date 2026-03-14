"""Tests for ClassificationStore -- persistent classification index."""

import json
from datetime import datetime
from unittest.mock import patch

import pytest

from src.analysis.classification_store import (
    ClassificationIndex,
    ClassificationRecord,
    ClassificationStore,
)


@pytest.fixture
def store(tmp_path):
    """ClassificationStore pointed at a temp directory."""
    return ClassificationStore(index_dir=tmp_path)


@pytest.fixture
def sample_record():
    """A sample ClassificationRecord."""
    return ClassificationRecord(
        title="Test Paper",
        item_type="journalArticle",
        document_type="research_paper",
        confidence=0.90,
        tier=1,
        reasons=["item_type=journalArticle"],
        extractable=True,
        word_count=5000,
        page_count=10,
        section_markers=5,
        classified_at=datetime.now().isoformat(),
    )


class TestClassificationRecord:
    def test_extractable_true_for_academic(self, sample_record):
        assert sample_record.extractable is True

    def test_extractable_false_for_non_academic(self):
        rec = ClassificationRecord(
            title="Course Slides",
            item_type="presentation",
            document_type="non_academic",
            confidence=0.95,
            tier=1,
            reasons=["item_type=presentation"],
            extractable=False,
            word_count=200,
            page_count=30,
            section_markers=0,
            classified_at=datetime.now().isoformat(),
        )
        assert rec.extractable is False


class TestClassificationStoreIO:
    def test_load_returns_empty_when_no_file(self, store):
        index = store.load()
        assert index.papers == {}
        assert index.schema_version == "1.0.0"

    def test_save_and_load_roundtrip(self, store, sample_record):
        index = ClassificationIndex()
        index.papers["paper-001"] = sample_record
        store.save(index)

        loaded = store.load()
        assert "paper-001" in loaded.papers
        assert loaded.papers["paper-001"].document_type == "research_paper"
        assert loaded.stats["total"] == 1

    def test_save_recomputes_stats(self, store, sample_record):
        index = ClassificationIndex()
        index.papers["p1"] = sample_record
        non_academic = ClassificationRecord(
            title="Slides",
            item_type="presentation",
            document_type="non_academic",
            confidence=0.95,
            tier=1,
            reasons=[],
            extractable=False,
            word_count=100,
            page_count=1,
            section_markers=0,
            classified_at=datetime.now().isoformat(),
        )
        index.papers["p2"] = non_academic
        store.save(index)

        loaded = store.load()
        assert loaded.stats["total"] == 2
        assert loaded.stats["extractable_count"] == 1
        assert loaded.stats["non_extractable_count"] == 1
        assert loaded.stats["by_type"]["research_paper"] == 1
        assert loaded.stats["by_type"]["non_academic"] == 1

    def test_load_corrupt_file_raises(self, store):
        path = store.index_path
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            store.load()

    def test_save_creates_parent_dir(self, tmp_path):
        nested = tmp_path / "a" / "b"
        s = ClassificationStore(index_dir=nested)
        s.save(ClassificationIndex())
        assert s.index_path.exists()


class TestClassificationStoreQuery:
    def test_get_extractable_ids(self, store, sample_record):
        index = ClassificationIndex()
        index.papers["p1"] = sample_record
        non_ext = ClassificationRecord(
            title="X",
            item_type="presentation",
            document_type="non_academic",
            confidence=0.95,
            tier=1,
            reasons=[],
            extractable=False,
            word_count=50,
            page_count=1,
            section_markers=0,
            classified_at=datetime.now().isoformat(),
        )
        index.papers["p2"] = non_ext
        store.save(index)

        loaded = store.load()
        ids = store.get_extractable_ids(loaded)
        assert ids == {"p1"}

    def test_get_paper(self, store, sample_record):
        index = ClassificationIndex()
        index.papers["p1"] = sample_record
        assert store.get_paper(index, "p1") == sample_record
        assert store.get_paper(index, "missing") is None

    def test_get_stats(self, store, sample_record):
        index = ClassificationIndex()
        index.papers["p1"] = sample_record
        store.save(index)

        loaded = store.load()
        stats = store.get_stats(loaded)
        assert stats["total"] == 1
        assert stats["extractable_count"] == 1


class TestClassificationStoreClassify:
    @patch("src.analysis.classification_store.classify")
    def test_classify_paper_uses_existing_classifier(self, mock_classify, store):
        from src.analysis.document_types import DocumentType
        from src.zotero.models import Author, PaperMetadata

        mock_classify.return_value = (DocumentType.RESEARCH_PAPER, 0.85)

        paper = PaperMetadata(
            zotero_key="ZK001",
            zotero_item_id=1,
            title="Test Paper",
            item_type="journalArticle",
            date_added=datetime(2026, 1, 1),
            date_modified=datetime(2026, 1, 1),
            authors=[Author(first_name="A", last_name="B")],
        )

        record = store.classify_paper(paper, text=None)
        assert record.document_type == "research_paper"
        assert record.confidence == 0.85
        assert record.tier == 1
        mock_classify.assert_called_once()

    @patch("src.analysis.classification_store.classify")
    def test_classify_paper_with_text_uses_tier2(self, mock_classify, store):
        from src.analysis.document_types import DocumentType
        from src.zotero.models import Author, PaperMetadata

        mock_classify.return_value = (DocumentType.REVIEW_PAPER, 0.92)

        paper = PaperMetadata(
            zotero_key="ZK002",
            zotero_item_id=2,
            title="Review Paper",
            item_type="journalArticle",
            date_added=datetime(2026, 1, 1),
            date_modified=datetime(2026, 1, 1),
            authors=[Author(first_name="C", last_name="D")],
        )

        record = store.classify_paper(
            paper, text="Full text here", word_count=5000, page_count=20,
        )
        assert record.document_type == "review_paper"
        assert record.tier == 2
        assert record.extractable is True

    @patch("src.analysis.classification_store.classify")
    def test_classify_paper_non_academic_not_extractable(self, mock_classify, store):
        from src.analysis.document_types import DocumentType
        from src.zotero.models import Author, PaperMetadata

        mock_classify.return_value = (DocumentType.NON_ACADEMIC, 0.95)

        paper = PaperMetadata(
            zotero_key="ZK003",
            zotero_item_id=3,
            title="Course Slides",
            item_type="presentation",
            date_added=datetime(2026, 1, 1),
            date_modified=datetime(2026, 1, 1),
            authors=[Author(first_name="E", last_name="F")],
        )

        record = store.classify_paper(paper, text=None)
        assert record.document_type == "non_academic"
        assert record.extractable is False
