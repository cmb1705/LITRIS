"""Tests for build_index classification gating."""

from datetime import datetime

import pytest

from src.analysis.classification_store import (
    ClassificationIndex,
    ClassificationRecord,
    ClassificationStore,
)


@pytest.fixture
def index_dir(tmp_path):
    return tmp_path / "index"


@pytest.fixture
def store(index_dir):
    return ClassificationStore(index_dir=index_dir)


def _make_record(doc_type="research_paper", extractable=True, word_count=5000):
    return ClassificationRecord(
        title="Test",
        item_type="journalArticle",
        document_type=doc_type,
        confidence=0.9,
        tier=1,
        reasons=[],
        extractable=extractable,
        word_count=word_count,
        page_count=10,
        section_markers=5,
        classified_at=datetime.now().isoformat(),
        extraction_intent="fast",
        intent_confidence=0.4,
        intent_reasons=["defaulted to fast"],
        intent_signals={"word_count": word_count},
        intent_classified_at=datetime.now().isoformat(),
        intent_source_tier="metadata_only",
        hybrid_profile_key="fast",
    )


class TestGatingFilter:
    def test_academic_only_filters_non_extractable(self, store, index_dir):
        index = ClassificationIndex()
        index.papers["p1"] = _make_record(extractable=True)
        index.papers["p2"] = _make_record(
            doc_type="non_academic", extractable=False, word_count=100,
        )
        store.save(index)

        loaded = store.load()
        ids = store.get_extractable_ids(loaded)
        assert ids == {"p1"}

    def test_index_all_keeps_everything(self, store, index_dir):
        index = ClassificationIndex()
        index.papers["p1"] = _make_record(extractable=True)
        index.papers["p2"] = _make_record(
            doc_type="non_academic", extractable=False,
        )
        store.save(index)

        loaded = store.load()
        # --index-all means we use all paper IDs, not just extractable
        all_ids = set(loaded.papers.keys())
        assert all_ids == {"p1", "p2"}

    def test_classify_only_produces_index_file(self, store, index_dir):
        """Verifies classification store creates the index file."""
        index = ClassificationIndex()
        index.papers["p1"] = _make_record()
        store.save(index)
        assert (index_dir / "classification_index.json").exists()

    def test_reclassify_replaces_all_records(self, store, index_dir):
        """--reclassify replaces existing records."""
        index = ClassificationIndex()
        index.papers["p1"] = _make_record()
        store.save(index)

        # Simulate reclassification
        new_index = ClassificationIndex()
        new_record = _make_record(doc_type="review_paper")
        new_index.papers["p1"] = new_record
        store.save(new_index)

        loaded = store.load()
        assert loaded.papers["p1"].document_type == "review_paper"


class TestSkipReport:
    def test_skip_summary_counts(self):
        """Skip summary correctly counts reasons."""
        index = ClassificationIndex()
        index.papers["p1"] = _make_record(extractable=True)
        index.papers["p2"] = _make_record(
            doc_type="non_academic", extractable=False,
        )
        index.papers["p3"] = _make_record(
            doc_type="research_paper", extractable=False, word_count=100,
        )

        skipped = {
            pid: rec for pid, rec in index.papers.items()
            if not rec.extractable
        }
        assert len(skipped) == 2

        non_academic = sum(
            1 for r in skipped.values() if r.document_type == "non_academic"
        )
        assert non_academic == 1


class TestInlineClassification:
    def test_inline_fallback_when_no_index(self, index_dir):
        """Store returns empty index when no file exists."""
        store = ClassificationStore(index_dir=index_dir)
        index = store.load()
        assert index.papers == {}

    def test_intent_fields_round_trip(self, store):
        index = ClassificationIndex()
        index.papers["p1"] = _make_record()
        store.save(index)

        loaded = store.load()
        record = loaded.papers["p1"]
        assert record.extraction_intent == "fast"
        assert record.hybrid_profile_key == "fast"
        assert record.intent_source_tier == "metadata_only"
