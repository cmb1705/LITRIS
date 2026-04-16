"""Persistent classification index for document type pre-filtering.

Stores per-paper classification results (document type, confidence,
extractability) to disk as JSON. Used by build_index.py to skip
non-academic content before expensive LLM extraction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.analysis.document_classifier import classify
from src.analysis.extraction_intent_classifier import ExtractionIntent
from src.utils.logging_config import get_logger

if TYPE_CHECKING:
    from src.zotero.models import PaperMetadata

logger = get_logger(__name__)

INDEX_FILENAME = "classification_index.json"
SCHEMA_VERSION = "1.1.0"


@dataclass
class ClassificationRecord:
    """Classification result for a single paper."""

    title: str
    item_type: str
    document_type: str
    confidence: float
    tier: int
    reasons: list[str]
    extractable: bool
    word_count: int | None
    page_count: int | None
    section_markers: int | None
    classified_at: str
    classification_error: str | None = None
    extraction_method: str | None = None
    extraction_intent: str | None = None
    intent_confidence: float = 0.0
    intent_reasons: list[str] = field(default_factory=list)
    intent_signals: dict[str, int | float | bool | str] = field(default_factory=dict)
    intent_classified_at: str | None = None
    intent_source_tier: str | None = None
    hybrid_profile_key: str | None = None


@dataclass
class ClassificationIndex:
    """In-memory representation of the classification index."""

    schema_version: str = SCHEMA_VERSION
    classified_at: str = ""
    stats: dict = field(default_factory=dict)
    papers: dict[str, ClassificationRecord] = field(default_factory=dict)


def _is_extractable(
    document_type: str,
    word_count: int | None,
    page_count: int | None,
    min_publication_words: int = 500,
    min_publication_pages: int = 2,
) -> bool:
    """Determine if a paper should be extracted.

    A paper is extractable when ALL of:
    - document_type is not NON_ACADEMIC
    - word_count >= min_publication_words (if known)
    - page_count >= min_publication_pages (if known)
    """
    if document_type.lower() == "non_academic":
        return False
    if word_count is not None and word_count < min_publication_words:
        return False
    if page_count is not None and page_count < min_publication_pages:
        return False
    return True


def _compute_stats(papers: dict[str, ClassificationRecord]) -> dict:
    """Recompute summary statistics from paper records."""
    by_type: dict[str, int] = {}
    by_intent: dict[str, int] = {}
    by_profile: dict[str, int] = {}
    extractable_count = 0
    non_extractable_count = 0

    for record in papers.values():
        by_type[record.document_type] = by_type.get(record.document_type, 0) + 1
        intent = record.extraction_intent or ExtractionIntent.FAST.value
        profile_key = record.hybrid_profile_key or "fast"
        by_intent[intent] = by_intent.get(intent, 0) + 1
        by_profile[profile_key] = by_profile.get(profile_key, 0) + 1
        if record.extractable:
            extractable_count += 1
        else:
            non_extractable_count += 1

    return {
        "total": len(papers),
        "by_type": by_type,
        "by_intent": by_intent,
        "by_profile": by_profile,
        "extractable_count": extractable_count,
        "non_extractable_count": non_extractable_count,
    }


class ClassificationStore:
    """Read/write/query the classification index on disk."""

    def __init__(
        self,
        index_dir: Path,
        min_publication_words: int = 500,
        min_publication_pages: int = 2,
    ):
        """Initialize store.

        Args:
            index_dir: Directory containing classification_index.json.
            min_publication_words: Minimum word count for extractability.
            min_publication_pages: Minimum page count for extractability.
        """
        self.index_dir = index_dir
        self.index_path = index_dir / INDEX_FILENAME
        self.min_publication_words = min_publication_words
        self.min_publication_pages = min_publication_pages

    def load(self) -> ClassificationIndex:
        """Load classification index from disk.

        Returns:
            ClassificationIndex (empty if file does not exist).

        Raises:
            json.JSONDecodeError: If the file is corrupt.
        """
        if not self.index_path.exists():
            return ClassificationIndex()

        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        index = ClassificationIndex(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            classified_at=data.get("classified_at", ""),
            stats=data.get("stats", {}),
        )
        for paper_id, rec_data in data.get("papers", {}).items():
            index.papers[paper_id] = ClassificationRecord(**rec_data)
        return index

    def save(self, index: ClassificationIndex) -> None:
        """Save classification index to disk with recomputed stats.

        Args:
            index: Classification index to save.
        """
        index.classified_at = datetime.now().isoformat()
        index.stats = _compute_stats(index.papers)

        data = {
            "schema_version": index.schema_version,
            "classified_at": index.classified_at,
            "stats": index.stats,
            "papers": {
                pid: asdict(rec) for pid, rec in index.papers.items()
            },
        }

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def classify_paper(
        self,
        paper: PaperMetadata,
        text: str | None = None,
        word_count: int | None = None,
        page_count: int | None = None,
        section_markers: int | None = None,
    ) -> ClassificationRecord:
        """Classify a single paper using the existing classifier.

        Args:
            paper: Paper metadata.
            text: Optional full text (triggers Tier 2 if needed).
            word_count: Precomputed word count.
            page_count: Precomputed page count.
            section_markers: Precomputed section marker count.

        Returns:
            ClassificationRecord with classification results.
        """
        doc_type, confidence = classify(
            paper=paper,
            text=text,
            word_count=word_count,
            page_count=page_count,
            section_marker_count=section_markers,
        )

        tier = 1 if text is None else 2
        doc_type_str = doc_type.value if hasattr(doc_type, "value") else str(doc_type)
        extractable = _is_extractable(
            doc_type_str,
            word_count,
            page_count,
            self.min_publication_words,
            self.min_publication_pages,
        )

        return ClassificationRecord(
            title=paper.title,
            item_type=paper.item_type,
            document_type=doc_type_str,
            confidence=confidence,
            tier=tier,
            reasons=[],
            extractable=extractable,
            word_count=word_count,
            page_count=page_count,
            section_markers=section_markers,
            classified_at=datetime.now().isoformat(),
            extraction_intent=ExtractionIntent.FAST.value,
            hybrid_profile_key="fast",
        )

    @staticmethod
    def get_extractable_ids(index: ClassificationIndex) -> set[str]:
        """Return IDs of all extractable papers.

        Args:
            index: Classification index.

        Returns:
            Set of paper IDs where extractable is True.
        """
        return {
            pid for pid, rec in index.papers.items() if rec.extractable
        }

    @staticmethod
    def get_paper(
        index: ClassificationIndex, paper_id: str
    ) -> ClassificationRecord | None:
        """Look up a single paper's classification.

        Args:
            index: Classification index.
            paper_id: Paper ID to look up.

        Returns:
            ClassificationRecord or None if not found.
        """
        return index.papers.get(paper_id)

    @staticmethod
    def get_stats(index: ClassificationIndex) -> dict:
        """Return summary statistics.

        Args:
            index: Classification index.

        Returns:
            Stats dictionary with totals and breakdowns.
        """
        return index.stats
