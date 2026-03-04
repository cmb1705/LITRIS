#!/usr/bin/env python
"""Retroactively classify existing extractions by document type.

Classifies all existing papers and extractions using metadata (Tier 1) and
cached text statistics (Tier 2). Writes document_type and type_confidence
into extractions.json without requiring LLM re-extraction.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.document_classifier import classify_metadata, classify_text
from src.analysis.document_types import DocumentType
from src.extraction.text_cleaner import TextCleaner
from src.utils.file_utils import safe_read_json, safe_write_json
from src.utils.logging_config import setup_logging
from src.zotero.models import PaperMetadata


def build_paper_metadata(paper_data: dict) -> PaperMetadata:
    """Build a PaperMetadata from papers.json entry for classification.

    Args:
        paper_data: Dictionary from papers.json.

    Returns:
        PaperMetadata instance with fields populated for classification.
    """
    return PaperMetadata(
        paper_id=paper_data.get("paper_id", ""),
        zotero_key=paper_data.get("zotero_key", ""),
        title=paper_data.get("title", ""),
        authors=[],
        publication_year=paper_data.get("publication_year"),
        item_type=paper_data.get("item_type", "document"),
        doi=paper_data.get("doi"),
        isbn=paper_data.get("isbn"),
        issn=paper_data.get("issn"),
        journal=paper_data.get("journal"),
        abstract=paper_data.get("abstract"),
        volume=paper_data.get("volume"),
        issue=paper_data.get("issue"),
        pages=paper_data.get("pages"),
    )


def classify_paper(
    paper_data: dict,
    cached_text_dir: Path | None = None,
    text_cleaner: TextCleaner | None = None,
) -> tuple[str, float]:
    """Classify a single paper by document type.

    Args:
        paper_data: Paper data from papers.json.
        cached_text_dir: Directory containing cached PDF text files.
        text_cleaner: TextCleaner for computing text statistics.

    Returns:
        Tuple of (document_type string, confidence).
    """
    paper = build_paper_metadata(paper_data)

    # Tier 1: metadata-only classification
    meta_type, meta_confidence = classify_metadata(paper)

    if meta_type is not None and meta_confidence >= 0.8:
        return meta_type.value, meta_confidence

    # Tier 2: try to use cached text if available
    if cached_text_dir and text_cleaner:
        paper_id = paper_data.get("paper_id", "")
        text = _load_cached_text(paper_id, cached_text_dir)
        if text:
            stats = text_cleaner.get_stats(text)
            section_hits = text_cleaner.count_section_markers(text)
            doc_type, confidence = classify_text(
                text=text,
                metadata_type=meta_type,
                word_count=stats.word_count,
                page_count=stats.page_count,
                section_marker_count=section_hits,
            )
            return doc_type.value, confidence

    # Return best Tier 1 result
    if meta_type is not None:
        return meta_type.value, meta_confidence
    return DocumentType.RESEARCH_PAPER.value, 0.3


def _load_cached_text(paper_id: str, cache_dir: Path) -> str | None:
    """Load cached PDF text for a paper if available.

    Args:
        paper_id: Paper identifier.
        cache_dir: Cache directory (containing pdf_text/ subfolder).

    Returns:
        Cached text content or None.
    """
    text_dir = cache_dir / "pdf_text"
    if not text_dir.exists():
        return None

    # Cache files may use paper_id as filename
    for ext in (".txt", ".json"):
        cache_file = text_dir / f"{paper_id}{ext}"
        if cache_file.exists():
            try:
                if ext == ".json":
                    import json
                    data = json.loads(cache_file.read_text(encoding="utf-8"))
                    return data.get("text", "")
                return cache_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
    return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Classify existing extractions by document type"
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=project_root / "data" / "index",
        help="Path to index directory",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=project_root / "data" / "cache",
        help="Path to cache directory (for cached PDF text)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report classifications without modifying files",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    logger = setup_logging(level="DEBUG" if args.verbose else "INFO")

    # Load papers
    papers_file = args.index_dir / "papers.json"
    if not papers_file.exists():
        logger.error(f"Papers file not found: {papers_file}")
        return 1

    papers_data = safe_read_json(papers_file, default={})
    if isinstance(papers_data, dict) and "papers" in papers_data:
        papers_list = papers_data["papers"]
    else:
        logger.error("Unexpected papers.json format")
        return 1

    papers_by_id = {p["paper_id"]: p for p in papers_list if "paper_id" in p}
    logger.info(f"Loaded {len(papers_by_id)} papers")

    # Load extractions
    extractions_file = args.index_dir / "extractions.json"
    extractions_data = {}
    raw_extractions: dict = {}
    extractions_nested = False
    if extractions_file.exists():
        raw_extractions = safe_read_json(extractions_file, default={})
        if isinstance(raw_extractions, dict) and "extractions" in raw_extractions:
            extractions_data = raw_extractions["extractions"]
            extractions_nested = True
        else:
            extractions_data = raw_extractions
    logger.info(f"Loaded {len(extractions_data)} extractions")

    # Set up text cleaner for Tier 2
    text_cleaner = TextCleaner()
    cache_dir = args.cache_dir if args.cache_dir.exists() else None

    # Classify all papers
    type_counter: Counter[str] = Counter()
    confidence_bins = {"high": 0, "medium": 0, "low": 0}
    classified_count = 0
    already_classified = 0

    for paper_id, paper_data in papers_by_id.items():
        doc_type, confidence = classify_paper(
            paper_data, cache_dir, text_cleaner
        )

        type_counter[doc_type] += 1
        if confidence >= 0.8:
            confidence_bins["high"] += 1
        elif confidence >= 0.6:
            confidence_bins["medium"] += 1
        else:
            confidence_bins["low"] += 1

        # Update extraction data
        if paper_id in extractions_data and not args.dry_run:
            ext = extractions_data[paper_id]
            # Check if nested structure
            if "extraction" in ext and isinstance(ext["extraction"], dict):
                ext["extraction"]["document_type"] = doc_type
                ext["document_type"] = doc_type
                ext["type_confidence"] = confidence
            else:
                ext["document_type"] = doc_type

            classified_count += 1

    # Report
    print("\n" + "=" * 60)
    print("DOCUMENT TYPE CLASSIFICATION REPORT")
    print("=" * 60)

    print(f"\nTotal papers: {len(papers_by_id)}")
    print(f"Classified extractions: {classified_count}")
    print(f"Already classified: {already_classified}")

    print("\nType Distribution:")
    for doc_type, count in type_counter.most_common():
        pct = count / len(papers_by_id) * 100 if papers_by_id else 0
        print(f"  {doc_type:25s} {count:5d} ({pct:5.1f}%)")

    print("\nConfidence Distribution:")
    for level, count in confidence_bins.items():
        pct = count / len(papers_by_id) * 100 if papers_by_id else 0
        print(f"  {level:10s} (>={'0.8' if level == 'high' else '0.6' if level == 'medium' else '0.0'}): {count:5d} ({pct:5.1f}%)")

    print(f"\nItems needing review (confidence < 0.6): {confidence_bins['low']}")

    # Save updated extractions
    if not args.dry_run and extractions_data:
        if extractions_nested:
            raw_extractions["extractions"] = extractions_data
            safe_write_json(extractions_file, raw_extractions)
        else:
            safe_write_json(extractions_file, extractions_data)
        print(f"\nUpdated extractions saved to: {extractions_file}")
    elif args.dry_run:
        print("\n[DRY RUN] No files modified.")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
