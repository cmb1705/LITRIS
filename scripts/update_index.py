#!/usr/bin/env python
"""Incremental update of the literature review index."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm

from src.analysis.schemas import PaperExtraction
from src.analysis.section_extractor import SectionExtractor
from src.config import Config
from src.indexing.embeddings import EmbeddingGenerator
from src.indexing.structured_store import StructuredStore
from src.indexing.vector_store import VectorStore
from src.utils.logging_config import setup_logging
from src.zotero.change_detector import ChangeDetector, ChangeSet
from src.zotero.database import ZoteroDatabase


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Incrementally update the literature review index"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only detect changes, don't apply them",
    )
    parser.add_argument(
        "--new-only",
        action="store_true",
        help="Only process new items, skip modified and deleted",
    )
    parser.add_argument(
        "--delete-only",
        action="store_true",
        help="Only process deleted items",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip LLM extraction (only update from existing data)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation",
    )
    parser.add_argument(
        "--mode",
        choices=["api", "cli"],
        default=None,
        help="Extraction mode (overrides config)",
    )
    parser.add_argument(
        "--summary-model",
        type=str,
        default=None,
        help="Override model for summary extraction fields",
    )
    parser.add_argument(
        "--methodology-model",
        type=str,
        default=None,
        help="Override model for methodology extraction fields",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers to process",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def process_new_papers(
    new_keys: list[str],
    zotero_db: ZoteroDatabase,
    extractor: SectionExtractor | None,
    store: StructuredStore,
    embedding_gen: EmbeddingGenerator | None,
    vector_store: VectorStore | None,
    skip_extraction: bool,
    skip_embeddings: bool,
    logger,
) -> tuple[int, int]:
    """Process newly added papers.

    Args:
        new_keys: List of Zotero keys for new items.
        zotero_db: Zotero database interface.
        extractor: Section extractor for LLM analysis.
        store: Structured store for papers/extractions.
        embedding_gen: Embedding generator.
        vector_store: Vector store for embeddings.
        skip_extraction: Skip LLM extraction.
        skip_embeddings: Skip embedding generation.
        logger: Logger instance.

    Returns:
        Tuple of (successful, failed) counts.
    """
    if not new_keys:
        return 0, 0

    logger.info(f"Processing {len(new_keys)} new papers...")

    # Get paper metadata for new items
    papers = []
    for key in tqdm(new_keys, desc="Loading new papers"):
        paper = zotero_db.get_paper_by_key(key)
        if paper:
            papers.append(paper)

    if not papers:
        logger.warning("No new papers could be loaded")
        return 0, 0

    # Load existing data
    existing_papers = store.load_papers()
    existing_extractions = store.load_extractions()

    # Add paper metadata
    for paper in papers:
        existing_papers[paper.paper_id] = paper.to_index_dict()

    successful = 0
    failed = 0

    # Run extraction if not skipped
    if not skip_extraction:
        if extractor is None:
            logger.error("Extractor not initialized but extraction requested")
            return 0, len(papers)
        logger.info("Running LLM extraction on new papers...")
        for result in tqdm(
            extractor.extract_batch(papers),
            total=len(papers),
            desc="Extracting"
        ):
            if result.success and result.extraction:
                existing_extractions[result.paper_id] = {
                    "paper_id": result.paper_id,
                    "extraction": result.extraction.to_index_dict(),
                    "timestamp": result.timestamp.isoformat(),
                    "model": result.model_used,
                    "duration": result.duration_seconds,
                }
                successful += 1
            else:
                logger.warning(f"Failed to extract {result.paper_id}: {result.error}")
                failed += 1
    else:
        successful = len(papers)

    # Save updated data
    store.save_papers(existing_papers)
    store.save_extractions(existing_extractions)

    # Generate embeddings if not skipped
    if not skip_embeddings and successful > 0:
        logger.info("Generating embeddings for new papers...")
        papers_with_extractions = [
            p for p in papers
            if p.paper_id in existing_extractions
        ]

        all_chunks = []
        for paper in papers_with_extractions:
            ext_data = existing_extractions.get(paper.paper_id, {})
            ext = ext_data.get("extraction", ext_data)
            if ext:
                try:
                    extraction = PaperExtraction(**ext)
                    chunks = embedding_gen.create_chunks(paper, extraction)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.warning(f"Failed to create chunks for {paper.paper_id}: {e}")

        if all_chunks:
            all_chunks = embedding_gen.generate_embeddings(all_chunks, batch_size=32)
            added = vector_store.add_chunks(all_chunks, batch_size=100)
            logger.info(f"Added {added} chunks to vector store")

    return successful, failed


def process_modified_papers(
    modified_ids: list[str],
    store: StructuredStore,
    zotero_db: ZoteroDatabase,
    extractor: SectionExtractor | None,
    embedding_gen: EmbeddingGenerator | None,
    vector_store: VectorStore | None,
    skip_extraction: bool,
    skip_embeddings: bool,
    logger,
) -> tuple[int, int]:
    """Process modified papers by re-extracting and re-embedding.

    Args:
        modified_ids: List of paper IDs for modified items.
        store: Structured store for papers/extractions.
        zotero_db: Zotero database interface.
        extractor: Section extractor for LLM analysis (None if skip_extraction).
        embedding_gen: Embedding generator (None if skip_embeddings).
        vector_store: Vector store for embeddings (None if skip_embeddings).
        skip_extraction: Skip LLM extraction.
        skip_embeddings: Skip embedding generation.
        logger: Logger instance.

    Returns:
        Tuple of (successful, failed) counts.
    """
    if not modified_ids:
        return 0, 0

    logger.info(f"Processing {len(modified_ids)} modified papers...")

    # Load current data
    existing_papers = store.load_papers()
    existing_extractions = store.load_extractions()

    # Get Zotero keys for modified papers
    papers = []
    for paper_id in modified_ids:
        paper_data = existing_papers.get(paper_id)
        if paper_data:
            zotero_key = paper_data.get("zotero_key")
            if zotero_key:
                paper = zotero_db.get_paper_by_key(zotero_key)
                if paper:
                    # Keep the same paper_id
                    paper.paper_id = paper_id
                    papers.append(paper)

    if not papers:
        logger.warning("No modified papers could be loaded")
        return 0, 0

    # Delete old embeddings first
    if not skip_embeddings and vector_store is not None:
        for paper in papers:
            vector_store.delete_paper(paper.paper_id)

    # Update paper metadata
    for paper in papers:
        existing_papers[paper.paper_id] = paper.to_index_dict()

    successful = 0
    failed = 0

    # Run extraction if not skipped
    if not skip_extraction:
        if extractor is None:
            logger.error("Extractor not initialized but extraction requested")
            return 0, len(papers)
        logger.info("Re-extracting modified papers...")
        for result in tqdm(
            extractor.extract_batch(papers),
            total=len(papers),
            desc="Extracting"
        ):
            if result.success and result.extraction:
                existing_extractions[result.paper_id] = {
                    "paper_id": result.paper_id,
                    "extraction": result.extraction.to_index_dict(),
                    "timestamp": result.timestamp.isoformat(),
                    "model": result.model_used,
                    "duration": result.duration_seconds,
                }
                successful += 1
            else:
                logger.warning(f"Failed to extract {result.paper_id}: {result.error}")
                failed += 1
    else:
        successful = len(papers)

    # Save updated data
    store.save_papers(existing_papers)
    store.save_extractions(existing_extractions)

    # Generate new embeddings if not skipped
    if not skip_embeddings and successful > 0:
        logger.info("Generating embeddings for modified papers...")
        papers_with_extractions = [
            p for p in papers
            if p.paper_id in existing_extractions
        ]

        all_chunks = []
        for paper in papers_with_extractions:
            ext_data = existing_extractions.get(paper.paper_id, {})
            ext = ext_data.get("extraction", ext_data)
            if ext:
                try:
                    extraction = PaperExtraction(**ext)
                    chunks = embedding_gen.create_chunks(paper, extraction)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.warning(f"Failed to create chunks for {paper.paper_id}: {e}")

        if all_chunks:
            all_chunks = embedding_gen.generate_embeddings(all_chunks, batch_size=32)
            added = vector_store.add_chunks(all_chunks, batch_size=100)
            logger.info(f"Added {added} chunks to vector store")

    return successful, failed


def process_deleted_papers(
    deleted_ids: list[str],
    store: StructuredStore,
    vector_store: VectorStore,
    logger,
) -> int:
    """Remove deleted papers from index.

    Args:
        deleted_ids: List of paper IDs to delete.
        store: Structured store for papers/extractions.
        vector_store: Vector store for embeddings.
        logger: Logger instance.

    Returns:
        Number of papers deleted.
    """
    if not deleted_ids:
        return 0

    logger.info(f"Removing {len(deleted_ids)} deleted papers...")

    # Load current data
    existing_papers = store.load_papers()
    existing_extractions = store.load_extractions()

    # Remove from structured store
    for paper_id in deleted_ids:
        if paper_id in existing_papers:
            del existing_papers[paper_id]
        if paper_id in existing_extractions:
            del existing_extractions[paper_id]

    # Save updated data
    store.save_papers(existing_papers)
    store.save_extractions(existing_extractions)

    # Remove from vector store
    total_chunks = vector_store.delete_papers(deleted_ids)
    logger.info(f"Removed {total_chunks} chunks from vector store")

    return len(deleted_ids)


def print_change_summary(changes: ChangeSet, verbose: bool = False):
    """Print summary of detected changes."""
    print("\n=== Change Detection Summary ===")
    print(f"New papers:       {len(changes.new_items)}")
    print(f"Modified papers:  {len(changes.modified_items)}")
    print(f"Deleted papers:   {len(changes.deleted_items)}")
    print(f"Unchanged papers: {len(changes.unchanged_items)}")
    print(f"Total changes:    {changes.total_changes}")

    if verbose:
        if changes.new_items:
            print("\nNew items (Zotero keys):")
            for key in changes.new_items[:10]:
                print(f"  + {key}")
            if len(changes.new_items) > 10:
                print(f"  ... and {len(changes.new_items) - 10} more")

        if changes.modified_items:
            print("\nModified items (paper IDs):")
            for pid in changes.modified_items[:10]:
                print(f"  ~ {pid}")
            if len(changes.modified_items) > 10:
                print(f"  ... and {len(changes.modified_items) - 10} more")

        if changes.deleted_items:
            print("\nDeleted items (paper IDs):")
            for pid in changes.deleted_items[:10]:
                print(f"  - {pid}")
            if len(changes.deleted_items) > 10:
                print(f"  ... and {len(changes.deleted_items) - 10} more")


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)

    # Load configuration
    try:
        config = Config.load(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Setup directories
    index_dir = project_root / "data" / "index"
    if not index_dir.exists():
        logger.error(
            f"Index directory not found: {index_dir}\n"
            "Please run build_index.py first to create the initial index."
        )
        return 1

    # Connect to Zotero
    logger.info("Connecting to Zotero database...")
    try:
        zotero_db = ZoteroDatabase(
            config.get_zotero_db_path(),
            config.get_storage_path(),
        )
    except Exception as e:
        logger.error(f"Failed to connect to Zotero: {e}")
        return 1

    # Initialize change detector
    detector = ChangeDetector(zotero_db=zotero_db, index_dir=index_dir)

    # Detect changes
    changes = detector.detect_changes()
    print_change_summary(changes, args.verbose)

    # Exit if detect-only mode
    if args.detect_only:
        return 0

    # Exit if no changes
    if not changes.has_changes:
        logger.info("No changes to process")
        return 0

    # Apply limits
    new_items = changes.new_items
    modified_items = changes.modified_items
    deleted_items = changes.deleted_items

    if args.limit:
        total_to_process = len(new_items) + len(modified_items)
        if total_to_process > args.limit:
            # Prioritize new items, then modified
            if len(new_items) >= args.limit:
                new_items = new_items[:args.limit]
                modified_items = []
            else:
                remaining = args.limit - len(new_items)
                modified_items = modified_items[:remaining]
            logger.info(f"Limited to {args.limit} items")

    # Filter based on options
    if args.new_only:
        modified_items = []
        deleted_items = []
    if args.delete_only:
        new_items = []
        modified_items = []

    # Initialize components
    store = StructuredStore(index_dir)
    mode = args.mode or config.extraction.mode
    cache_dir = config.get_cache_path() / "pdf_text"

    model_by_type = {}
    if args.summary_model:
        model_by_type["summary"] = args.summary_model
    if args.methodology_model:
        model_by_type["methodology"] = args.methodology_model
    if not model_by_type:
        model_by_type = None

    extractor = None
    if not args.skip_extraction and (new_items or modified_items):
        extractor = SectionExtractor(
            cache_dir=cache_dir,
            provider=config.extraction.provider,
            mode=mode,
            model=config.extraction.model,
            max_tokens=config.extraction.max_tokens,
            timeout=config.extraction.timeout,
            model_by_type=model_by_type,
            min_text_length=config.processing.min_text_length,
            ocr_on_fail=config.processing.ocr_on_fail,
            skip_non_publications=config.processing.skip_non_publications,
            min_publication_words=config.processing.min_publication_words,
            min_publication_pages=config.processing.min_publication_pages,
            min_section_hits=config.processing.min_section_hits,
            ocr_enabled=config.processing.ocr_enabled,
            ocr_config=config.processing.ocr_config,
            reasoning_effort=config.extraction.reasoning_effort,
        )

    embedding_gen = None
    vector_store = None
    if not args.skip_embeddings:
        chroma_dir = index_dir / "chroma"
        embedding_gen = EmbeddingGenerator(model_name=config.embeddings.model)
        vector_store = VectorStore(chroma_dir)

    # Track results
    start_time = datetime.now()
    new_success, new_failed = 0, 0
    mod_success, mod_failed = 0, 0
    deleted_count = 0

    # Process new papers
    if new_items:
        new_success, new_failed = process_new_papers(
            new_items,
            zotero_db,
            extractor,
            store,
            embedding_gen,
            vector_store,
            args.skip_extraction,
            args.skip_embeddings,
            logger,
        )

    # Process modified papers
    if modified_items:
        mod_success, mod_failed = process_modified_papers(
            modified_items,
            store,
            zotero_db,
            extractor,
            embedding_gen,
            vector_store,
            args.skip_extraction,
            args.skip_embeddings,
            logger,
        )

    # Process deleted papers
    if deleted_items:
        deleted_count = process_deleted_papers(
            deleted_items,
            store,
            vector_store,
            logger,
        )

    # Update summary
    try:
        store.save_summary()
    except Exception as e:
        logger.warning(f"Failed to update summary: {e}")

    # Update metadata
    try:
        metadata = store.load_metadata()
        metadata["last_update"] = datetime.now().isoformat()
        metadata["last_update_stats"] = {
            "new_added": new_success,
            "new_failed": new_failed,
            "modified_updated": mod_success,
            "modified_failed": mod_failed,
            "deleted": deleted_count,
        }
        from src.utils.file_utils import safe_write_json
        safe_write_json(index_dir / "metadata.json", metadata)
    except Exception as e:
        logger.warning(f"Failed to update metadata: {e}")

    # Print summary
    duration = datetime.now() - start_time
    print("\n=== Update Complete ===")
    print(f"Duration: {duration}")
    print(f"New papers:      {new_success} added, {new_failed} failed")
    print(f"Modified papers: {mod_success} updated, {mod_failed} failed")
    print(f"Deleted papers:  {deleted_count} removed")

    return 0 if (new_failed + mod_failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
