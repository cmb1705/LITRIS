#!/usr/bin/env python
"""Build the literature review index from Zotero library."""

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
from src.utils.checkpoint import CheckpointManager
from src.utils.file_utils import safe_read_json, safe_write_json
from src.utils.logging_config import LogContext, setup_logging
from src.zotero.database import ZoteroDatabase
from src.zotero.models import PaperMetadata


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build literature review index from Zotero library"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers to process",
    )
    parser.add_argument(
        "--mode",
        choices=["api", "cli"],
        default=None,
        help="Extraction mode (overrides config)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running",
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Estimate cost without running extraction",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip LLM extraction (use existing extractions)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation and vector store",
    )
    parser.add_argument(
        "--rebuild-embeddings",
        action="store_true",
        help="Rebuild embeddings even if they exist",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry previously failed extractions",
    )
    parser.add_argument(
        "--skip-paper",
        type=str,
        action="append",
        default=[],
        metavar="PAPER_ID",
        help="Skip specific paper ID (can be used multiple times)",
    )
    parser.add_argument(
        "--show-failed",
        action="store_true",
        help="Show list of failed papers from previous run",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Reset checkpoint to start fresh",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def load_checkpoint(index_dir: Path) -> set[str]:
    """Load set of already-processed paper IDs."""
    extractions_file = index_dir / "extractions.json"
    if not extractions_file.exists():
        return set()

    data = safe_read_json(extractions_file, default={})
    if isinstance(data, dict) and "extractions" in data:
        return set(data["extractions"].keys())
    return set(data.keys())


def save_checkpoint(
    index_dir: Path,
    papers: list[dict],
    extractions: dict[str, dict],
    metadata: dict,
):
    """Save current progress to disk."""
    # Save papers using structured store format
    papers_data = {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "paper_count": len(papers),
        "papers": papers,
    }
    safe_write_json(index_dir / "papers.json", papers_data)

    # Save extractions
    extractions_data = {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "extraction_count": len(extractions),
        "extractions": extractions,
    }
    safe_write_json(index_dir / "extractions.json", extractions_data)

    # Save metadata
    safe_write_json(index_dir / "metadata.json", metadata)


def run_extraction(
    papers: list[PaperMetadata],
    extractor: SectionExtractor,
    index_dir: Path,
    existing_papers: dict,
    existing_extractions: dict,
    checkpoint_mgr: CheckpointManager,
    logger,
) -> tuple[dict, dict, list]:
    """Run LLM extraction on papers.

    Returns:
        Tuple of (paper_dicts, extractions, results)
    """
    # Convert papers to dicts
    paper_dicts = dict(existing_papers)
    for paper in papers:
        paper_dicts[paper.paper_id] = paper.to_index_dict()

    # Initialize or update checkpoint
    state = checkpoint_mgr.load()
    if not state:
        checkpoint_mgr.initialize(
            total_items=len(papers),
            metadata={"started_at": datetime.now().isoformat()},
        )

    # Progress bar
    pbar = tqdm(total=len(papers), desc="Extracting", unit="paper")

    # Run extraction
    results = []
    try:
        for result in extractor.extract_batch(papers):
            results.append(result)

            # Track in checkpoint
            if result.success and result.extraction:
                checkpoint_mgr.complete_item(result.paper_id, success=True)
                existing_extractions[result.paper_id] = {
                    "paper_id": result.paper_id,
                    "extraction": result.extraction.to_index_dict(),
                    "timestamp": result.timestamp.isoformat(),
                    "model": result.model_used,
                    "duration": result.duration_seconds,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                }
            else:
                error = Exception(result.error) if result.error else None
                checkpoint_mgr.complete_item(result.paper_id, success=False, error=error)

            # Checkpoint every 10 papers
            if len(results) % 10 == 0:
                checkpoint_mgr.save()
                save_checkpoint(
                    index_dir,
                    list(paper_dicts.values()),
                    existing_extractions,
                    {
                        "last_updated": datetime.now().isoformat(),
                        "paper_count": len(paper_dicts),
                        "extraction_count": len(existing_extractions),
                        **checkpoint_mgr.get_progress(),
                    },
                )

            pbar.update(1)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user, saving checkpoint...")
        checkpoint_mgr.save()
    finally:
        pbar.close()
        checkpoint_mgr.save()

    return paper_dicts, existing_extractions, results


def run_embedding_generation(
    papers: list[PaperMetadata],
    extractions: dict[str, dict],
    index_dir: Path,
    embedding_model: str,
    rebuild: bool,
    logger,
) -> int:
    """Generate embeddings and populate vector store.

    Returns:
        Number of chunks added.
    """
    chroma_dir = index_dir / "chroma"

    logger.info("Initializing embedding generator...")
    embedding_gen = EmbeddingGenerator(model_name=embedding_model)

    logger.info("Initializing vector store...")
    vector_store = VectorStore(chroma_dir)

    if rebuild:
        logger.info("Clearing existing embeddings...")
        vector_store.clear()

    # Build paper lookup
    paper_lookup = {p.paper_id: p for p in papers}

    # Filter to papers with extractions
    papers_with_extractions = []
    extraction_lookup = {}

    for paper_id, ext_data in extractions.items():
        if paper_id in paper_lookup:
            papers_with_extractions.append(paper_lookup[paper_id])
            # Extract the actual extraction data
            ext = ext_data.get("extraction", ext_data)
            extraction_lookup[paper_id] = PaperExtraction(**ext)

    if not papers_with_extractions:
        logger.warning("No papers with extractions found for embedding")
        return 0

    logger.info(f"Generating embeddings for {len(papers_with_extractions)} papers...")

    # Create chunks
    all_chunks = []
    for paper in tqdm(papers_with_extractions, desc="Creating chunks"):
        extraction = extraction_lookup.get(paper.paper_id)
        if extraction:
            chunks = embedding_gen.create_chunks(paper, extraction)
            all_chunks.extend(chunks)

    logger.info(f"Created {len(all_chunks)} chunks")

    # Generate embeddings in batches
    logger.info("Generating embeddings...")
    all_chunks = embedding_gen.generate_embeddings(all_chunks, batch_size=32)

    # Add to vector store
    logger.info("Adding to vector store...")
    added = vector_store.add_chunks(all_chunks, batch_size=100)

    logger.info(f"Added {added} chunks to vector store")
    return added


def generate_summary(index_dir: Path, logger):
    """Generate summary statistics."""
    logger.info("Generating summary statistics...")
    store = StructuredStore(index_dir)
    summary = store.save_summary()
    logger.info(f"Summary saved with {summary.get('total_papers', 0)} papers")


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
    index_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = config.get_cache_path() / "pdf_text"

    # Connect to Zotero
    logger.info("Connecting to Zotero database...")
    try:
        zotero = ZoteroDatabase(
            config.get_zotero_db_path(),
            config.get_storage_path(),
        )
    except Exception as e:
        logger.error(f"Failed to connect to Zotero: {e}")
        return 1

    # Get all papers
    with LogContext(logger, "Loading papers from Zotero"):
        papers = list(zotero.get_all_papers())
        logger.info(f"Found {len(papers)} papers with PDFs")

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(index_dir, checkpoint_id="extraction")

    # Handle reset checkpoint
    if args.reset_checkpoint:
        checkpoint_mgr.reset()
        logger.info("Checkpoint reset, starting fresh")

    # Handle show failed
    if args.show_failed:
        state = checkpoint_mgr.load()
        if state and state.failed_items:
            print(f"\nFailed papers ({len(state.failed_items)}):")
            for item in state.failed_items:
                print(f"  - {item.item_id}")
                print(f"    Error: {item.error_type}: {item.error_message}")
                if item.retry_count > 0:
                    print(f"    Retries: {item.retry_count}")
        else:
            print("\nNo failed papers recorded")
        return 0

    # Apply limit if specified
    if args.limit:
        papers = papers[: args.limit]
        logger.info(f"Limited to {len(papers)} papers")

    # Build skip set from command line
    skip_paper_ids = set(args.skip_paper)
    if skip_paper_ids:
        logger.info(f"Will skip {len(skip_paper_ids)} papers from command line")

    # Check for resume or retry-failed
    processed_ids = set()
    papers_to_extract = papers

    if args.retry_failed and not args.skip_extraction:
        # Load checkpoint and retry failed items
        state = checkpoint_mgr.load()
        if state:
            failed_ids = set(checkpoint_mgr.get_failed_ids())
            if failed_ids:
                # Clear failed items so they can be retried
                checkpoint_mgr.clear_failed()
                papers_to_extract = [p for p in papers if p.paper_id in failed_ids]
                logger.info(f"Retrying {len(papers_to_extract)} previously failed papers")
            else:
                logger.info("No failed papers to retry")
                papers_to_extract = []
        else:
            logger.warning("No checkpoint found, nothing to retry")
            papers_to_extract = []

    elif args.resume and not args.skip_extraction:
        # Resume from checkpoint
        state = checkpoint_mgr.load()
        if state:
            processed_ids = set(state.processed_ids)
            failed_ids = set(checkpoint_mgr.get_failed_ids())
            papers_to_extract = [
                p for p in papers
                if p.paper_id not in processed_ids and p.paper_id not in failed_ids
            ]
            logger.info(
                f"Resuming: {len(processed_ids)} successful, {len(failed_ids)} failed, "
                f"{len(papers_to_extract)} remaining"
            )
        else:
            processed_ids = load_checkpoint(index_dir)  # Legacy checkpoint
            papers_to_extract = [p for p in papers if p.paper_id not in processed_ids]
            logger.info(
                f"Resuming (legacy): {len(processed_ids)} already processed, "
                f"{len(papers_to_extract)} remaining"
            )

    # Apply skip-paper filter
    if skip_paper_ids:
        papers_to_extract = [
            p for p in papers_to_extract if p.paper_id not in skip_paper_ids
        ]

    if not papers and not args.skip_extraction:
        logger.info("No papers to process")
        return 0

    # Dry run - just show what would be processed
    if args.dry_run:
        print(f"\nWould process {len(papers_to_extract)} papers:")
        for i, paper in enumerate(papers_to_extract[:20], 1):
            print(f"  {i}. {paper.title[:60]}...")
        if len(papers_to_extract) > 20:
            print(f"  ... and {len(papers_to_extract) - 20} more")
        return 0

    # Initialize extractor
    mode = args.mode or config.extraction.mode

    # Cost estimation
    if args.estimate_cost:
        extractor = SectionExtractor(
            cache_dir=cache_dir,
            mode=mode,
            model=config.extraction.model,
            max_tokens=config.extraction.max_tokens,
            min_text_length=config.processing.min_text_length,
        )
        print("\nEstimating extraction cost...")
        estimate = extractor.estimate_batch_cost(papers_to_extract)
        print(f"\nPapers with PDFs: {estimate['papers_with_pdf']}")
        if "average_text_length" in estimate:
            print(f"Average text length: {estimate['average_text_length']:,} chars")
            print(f"Cost per paper: ${estimate['estimated_cost_per_paper']:.4f}")
            print(f"Total estimated cost: ${estimate['estimated_total_cost']:.2f}")
            print(f"Model: {estimate['model']}")
        else:
            print(f"Note: {estimate.get('note', 'Unknown')}")
        return 0

    start_time = datetime.now()

    # Load existing data
    existing_papers_data = safe_read_json(index_dir / "papers.json", default={})
    if isinstance(existing_papers_data, dict) and "papers" in existing_papers_data:
        existing_papers = {
            p["paper_id"]: p for p in existing_papers_data["papers"] if "paper_id" in p
        }
    elif isinstance(existing_papers_data, list):
        existing_papers = {
            p["paper_id"]: p for p in existing_papers_data if "paper_id" in p
        }
    else:
        existing_papers = {}

    existing_extractions_data = safe_read_json(
        index_dir / "extractions.json", default={}
    )
    if isinstance(existing_extractions_data, dict) and "extractions" in existing_extractions_data:
        existing_extractions = existing_extractions_data["extractions"]
    else:
        existing_extractions = existing_extractions_data

    # Step 1: LLM Extraction
    results = []
    if not args.skip_extraction and papers_to_extract:
        logger.info(f"Starting extraction with mode: {mode}")
        extractor = SectionExtractor(
            cache_dir=cache_dir,
            mode=mode,
            model=config.extraction.model,
            max_tokens=config.extraction.max_tokens,
            min_text_length=config.processing.min_text_length,
        )

        paper_dicts, existing_extractions, results = run_extraction(
            papers_to_extract,
            extractor,
            index_dir,
            existing_papers,
            existing_extractions,
            checkpoint_mgr,
            logger,
        )

        # Final save after extraction
        checkpoint_mgr.save()
        save_checkpoint(
            index_dir,
            list(paper_dicts.values()),
            existing_extractions,
            {
                "last_updated": datetime.now().isoformat(),
                "paper_count": len(paper_dicts),
                "extraction_count": len(existing_extractions),
                "extraction_mode": mode,
                "model": config.extraction.model,
                **checkpoint_mgr.get_progress(),
            },
        )

        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        logger.info(f"Extraction: {successful} successful, {failed} failed")
    else:
        # Update paper dicts from existing
        paper_dicts = dict(existing_papers)
        for paper in papers:
            if paper.paper_id not in paper_dicts:
                paper_dicts[paper.paper_id] = paper.to_index_dict()

    # Step 2: Embedding Generation
    if not args.skip_embeddings:
        try:
            chunks_added = run_embedding_generation(
                papers,
                existing_extractions,
                index_dir,
                config.embeddings.model,
                args.rebuild_embeddings,
                logger,
            )
            logger.info(f"Embedding generation complete: {chunks_added} chunks")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Step 3: Summary Generation
    try:
        generate_summary(index_dir, logger)
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")

    # Final summary
    duration = datetime.now() - start_time
    logger.info(f"\nBuild complete in {duration}")
    logger.info(f"  Total papers: {len(paper_dicts)}")
    logger.info(f"  Total extractions: {len(existing_extractions)}")
    logger.info(f"  Output: {index_dir}")

    if results:
        failed = sum(1 for r in results if not r.success)
        return 0 if failed == 0 else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
