#!/usr/bin/env python
"""Build the literature review index from Zotero library."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm

from src.analysis.schemas import ExtractionResult
from src.analysis.section_extractor import SectionExtractor
from src.config import Config
from src.utils.file_utils import safe_read_json, safe_write_json
from src.utils.logging_config import LogContext, setup_logging
from src.zotero.database import ZoteroDatabase


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
    return set(data.keys())


def save_checkpoint(
    index_dir: Path,
    papers: list[dict],
    extractions: dict[str, dict],
    metadata: dict,
):
    """Save current progress to disk."""
    # Save papers
    safe_write_json(index_dir / "papers.json", papers)

    # Save extractions
    safe_write_json(index_dir / "extractions.json", extractions)

    # Save metadata
    safe_write_json(index_dir / "metadata.json", metadata)


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

    # Apply limit if specified
    if args.limit:
        papers = papers[:args.limit]
        logger.info(f"Limited to {len(papers)} papers")

    # Check for resume
    processed_ids = set()
    if args.resume:
        processed_ids = load_checkpoint(index_dir)
        papers = [p for p in papers if p.paper_id not in processed_ids]
        logger.info(f"Resuming: {len(processed_ids)} already processed, {len(papers)} remaining")

    if not papers:
        logger.info("No papers to process")
        return 0

    # Dry run - just show what would be processed
    if args.dry_run:
        print(f"\nWould process {len(papers)} papers:")
        for i, paper in enumerate(papers[:20], 1):
            print(f"  {i}. {paper.title[:60]}...")
        if len(papers) > 20:
            print(f"  ... and {len(papers) - 20} more")
        return 0

    # Initialize extractor
    mode = args.mode or config.extraction.mode
    extractor = SectionExtractor(
        cache_dir=cache_dir,
        mode=mode,
        model=config.extraction.model,
        max_tokens=config.extraction.max_tokens,
        min_text_length=config.processing.min_text_length,
    )

    # Cost estimation
    if args.estimate_cost:
        print("\nEstimating extraction cost...")
        estimate = extractor.estimate_batch_cost(papers)
        print(f"\nPapers with PDFs: {estimate['papers_with_pdf']}")
        if "average_text_length" in estimate:
            print(f"Average text length: {estimate['average_text_length']:,} chars")
            print(f"Cost per paper: ${estimate['estimated_cost_per_paper']:.4f}")
            print(f"Total estimated cost: ${estimate['estimated_total_cost']:.2f}")
            print(f"Model: {estimate['model']}")
        else:
            print(f"Note: {estimate.get('note', 'Unknown')}")
        return 0

    # Run extraction
    logger.info(f"Starting extraction with mode: {mode}")
    start_time = datetime.now()

    # Load existing data for resume
    existing_papers = safe_read_json(index_dir / "papers.json", default=[])
    existing_extractions = safe_read_json(index_dir / "extractions.json", default={})

    # Convert papers to dicts
    paper_dicts = {p["paper_id"]: p for p in existing_papers}
    for paper in papers:
        paper_dicts[paper.paper_id] = paper.to_index_dict()

    # Progress bar
    pbar = tqdm(total=len(papers), desc="Extracting", unit="paper")

    def progress_callback(current, total, title):
        pbar.set_postfix_str(title[:30] + "...")
        pbar.update(1)

    # Run extraction
    results = []
    try:
        for result in extractor.extract_batch(papers):
            results.append(result)

            # Save extraction if successful
            if result.success and result.extraction:
                existing_extractions[result.paper_id] = {
                    "extraction": result.extraction.to_index_dict(),
                    "timestamp": result.timestamp.isoformat(),
                    "model": result.model_used,
                    "duration": result.duration_seconds,
                }

            # Checkpoint every 10 papers
            if len(results) % 10 == 0:
                save_checkpoint(
                    index_dir,
                    list(paper_dicts.values()),
                    existing_extractions,
                    {
                        "last_updated": datetime.now().isoformat(),
                        "paper_count": len(paper_dicts),
                        "extraction_count": len(existing_extractions),
                    },
                )

            pbar.update(1)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user, saving checkpoint...")
    finally:
        pbar.close()

    # Final save
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
        },
    )

    # Summary
    duration = datetime.now() - start_time
    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    logger.info(f"\nExtraction complete in {duration}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Total extractions: {len(existing_extractions)}")
    logger.info(f"  Output: {index_dir}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
