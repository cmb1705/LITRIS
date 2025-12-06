#!/usr/bin/env python
"""Batch extraction using Anthropic Batch API.

The Batch API processes requests asynchronously with 50% cost savings.
Batches complete within 24 hours (usually faster).

Workflow:
1. Submit: python batch_extract.py submit --limit 100
2. Check:  python batch_extract.py status <batch_id>
3. Collect: python batch_extract.py collect <batch_id>
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.batch_client import BatchExtractionClient
from src.config import Config
from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.text_cleaner import TextCleaner
from src.indexing.structured_store import StructuredStore
from src.utils.checkpoint import CheckpointManager
from src.utils.file_utils import safe_read_json, safe_write_json
from src.utils.logging_config import setup_logging
from src.zotero.database import ZoteroDatabase


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch extraction using Anthropic Batch API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit a new batch")
    submit_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml",
    )
    submit_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers",
    )
    submit_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be submitted without submitting",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Check batch status")
    status_parser.add_argument("batch_id", help="Batch ID to check")

    # List command
    subparsers.add_parser("list", help="List pending batches")

    # Wait command
    wait_parser = subparsers.add_parser("wait", help="Wait for batch to complete")
    wait_parser.add_argument("batch_id", help="Batch ID to wait for")
    wait_parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between status checks",
    )

    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect batch results")
    collect_parser.add_argument("batch_id", help="Batch ID to collect")
    collect_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml",
    )

    # Estimate command
    estimate_parser = subparsers.add_parser("estimate", help="Estimate batch cost")
    estimate_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml",
    )
    estimate_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def cmd_submit(args, logger):
    """Submit a new batch for extraction."""
    config = Config.load(args.config)
    index_dir = project_root / "data" / "index"
    cache_dir = config.get_cache_path() / "pdf_text"

    # Load papers from Zotero
    logger.info("Connecting to Zotero...")
    zotero = ZoteroDatabase(
        config.get_zotero_db_path(),
        config.get_storage_path(),
    )
    papers = list(zotero.get_all_papers())
    logger.info(f"Found {len(papers)} papers with PDFs")

    if args.limit:
        papers = papers[:args.limit]
        logger.info(f"Limited to {len(papers)} papers")

    # Check for already-extracted papers
    existing_extractions = safe_read_json(
        index_dir / "extractions.json", default={}
    )
    if isinstance(existing_extractions, dict) and "extractions" in existing_extractions:
        existing_ids = set(existing_extractions["extractions"].keys())
    else:
        existing_ids = set(existing_extractions.keys())

    papers_to_extract = [p for p in papers if p.paper_id not in existing_ids]
    logger.info(f"Papers to extract: {len(papers_to_extract)} (skipping {len(existing_ids)} existing)")

    if not papers_to_extract:
        logger.info("No papers to extract")
        return 0

    # Initialize extractors
    pdf_extractor = PDFExtractor(cache_dir=cache_dir)
    text_cleaner = TextCleaner()

    def get_text(paper):
        """Get cleaned text from paper."""
        if not paper.pdf_path or not paper.pdf_path.exists():
            raise ValueError("No PDF available")
        text = pdf_extractor.extract_text(paper.pdf_path)
        text = text_cleaner.clean(text)
        text = text_cleaner.truncate_for_llm(text)
        return text

    # Create batch client
    batch_client = BatchExtractionClient(
        model=config.extraction.model,
        max_tokens=config.extraction.max_tokens,
        batch_dir=project_root / "data" / "batches",
    )

    # Create batch requests
    logger.info("Creating batch requests...")
    requests = batch_client.create_batch_requests(papers_to_extract, get_text)
    logger.info(f"Created {len(requests)} requests")

    if args.dry_run:
        print(f"\nDry run: Would submit {len(requests)} requests")
        print("\nSample papers:")
        for req in requests[:5]:
            print(f"  - {req.paper.title[:60]}...")
        if len(requests) > 5:
            print(f"  ... and {len(requests) - 5} more")

        # Show cost estimate
        avg_length = 50000  # Rough average
        estimate = batch_client.estimate_cost(len(requests), avg_length)
        print(f"\nEstimated cost: ${estimate['estimated_cost']:.2f}")
        print(f"Discount: {estimate['discount']}")
        return 0

    # Submit batch
    batch_id = batch_client.submit_batch(requests)
    print(f"\nBatch submitted: {batch_id}")
    print(f"Papers: {len(requests)}")
    print("\nNext steps:")
    print(f"  1. Check status: python scripts/batch_extract.py status {batch_id}")
    print(f"  2. Wait for completion: python scripts/batch_extract.py wait {batch_id}")
    print(f"  3. Collect results: python scripts/batch_extract.py collect {batch_id}")

    return 0


def cmd_status(args, logger):
    """Check batch status."""
    batch_client = BatchExtractionClient(
        batch_dir=project_root / "data" / "batches",
    )

    status = batch_client.get_batch_status(args.batch_id)

    print(f"\nBatch: {status.batch_id}")
    print(f"Status: {status.status}")
    print(f"Created: {status.created_at}")
    print(f"Progress: {status.completed_requests}/{status.total_requests}")
    if status.failed_requests > 0:
        print(f"Failed: {status.failed_requests}")

    return 0


def cmd_list(args, logger):
    """List pending batches."""
    batch_client = BatchExtractionClient(
        batch_dir=project_root / "data" / "batches",
    )

    pending = batch_client.list_pending_batches()

    if not pending:
        print("\nNo pending batches")
        return 0

    print(f"\nPending batches ({len(pending)}):")
    for batch_id in pending:
        status = batch_client.get_batch_status(batch_id)
        print(f"  {batch_id}: {status.status} ({status.completed_requests}/{status.total_requests})")

    return 0


def cmd_wait(args, logger):
    """Wait for batch to complete."""
    batch_client = BatchExtractionClient(
        batch_dir=project_root / "data" / "batches",
    )

    def progress_callback(status):
        print(f"  {status.status}: {status.completed_requests}/{status.total_requests}")

    print(f"\nWaiting for batch {args.batch_id}...")
    status = batch_client.wait_for_batch(
        args.batch_id,
        poll_interval=args.poll_interval,
        progress_callback=progress_callback,
    )

    print(f"\nBatch completed!")
    print(f"Successful: {status.completed_requests}")
    print(f"Failed: {status.failed_requests}")

    return 0


def cmd_collect(args, logger):
    """Collect results from completed batch."""
    config = Config.load(args.config)
    index_dir = project_root / "data" / "index"

    batch_client = BatchExtractionClient(
        model=config.extraction.model,
        batch_dir=project_root / "data" / "batches",
    )

    # Check batch is complete
    status = batch_client.get_batch_status(args.batch_id)
    if status.status != "ended":
        print(f"Batch status is '{status.status}', not 'ended'. Wait for completion first.")
        return 1

    # Load existing data
    existing_papers = safe_read_json(index_dir / "papers.json", default={})
    if isinstance(existing_papers, dict) and "papers" in existing_papers:
        papers_dict = {p["paper_id"]: p for p in existing_papers["papers"]}
    else:
        papers_dict = {}

    existing_extractions = safe_read_json(index_dir / "extractions.json", default={})
    if isinstance(existing_extractions, dict) and "extractions" in existing_extractions:
        extractions_dict = existing_extractions["extractions"]
    else:
        extractions_dict = dict(existing_extractions)

    # Collect results
    logger.info(f"Collecting results from batch {args.batch_id}...")
    successful = 0
    failed = 0

    for result in batch_client.get_results(args.batch_id):
        if result.success and result.extraction:
            extractions_dict[result.paper_id] = {
                "paper_id": result.paper_id,
                "extraction": result.extraction.to_index_dict(),
                "timestamp": datetime.now().isoformat(),
                "model": result.model_used,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "batch_id": args.batch_id,
            }
            successful += 1
        else:
            logger.warning(f"Failed: {result.paper_id}: {result.error}")
            failed += 1

    # Save results
    extractions_data = {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "extraction_count": len(extractions_dict),
        "extractions": extractions_dict,
    }
    safe_write_json(index_dir / "extractions.json", extractions_data)

    print(f"\nCollected {successful} successful, {failed} failed")
    print(f"Total extractions: {len(extractions_dict)}")

    return 0


def cmd_estimate(args, logger):
    """Estimate batch cost."""
    config = Config.load(args.config)
    cache_dir = config.get_cache_path() / "pdf_text"

    # Load papers from Zotero
    logger.info("Connecting to Zotero...")
    zotero = ZoteroDatabase(
        config.get_zotero_db_path(),
        config.get_storage_path(),
    )
    papers = list(zotero.get_all_papers())

    if args.limit:
        papers = papers[:args.limit]

    # Sample text lengths
    pdf_extractor = PDFExtractor(cache_dir=cache_dir)
    text_cleaner = TextCleaner()

    sample_size = min(10, len(papers))
    sample_lengths = []

    for paper in papers[:sample_size]:
        try:
            if paper.pdf_path and paper.pdf_path.exists():
                text = pdf_extractor.extract_text(paper.pdf_path)
                text = text_cleaner.clean(text)
                text = text_cleaner.truncate_for_llm(text)
                sample_lengths.append(len(text))
        except Exception:
            continue

    if not sample_lengths:
        print("Could not sample papers for estimate")
        return 1

    avg_length = sum(sample_lengths) / len(sample_lengths)

    batch_client = BatchExtractionClient(
        model=config.extraction.model,
    )
    estimate = batch_client.estimate_cost(len(papers), int(avg_length))

    print(f"\nPapers: {estimate['num_papers']}")
    print(f"Average text length: {int(avg_length):,} chars")
    print(f"Estimated tokens: {estimate['estimated_input_tokens']:,} input, {estimate['estimated_output_tokens']:,} output")
    print(f"Estimated cost: ${estimate['estimated_cost']:.2f}")
    print(f"Discount: {estimate['discount']}")
    print(f"Model: {estimate['model']}")

    return 0


def main():
    """Main entry point."""
    args = parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)

    if args.command == "submit":
        return cmd_submit(args, logger)
    elif args.command == "status":
        return cmd_status(args, logger)
    elif args.command == "list":
        return cmd_list(args, logger)
    elif args.command == "wait":
        return cmd_wait(args, logger)
    elif args.command == "collect":
        return cmd_collect(args, logger)
    elif args.command == "estimate":
        return cmd_estimate(args, logger)
    else:
        print("Use --help for usage information")
        return 1


if __name__ == "__main__":
    sys.exit(main())
