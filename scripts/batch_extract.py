#!/usr/bin/env python
"""Batch extraction via OpenAI or Anthropic Batch API.

Submits papers for async 6-pass extraction with 50% cost savings.

Usage:
    # Estimate cost
    python scripts/batch_extract.py estimate --limit 50

    # Submit batch (OpenAI default)
    python scripts/batch_extract.py submit --limit 50
    python scripts/batch_extract.py submit --limit 50 --provider anthropic

    # Check status
    python scripts/batch_extract.py status <batch_id>

    # Wait for completion
    python scripts/batch_extract.py wait <batch_id>

    # Collect results into index
    python scripts/batch_extract.py collect <batch_id>

    # List pending batches
    python scripts/batch_extract.py pending
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.text_cleaner import TextCleaner
from src.indexing.structured_store import safe_read_json, safe_write_json
from src.utils.logging_config import setup_logging
from src.zotero.database import ZoteroDatabase


def create_client(provider: str, config: Config):
    """Create the appropriate batch client.

    Args:
        provider: "openai" or "anthropic".
        config: Application config.

    Returns:
        Batch client instance.
    """
    config.extraction.apply_provider(provider)
    batch_dir = project_root / "data" / "batches"

    if provider == "openai":
        from src.analysis.openai_batch_client import OpenAIBatchClient

        return OpenAIBatchClient(
            model=config.extraction.model,
            reasoning_effort=config.extraction.reasoning_effort,
            batch_dir=batch_dir,
        )
    elif provider == "anthropic":
        from src.analysis.batch_client import BatchExtractionClient

        return BatchExtractionClient(
            model=config.extraction.model,
            batch_dir=batch_dir,
        )
    else:
        print(f"Batch API not supported for provider: {provider}")
        sys.exit(1)


def get_unextracted_papers(config: Config, limit: int | None = None):
    """Get papers not yet in semantic_analyses.json."""
    index_dir = project_root / "data" / "index"

    db = ZoteroDatabase(config.zotero.database_path, config.zotero.storage_path)
    all_papers = list(db.get_all_papers())
    papers = [p for p in all_papers if p.pdf_path and p.pdf_path.exists()]

    sa_data = safe_read_json(index_dir / "semantic_analyses.json", default={})
    if isinstance(sa_data, dict) and "extractions" in sa_data:
        existing_ids = set(sa_data["extractions"].keys())
    else:
        existing_ids = set()

    # Load skip list (papers rejected by content filters)
    skip_data = safe_read_json(index_dir / "skip_papers.json", default={})
    skip_ids = set(skip_data.get("papers", {}).keys())

    unextracted = [
        p for p in papers
        if p.paper_id not in existing_ids and p.paper_id not in skip_ids
    ]
    print(f"Total papers with PDFs: {len(papers)}")
    print(f"Already extracted: {len(existing_ids)}")
    if skip_ids:
        print(f"Skipped (content policy): {len(skip_ids)}")
    print(f"Remaining: {len(unextracted)}")

    if limit:
        unextracted = unextracted[:limit]
        print(f"Limited to: {len(unextracted)}")

    return unextracted


def make_text_getter(config: Config):
    """Create a function that extracts and cleans text from a paper.

    Checks the cascade text cache first (from preextract_text.py).
    Falls back to the full extraction cascade: arXiv HTML -> PyMuPDF -> Marker -> OCR.
    """
    from src.extraction.cascade import ExtractionCascade

    cascade_cache = Path("data/cache/cascade_text")

    pdf_extractor = PDFExtractor(
        enable_ocr=config.processing.ocr_enabled or config.processing.ocr_on_fail,
    )
    text_cleaner = TextCleaner()
    cascade = ExtractionCascade(
        pdf_extractor=pdf_extractor,
        enable_arxiv=config.processing.arxiv_enabled,
        enable_marker=config.processing.marker_enabled,
    )

    def text_getter(paper):
        if not paper.pdf_path or not paper.pdf_path.exists():
            raise ValueError(f"No PDF for {paper.paper_id}")

        # Check cascade text cache first
        cached_path = cascade_cache / f"{paper.paper_id}.txt"
        if cached_path.exists():
            text = cached_path.read_text(encoding="utf-8")
            if text:
                return text

        # Fall back to live cascade extraction
        result = cascade.extract_text(
            paper.pdf_path,
            doi=getattr(paper, "doi", None),
            url=getattr(paper, "url", None),
        )
        text = result.text
        if not text:
            raise ValueError(f"Empty text for {paper.paper_id}")
        if not result.is_markdown:
            text = text_cleaner.clean(text)
        text = text_cleaner.truncate_for_llm(text)
        return text

    return text_getter


def cmd_estimate(args, config, logger):
    """Estimate batch cost."""
    papers = get_unextracted_papers(config, args.limit)
    if not papers:
        print("No papers to estimate.")
        return 0

    client = create_client(args.provider, config)
    avg_text_length = 40000  # ~10K tokens typical
    estimate = client.estimate_cost(len(papers), avg_text_length)

    print(f"\n=== Batch Cost Estimate ({args.provider}) ===")
    for key, value in estimate.items():
        if isinstance(value, int) and value > 1000:
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")
    return 0


def cmd_submit(args, config, logger):
    """Submit a new batch."""
    papers = get_unextracted_papers(config, args.limit)
    if not papers:
        print("No papers to submit.")
        return 0

    client = create_client(args.provider, config)
    text_getter = make_text_getter(config)

    print(f"\nCreating batch requests for {len(papers)} papers "
          f"({len(papers) * 6} API calls)...")
    requests = client.create_batch_requests(papers, text_getter)
    print(f"Created {len(requests)} requests")

    # Show cost estimate
    avg_text_length = 40000
    estimate = client.estimate_cost(len(papers), avg_text_length)
    print(f"Estimated cost: ${estimate['estimated_cost']:.2f} ({estimate['discount']})")

    if args.dry_run:
        print("\n[DRY RUN] Would submit the above. Run without --dry-run to execute.")
        return 0

    try:
        response = input(f"\nSubmit batch for {len(papers)} papers? [y/N]: ")
        if response.lower() not in ("y", "yes"):
            print("Cancelled.")
            return 0
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 0

    batch_id = client.submit_batch(requests)
    print(f"\nBatch submitted: {batch_id}")
    print("\nNext steps:")
    print(f"  python scripts/batch_extract.py status {batch_id}")
    print(f"  python scripts/batch_extract.py wait {batch_id}")
    print(f"  python scripts/batch_extract.py collect {batch_id} --provider {args.provider}")


def cmd_status(args, config, logger):
    """Check batch status."""
    client = create_client(args.provider, config)
    status = client.get_batch_status(args.batch_id)

    print(f"\nBatch: {status.batch_id}")
    print(f"Status: {status.status}")
    print(f"Progress: {status.completed_requests}/{status.total_requests}")
    if status.failed_requests > 0:
        print(f"Failed: {status.failed_requests}")
    if hasattr(status, "output_file_id") and status.output_file_id:
        print(f"Output file: {status.output_file_id}")
    return 0


def cmd_wait(args, config, logger):
    """Wait for batch completion."""
    client = create_client(args.provider, config)

    def progress(status):
        print(
            f"  {status.completed_requests}/{status.total_requests} "
            f"({status.status})",
            flush=True,
        )

    print(f"Waiting for batch {args.batch_id}...")
    status = client.wait_for_batch(
        args.batch_id,
        poll_interval=args.poll_interval,
        progress_callback=progress,
    )
    print(f"\nComplete: {status.completed_requests} succeeded, "
          f"{status.failed_requests} failed")
    return 0


def cmd_collect(args, config, logger):
    """Collect results into the index."""
    client = create_client(args.provider, config)
    index_dir = project_root / "data" / "index"
    sa_path = index_dir / "semantic_analyses.json"

    print(f"Collecting results for batch {args.batch_id}...")

    # Load existing
    if sa_path.exists():
        sa_data = json.loads(sa_path.read_text(encoding="utf-8"))
    else:
        sa_data = {"schema_version": "1.0", "extractions": {}}

    extractions = sa_data.get("extractions", {})
    added = 0
    failed = 0

    for result in client.get_results(args.batch_id):
        if result.success and result.extraction:
            extractions[result.paper_id] = {
                "paper_id": result.paper_id,
                "extraction": result.extraction.to_index_dict(),
                "timestamp": datetime.now().isoformat(),
                "model": result.model_used,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "batch_id": args.batch_id,
            }
            added += 1
            print(f"  {result.paper_id}: "
                  f"{result.extraction.dimension_coverage:.0%} coverage")
        else:
            failed += 1
            print(f"  {result.paper_id}: FAILED - {result.error}")

    sa_data["extractions"] = extractions
    sa_data["extraction_count"] = len(extractions)
    sa_data["generated_at"] = datetime.now().isoformat()
    safe_write_json(sa_path, sa_data)

    print(f"\nCollected: {added} successful, {failed} failed")
    print(f"Total extractions: {len(extractions)}")
    print("\nGenerate embeddings: python scripts/build_index.py --skip-extraction")
    return 0


def cmd_pending(args, config, logger):
    """List pending batches."""
    client = create_client(args.provider, config)
    pending = client.list_pending_batches()
    if pending:
        print(f"Pending batches ({len(pending)}):")
        for batch_id in pending:
            status = client.get_batch_status(batch_id)
            print(f"  {batch_id}: {status.status} "
                  f"({status.completed_requests}/{status.total_requests})")
    else:
        print("No pending batches.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Batch extraction via OpenAI/Anthropic Batch API",
    )
    parser.add_argument(
        "--provider", default="openai", choices=["openai", "anthropic"],
        help="Batch API provider (default: openai)",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # estimate
    est = subparsers.add_parser("estimate", help="Estimate batch cost")
    est.add_argument("--limit", type=int, default=None)

    # submit
    sub = subparsers.add_parser("submit", help="Submit batch")
    sub.add_argument("--limit", type=int, default=None)
    sub.add_argument("--dry-run", action="store_true")

    # status
    st = subparsers.add_parser("status", help="Check batch status")
    st.add_argument("batch_id")

    # wait
    wt = subparsers.add_parser("wait", help="Wait for completion")
    wt.add_argument("batch_id")
    wt.add_argument("--poll-interval", type=int, default=60)

    # collect
    col = subparsers.add_parser("collect", help="Collect results")
    col.add_argument("batch_id")

    # pending
    subparsers.add_parser("pending", help="List pending batches")

    args = parser.parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)

    config = Config.load(args.config)

    commands = {
        "estimate": cmd_estimate,
        "submit": cmd_submit,
        "status": cmd_status,
        "wait": cmd_wait,
        "collect": cmd_collect,
        "pending": cmd_pending,
    }

    return commands[args.command](args, config, logger)


if __name__ == "__main__":
    sys.exit(main() or 0)
