#!/usr/bin/env python
"""Auto-loop index-scoped semantic batch extraction."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.indexing.semantic_jobs import (
    collect_semantic_batch,
    plan_semantic_batch,
    submit_semantic_batch,
    wait_for_semantic_batch,
)
from src.utils.logging_config import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-loop semantic batch extraction")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=project_root / "data" / "index")
    parser.add_argument("--dimension-profile", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=None, help="Stop after N batches")
    parser.add_argument("--pause", type=int, default=30, help="Seconds between submit and first poll")
    args = parser.parse_args()

    logger = setup_logging(level="INFO")
    config = Config.load(args.config)

    batch_count = 0
    total_collected = 0

    while True:
        plan = plan_semantic_batch(
            index_dir=args.index_dir,
            config=config,
            provider=args.provider,
            limit=args.batch_size,
            profile_reference=args.dimension_profile,
        )
        if not plan.selected:
            print("\nNo more eligible papers for batch extraction.")
            break

        if args.max_batches and batch_count >= args.max_batches:
            print(f"\nReached max batches ({args.max_batches}). Stopping.")
            break

        print(f"\n{'=' * 60}")
        print(f"Batch {batch_count + 1}: submitting {len(plan.selected)} papers...")
        print(f"{'=' * 60}")

        manifest = submit_semantic_batch(plan, config=config)
        batch_id = manifest["batch_id"]
        print(f"Submitted: {batch_id}")

        print(f"Waiting {args.pause}s before first poll...")
        time.sleep(args.pause)

        _manifest, status = wait_for_semantic_batch(
            index_dir=args.index_dir,
            batch_id=batch_id,
            config=config,
            poll_interval=60,
            progress_callback=lambda state: print(
                f"  {state.completed_requests}/{state.total_requests} ({state.status})",
                flush=True,
            ),
        )
        if status.status not in {"completed", "ended"}:
            print(f"Batch ended in terminal status: {status.status}")
            break

        summary = collect_semantic_batch(
            index_dir=args.index_dir,
            batch_id=batch_id,
            config=config,
            logger_override=logger,
        )
        batch_count += 1
        total_collected += int(summary["added"])
        print(
            f"\nBatch {batch_count} complete: {summary['added']} papers collected"
        )
        print(
            f"Running total: {total_collected} new + "
            f"{summary['total_extractions']} total extractions"
        )

    print(f"\n{'=' * 60}")
    print(f"Loop complete: {batch_count} batches, {total_collected} papers extracted")
    print("\nGenerate embeddings: python scripts/build_index.py --skip-extraction")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
