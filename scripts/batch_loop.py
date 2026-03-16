#!/usr/bin/env python
"""Auto-loop batch extraction: submit -> wait -> collect -> repeat.

Submits batches of papers until all remaining papers are extracted.
Handles Tier 3 limits by submitting one batch at a time and waiting
for completion before submitting the next.

Usage:
    python scripts/batch_loop.py --batch-size 50
    python scripts/batch_loop.py --batch-size 50 --provider anthropic
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import batch_extract functions
import scripts.batch_extract as be
from src.config import Config
from src.utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Auto-loop batch extraction")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-batches", type=int, default=None, help="Stop after N batches")
    parser.add_argument("--pause", type=int, default=30, help="Seconds between submit and first poll")
    args = parser.parse_args()

    setup_logging(level="INFO")
    config = Config.load(args.config)

    batch_count = 0
    total_collected = 0

    while True:
        # Check remaining
        papers = be.get_unextracted_papers(config, limit=args.batch_size)
        if not papers:
            print("\nAll papers extracted!")
            break

        if args.max_batches and batch_count >= args.max_batches:
            print(f"\nReached max batches ({args.max_batches}). Stopping.")
            break

        # Create client and submit
        client = be.create_client(args.provider, config)
        text_getter = be.make_text_getter(config)

        print(f"\n{'='*60}")
        print(f"Batch {batch_count + 1}: submitting {len(papers)} papers...")
        print(f"{'='*60}")

        requests = client.create_batch_requests(papers, text_getter)
        if not requests:
            print("No requests created. Stopping.")
            break

        try:
            batch_id = client.submit_batch(requests)
        except Exception as exc:
            error_str = str(exc).lower()
            if any(p in error_str for p in (
                "billing", "hard_limit", "insufficient", "payment", "spending",
            )):
                print(f"\nBilling limit reached: {exc}")
                print(f"Batches completed: {batch_count}")
                print(f"Papers collected: {total_collected}")
                return
            raise
        print(f"Submitted: {batch_id}")

        # Wait for completion
        print(f"Waiting {args.pause}s before first poll...")
        time.sleep(args.pause)

        while True:
            status = client.get_batch_status(batch_id)
            print(
                f"  {status.completed_requests}/{status.total_requests} "
                f"({status.status})",
                flush=True,
            )

            if status.status == "completed":
                break
            elif status.status in ("failed", "expired", "cancelled"):
                print(f"Batch {status.status}! Checking error...")
                error_msg = ""
                try:
                    from openai import OpenAI

                    from src.utils.secrets import get_openai_api_key
                    oc = OpenAI(api_key=get_openai_api_key())
                    b = oc.batches.retrieve(batch_id)
                    print(f"Error: {b.errors}")
                    if b.errors and b.errors.data:
                        error_msg = str(b.errors.data[0].message).lower()
                except Exception:
                    pass

                # Exit on budget/billing errors
                budget_phrases = (
                    "billing",
                    "budget",
                    "insufficient",
                    "payment",
                    "spending limit",
                    "exceeded your current",
                )
                if any(p in error_msg for p in budget_phrases):
                    print("\nBudget or billing limit reached. Exiting gracefully.")
                    print(f"Batches completed: {batch_count}")
                    print(f"Papers collected: {total_collected}")
                    return

                # Token limit -- retry with smaller batch or wait
                if "token_limit" in error_msg or "enqueued" in error_msg:
                    print("Token enqueue limit -- waiting 5 min for queue to clear...")
                    time.sleep(300)
                    break

                print("Waiting 60s before retrying...")
                time.sleep(60)
                break

            time.sleep(60)  # Poll every 60s

        if status.status != "completed":
            continue  # Retry loop

        # Collect results
        import json
        from datetime import datetime

        from src.indexing.structured_store import safe_write_json

        index_dir = Path("data/index")
        sa_path = index_dir / "semantic_analyses.json"

        if sa_path.exists():
            sa_data = json.loads(sa_path.read_text(encoding="utf-8"))
        else:
            sa_data = {"schema_version": "1.0", "extractions": {}}

        extractions = sa_data.get("extractions", {})
        added = 0

        for result in client.get_results(batch_id):
            if result.success and result.extraction:
                extractions[result.paper_id] = {
                    "paper_id": result.paper_id,
                    "extraction": result.extraction.to_index_dict(),
                    "timestamp": datetime.now().isoformat(),
                    "model": result.model_used,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "batch_id": batch_id,
                }
                added += 1

        sa_data["extractions"] = extractions
        sa_data["extraction_count"] = len(extractions)
        sa_data["generated_at"] = datetime.now().isoformat()
        safe_write_json(sa_path, sa_data)

        batch_count += 1
        total_collected += added
        print(f"\nBatch {batch_count} complete: {added} papers collected")
        print(f"Running total: {total_collected} new + {len(extractions)} total extractions")

    print(f"\n{'='*60}")
    print(f"Loop complete: {batch_count} batches, {total_collected} papers extracted")
    print("\nGenerate embeddings: python scripts/build_index.py --skip-extraction")


if __name__ == "__main__":
    sys.exit(main() or 0)
