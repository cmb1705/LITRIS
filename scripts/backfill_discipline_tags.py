#!/usr/bin/env python
"""Backfill missing discipline tags without full re-extraction.

This script identifies papers with empty discipline_tags and performs a minimal
extraction to fill them in, avoiding the cost and time of full re-extraction.

Usage:
    python scripts/backfill_discipline_tags.py --limit 100
    python scripts/backfill_discipline_tags.py --dry-run
    python scripts/backfill_discipline_tags.py --model haiku --limit 500
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.analysis.cli_executor import ClaudeCliExecutor

# Use print for user-facing output (logger may not be configured for console)
def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")

def log_warning(msg: str) -> None:
    print(f"[WARN] {msg}")


def extract_with_api(prompt: str, model: str) -> list[str] | None:
    """Extract discipline tags using Anthropic API (for model comparison).

    Args:
        prompt: The extraction prompt.
        model: Model name (haiku, sonnet, opus).

    Returns:
        List of discipline tags or None on failure.
    """
    try:
        from anthropic import Anthropic

        from src.utils.secrets import get_anthropic_api_key

        api_key = get_anthropic_api_key()
        if not api_key:
            log_warning("No API key found for compare mode")
            return None

        # Map short names to full model IDs
        model_map = {
            "haiku": "claude-3-5-haiku-20241022",
            "sonnet": "claude-sonnet-4-20250514",
            "opus": "claude-opus-4-5-20251101",
        }
        model_id = model_map.get(model, model)

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_id,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse response
        text = response.content[0].text
        data = json.loads(text)
        tags = data.get("discipline_tags", [])

        # Normalize
        normalized = []
        seen = set()
        for tag in tags:
            if isinstance(tag, str):
                clean = tag.lower().strip()
                if clean and clean not in seen:
                    normalized.append(clean)
                    seen.add(clean)
        return normalized if normalized else None
    except Exception as e:
        log_warning(f"API extraction failed: {e}")
        return None

# Minimal prompt for discipline extraction only
DISCIPLINE_EXTRACTION_PROMPT = """Analyze this academic paper and identify 2-5 academic disciplines it contributes to.

PAPER METADATA:
Title: {title}
Authors: {authors}
Year: {year}
Type: {item_type}

PAPER CONTENT (abbreviated):
{text}

Return ONLY a JSON object with discipline_tags. Use lowercase. Examples of valid disciplines:
scientometrics, bibliometrics, network science, science policy, information science,
machine learning, complex systems, science and technology studies, research evaluation,
computational social science, innovation studies, public administration, military studies,
defense policy, organizational behavior, philosophy of science, history of science,
sociology of science, economics of innovation, technology assessment, data science,
artificial intelligence, epidemiology, statistical physics, graph theory, computer science

{{
  "discipline_tags": ["discipline1", "discipline2", "discipline3"]
}}

Respond ONLY with valid JSON. No additional text."""


def load_extractions(extractions_path: Path) -> dict:
    """Load existing extractions."""
    with open(extractions_path, encoding="utf-8") as f:
        return json.load(f)


def load_papers(papers_path: Path) -> dict:
    """Load paper metadata."""
    with open(papers_path, encoding="utf-8") as f:
        data = json.load(f)
    # Build lookup by paper_id
    return {p["paper_id"]: p for p in data.get("papers", []) if p.get("paper_id")}


def find_papers_missing_tags(extractions: dict) -> list[str]:
    """Find paper IDs with missing discipline_tags."""
    missing = []
    for paper_id, paper_data in extractions.get("extractions", {}).items():
        ext = paper_data.get("extraction", {})
        tags = ext.get("discipline_tags", [])
        if not tags or len(tags) == 0:
            missing.append(paper_id)
    return missing


def get_paper_text(paper_id: str, papers_meta: dict) -> str | None:
    """Get paper text from PDF path."""
    meta = papers_meta.get(paper_id)
    if not meta:
        return None

    pdf_path = meta.get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        return None

    try:
        import pymupdf

        doc = pymupdf.open(pdf_path)
        text_parts = []
        # Only extract first 10 pages for speed
        for page_num in range(min(10, len(doc))):
            page = doc[page_num]
            text_parts.append(page.get_text())
        doc.close()

        text = "\n".join(text_parts)
        # Truncate to ~8000 chars for minimal prompt
        if len(text) > 8000:
            text = text[:8000] + "\n...[truncated]"
        return text
    except Exception as e:
        log_warning(f"Failed to read PDF {pdf_path}: {e}")
        return None


def extract_discipline_tags(
    executor: ClaudeCliExecutor,
    paper_id: str,
    papers_meta: dict,
) -> list[str] | None:
    """Extract discipline tags for a single paper."""
    meta = papers_meta.get(paper_id, {})
    text = get_paper_text(paper_id, papers_meta)

    if not text:
        return None

    prompt = DISCIPLINE_EXTRACTION_PROMPT.format(
        title=meta.get("title", "Unknown"),
        authors=meta.get("authors", "Unknown"),
        year=meta.get("year", "Unknown"),
        item_type=meta.get("item_type", "article"),
        text=text,
    )

    try:
        response = executor.call_with_prompt(prompt)
        # Parse JSON response
        data = json.loads(response)
        tags = data.get("discipline_tags", [])
        # Normalize
        normalized = []
        seen = set()
        for tag in tags:
            if isinstance(tag, str):
                clean = tag.lower().strip()
                if clean and clean not in seen:
                    normalized.append(clean)
                    seen.add(clean)
        return normalized if normalized else None
    except json.JSONDecodeError as e:
        log_warning(f"Failed to parse response for {paper_id}: {e}")
        return None
    except Exception as e:
        log_warning(f"Extraction failed for {paper_id}: {e}")
        return None


def update_extraction_store(
    extractions_path: Path,
    extractions: dict,
    updates: dict[str, list[str]],
) -> None:
    """Update extraction store with new discipline tags."""
    for paper_id, tags in updates.items():
        if paper_id in extractions.get("extractions", {}):
            extractions["extractions"][paper_id]["extraction"]["discipline_tags"] = tags

    # Update metadata
    extractions["generated_at"] = datetime.now().isoformat()

    # Write back
    with open(extractions_path, "w", encoding="utf-8") as f:
        json.dump(extractions, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Backfill missing discipline tags")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of papers to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--model",
        default="haiku",
        choices=["haiku", "sonnet", "opus"],
        help="Model for API mode (ignored for CLI; default: haiku)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/index"),
        help="Data directory",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare haiku vs sonnet on sample papers (no changes saved)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, max recommended: 8)",
    )
    args = parser.parse_args()

    extractions_path = args.data_dir / "extractions.json"
    papers_path = args.data_dir / "papers.json"

    if not extractions_path.exists():
        log_warning(f"Extractions file not found: {extractions_path}")
        sys.exit(1)

    if not papers_path.exists():
        log_warning(f"Papers file not found: {papers_path}")
        sys.exit(1)

    # Load data
    log_info("Loading extraction data...")
    extractions = load_extractions(extractions_path)
    papers_meta = load_papers(papers_path)

    # Find papers missing tags
    missing = find_papers_missing_tags(extractions)
    log_info(f"Found {len(missing)} papers with missing discipline_tags")

    if args.limit:
        missing = missing[: args.limit]
        log_info(f"Limited to {len(missing)} papers")

    if args.dry_run:
        log_info("DRY RUN - would process these papers:")
        for paper_id in missing[:20]:
            meta = papers_meta.get(paper_id, {})
            log_info(f"  - {meta.get('title', paper_id)[:60]}")
        if len(missing) > 20:
            log_info(f"  ... and {len(missing) - 20} more")
        return

    if args.compare:
        # Compare haiku vs sonnet on sample using API
        compare_limit = min(args.limit or 5, 5)  # Max 5 for compare (API costs)
        sample = missing[:compare_limit]
        log_info(f"COMPARE MODE: Running haiku vs sonnet on {len(sample)} papers")
        log_info("NOTE: This uses Anthropic API and will incur costs (~$0.01 per paper)")
        log_info("-" * 80)

        for paper_id in sample:
            meta = papers_meta.get(paper_id, {})
            title = meta.get("title", paper_id)[:55]
            log_info(f"\nPaper: {title}")

            # Build prompt
            text = get_paper_text(paper_id, papers_meta)
            if not text:
                log_info("  (could not read PDF)")
                continue

            prompt = DISCIPLINE_EXTRACTION_PROMPT.format(
                title=meta.get("title", "Unknown"),
                authors=meta.get("authors", "Unknown"),
                year=meta.get("year", "Unknown"),
                item_type=meta.get("item_type", "article"),
                text=text,
            )

            # Extract with haiku
            haiku_tags = extract_with_api(prompt, "haiku")
            time.sleep(0.5)

            # Extract with sonnet
            sonnet_tags = extract_with_api(prompt, "sonnet")
            time.sleep(0.5)

            # Display comparison
            haiku_str = ", ".join(haiku_tags) if haiku_tags else "(none)"
            sonnet_str = ", ".join(sonnet_tags) if sonnet_tags else "(none)"
            log_info(f"  Haiku:  {haiku_str}")
            log_info(f"  Sonnet: {sonnet_str}")

            # Check agreement
            if haiku_tags and sonnet_tags:
                overlap = set(haiku_tags) & set(sonnet_tags)
                if overlap:
                    log_info(f"  Agree:  {', '.join(overlap)}")
                else:
                    log_info("  Agree:  (no overlap)")

        log_info("-" * 80)
        log_info("Compare complete. No changes saved.")
        return

    # Process papers
    updates = {}
    failed = 0
    success = 0
    updates_lock = Lock()

    def process_paper(paper_id: str) -> tuple[str, list[str] | None]:
        """Process a single paper (thread-safe)."""
        # Each thread gets its own executor
        executor = ClaudeCliExecutor()
        tags = extract_discipline_tags(executor, paper_id, papers_meta)
        return paper_id, tags

    num_workers = max(1, min(args.parallel, 16))  # Cap at 16 workers
    log_info(f"Processing with {num_workers} parallel worker(s)...")

    if num_workers == 1:
        # Sequential mode
        with tqdm(missing, desc="Backfilling discipline tags") as pbar:
            for paper_id in pbar:
                _, tags = process_paper(paper_id)
                if tags:
                    updates[paper_id] = tags
                    success += 1
                else:
                    failed += 1
                pbar.set_postfix({"success": success, "failed": failed})
    else:
        # Parallel mode
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_paper, pid): pid for pid in missing}

            with tqdm(total=len(missing), desc="Backfilling discipline tags") as pbar:
                for future in as_completed(futures):
                    paper_id, tags = future.result()
                    with updates_lock:
                        if tags:
                            updates[paper_id] = tags
                            success += 1
                        else:
                            failed += 1
                        pbar.set_postfix({"success": success, "failed": failed})
                    pbar.update(1)

    # Update store
    if updates:
        log_info(f"Updating extraction store with {len(updates)} new discipline tags...")
        update_extraction_store(extractions_path, extractions, updates)
        log_info("Done!")
    else:
        log_info("No updates to apply")

    # Summary
    log_info(f"Summary: {success} success, {failed} failed out of {len(missing)} papers")


if __name__ == "__main__":
    main()
