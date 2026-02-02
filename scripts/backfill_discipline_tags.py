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
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.analysis.cli_executor import ClaudeCliExecutor
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

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
        logger.warning(f"Failed to read PDF {pdf_path}: {e}")
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
        logger.warning(f"Failed to parse response for {paper_id}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Extraction failed for {paper_id}: {e}")
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
        help="Model to use for extraction (default: haiku for speed)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/index"),
        help="Data directory",
    )
    args = parser.parse_args()

    extractions_path = args.data_dir / "extractions.json"
    papers_path = args.data_dir / "papers.json"

    if not extractions_path.exists():
        logger.error(f"Extractions file not found: {extractions_path}")
        sys.exit(1)

    if not papers_path.exists():
        logger.error(f"Papers file not found: {papers_path}")
        sys.exit(1)

    # Load data
    logger.info("Loading extraction data...")
    extractions = load_extractions(extractions_path)
    papers_meta = load_papers(papers_path)

    # Find papers missing tags
    missing = find_papers_missing_tags(extractions)
    logger.info(f"Found {len(missing)} papers with missing discipline_tags")

    if args.limit:
        missing = missing[: args.limit]
        logger.info(f"Limited to {len(missing)} papers")

    if args.dry_run:
        logger.info("DRY RUN - would process these papers:")
        for paper_id in missing[:20]:
            meta = papers_meta.get(paper_id, {})
            logger.info(f"  - {meta.get('title', paper_id)[:60]}")
        if len(missing) > 20:
            logger.info(f"  ... and {len(missing) - 20} more")
        return

    # Initialize executor
    executor = ClaudeCliExecutor(model=args.model)

    # Process papers
    updates = {}
    failed = 0
    success = 0

    with tqdm(missing, desc="Backfilling discipline tags") as pbar:
        for paper_id in pbar:
            tags = extract_discipline_tags(executor, paper_id, papers_meta)
            if tags:
                updates[paper_id] = tags
                success += 1
                pbar.set_postfix({"success": success, "failed": failed})
            else:
                failed += 1
                pbar.set_postfix({"success": success, "failed": failed})

            # Small delay between requests
            time.sleep(0.1)

    # Update store
    if updates:
        logger.info(f"Updating extraction store with {len(updates)} new discipline tags...")
        update_extraction_store(extractions_path, extractions, updates)
        logger.info("Done!")
    else:
        logger.info("No updates to apply")

    # Summary
    logger.info(f"Summary: {success} success, {failed} failed out of {len(missing)} papers")


if __name__ == "__main__":
    main()
