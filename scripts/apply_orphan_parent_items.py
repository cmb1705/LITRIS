#!/usr/bin/env python
"""Apply parent item creation from pre-computed orphan PDFs report.

Uses the orphan_pdfs_report_v4.csv with merged web search DOIs to create
parent items in Zotero database.

Usage:
    python scripts/apply_orphan_parent_items.py --dry-run
    python scripts/apply_orphan_parent_items.py --apply
"""

import argparse
import csv
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import get_logger
from src.zotero.metadata_enricher import EnrichedMetadata
from src.zotero.orphan_metadata_extractor import ExtractedMetadata, MetadataSource
from src.zotero.parent_item_creator import ParentItemCreator

logger = get_logger(__name__)


def load_config() -> dict:
    """Load configuration from config.yaml."""
    import yaml

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_report(csv_path: Path) -> list[dict]:
    """Load orphan PDFs report."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def row_to_enriched_metadata(row: dict) -> EnrichedMetadata:
    """Convert CSV row to EnrichedMetadata object."""
    # Parse authors
    authors = []
    if row.get("authors"):
        authors = [a.strip() for a in row["authors"].split(";") if a.strip()]

    # Parse year
    year = None
    if row.get("year"):
        try:
            year = int(row["year"])
        except ValueError:
            pass

    # Create ExtractedMetadata (minimal, for original field)
    extracted = ExtractedMetadata(
        title=row.get("title") or row.get("filename", "Untitled"),
        authors=authors,
        publication_year=year,
        doi=row.get("doi") if not row.get("enrichment_source") else None,
        source=MetadataSource.FILENAME_PARSED,
        confidence=float(row.get("extraction_confidence", 0.5)),
        attachment_key=row.get("attachment_key"),
    )

    # Get attachment item ID from database
    # (We'll set this during processing)

    # Create EnrichedMetadata
    enriched = EnrichedMetadata(original=extracted)
    enriched.title = row.get("title")
    enriched.authors = authors
    enriched.publication_year = year
    enriched.doi = row.get("doi") or None
    enriched.enrichment_source = row.get("enrichment_source", "none")
    enriched.enrichment_confidence = float(row.get("enrichment_confidence", 0.5))

    return enriched


def get_attachment_item_ids(db_path: str, keys: list[str]) -> dict[str, int]:
    """Get item IDs for attachment keys."""
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row

    placeholders = ",".join("?" * len(keys))
    cursor = conn.execute(
        f"SELECT key, itemID FROM items WHERE key IN ({placeholders})",
        keys,
    )

    result = {}
    for row in cursor:
        result[row["key"]] = row["itemID"]

    conn.close()
    return result


def create_parent_items(
    report: list[dict],
    db_path: str,
    collection_id: int,
    dry_run: bool = True,
) -> tuple[int, int]:
    """Create parent items for all orphan PDFs.

    Returns:
        Tuple of (success_count, fail_count)
    """
    # Get attachment item IDs
    keys = [row["attachment_key"] for row in report if row.get("attachment_key")]
    key_to_id = get_attachment_item_ids(db_path, keys)

    success_count = 0
    fail_count = 0

    with ParentItemCreator(db_path, dry_run=dry_run) as creator:
        for i, row in enumerate(report):
            attachment_key = row.get("attachment_key")
            if not attachment_key:
                fail_count += 1
                continue

            attachment_id = key_to_id.get(attachment_key)
            if not attachment_id:
                logger.warning(f"No item ID for key {attachment_key}")
                fail_count += 1
                continue

            # Convert row to EnrichedMetadata
            enriched = row_to_enriched_metadata(row)
            enriched.original.attachment_item_id = attachment_id
            enriched.original.attachment_key = attachment_key

            # Create parent item
            result = creator.create_parent_item(
                enriched,
                collection_id=collection_id,
                tag="auto-created",
            )

            if result.success:
                success_count += 1
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(report)} PDFs...")
            else:
                fail_count += 1
                logger.warning(f"Failed for {attachment_key}: {result.error}")

        if not dry_run:
            creator.commit()

    return success_count, fail_count


def print_summary(report: list[dict]) -> None:
    """Print summary of report data."""
    total = len(report)
    with_doi = sum(1 for r in report if r.get("doi", "").strip())
    with_title = sum(1 for r in report if r.get("title", "").strip())
    with_authors = sum(1 for r in report if r.get("authors", "").strip())
    with_year = sum(1 for r in report if r.get("year", "").strip())

    print("\n" + "=" * 60)
    print("REPORT SUMMARY")
    print("=" * 60)
    print(f"Total PDFs: {total}")
    print(f"With DOI: {with_doi} ({with_doi*100/total:.1f}%)")
    print(f"With title: {with_title} ({with_title*100/total:.1f}%)")
    print(f"With authors: {with_authors} ({with_authors*100/total:.1f}%)")
    print(f"With year: {with_year} ({with_year*100/total:.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Create parent items from orphan PDFs report"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without committing",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes to database",
    )
    parser.add_argument(
        "--collection-id",
        type=int,
        default=71,
        help="Zotero collection ID (default: 71 for HSD_PDFs-only)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to orphan PDFs report (default: orphan_pdfs_report_v4.csv)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create database backup before applying changes",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.dry_run and not args.apply:
        parser.print_help()
        print("\nError: Must specify --dry-run or --apply")
        sys.exit(1)

    if args.apply and args.dry_run:
        print("Error: Cannot use both --apply and --dry-run")
        sys.exit(1)

    # Load config
    config = load_config()
    db_path = config["zotero"]["database_path"]

    # Set up report path
    report_path = args.report or Path("data/query_results/orphan_pdfs_report_v4.csv")
    if not report_path.exists():
        print(f"Error: Report not found at {report_path}")
        sys.exit(1)

    # Create backup if requested
    if args.backup and args.apply:
        backup_path = Path(db_path).with_suffix(
            f".backup_{datetime.now():%Y%m%d_%H%M%S}.sqlite"
        )
        shutil.copy2(db_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        print(f"Created backup: {backup_path}")

    # Load report
    logger.info(f"Loading report from {report_path}...")
    report = load_report(report_path)
    logger.info(f"Loaded {len(report)} PDF records")

    # Print summary
    print_summary(report)

    # Process
    if args.dry_run:
        print("\n[DRY RUN] Previewing parent item creation...")
    else:
        print("\n[APPLY] Creating parent items in database...")

    success_count, fail_count = create_parent_items(
        report,
        db_path,
        args.collection_id,
        dry_run=args.dry_run,
    )

    print(f"\nParent items created: {success_count}")
    print(f"Failed: {fail_count}")

    if args.dry_run:
        print("\n[DRY RUN] No changes were committed. Use --apply to commit changes.")
    else:
        print("\nChanges have been committed to the database.")
        print("Open Zotero to verify the results.")


if __name__ == "__main__":
    main()
