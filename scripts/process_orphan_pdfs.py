#!/usr/bin/env python
"""Process orphan PDF attachments in Zotero.

Extracts metadata, enriches via APIs, and creates parent items.
Supports dry-run mode for preview before committing changes.

Usage:
    python scripts/process_orphan_pdfs.py --dry-run
    python scripts/process_orphan_pdfs.py --apply
    python scripts/process_orphan_pdfs.py --report-only
"""

import argparse
import csv
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import get_logger
from src.zotero.metadata_enricher import MetadataEnricher
from src.zotero.orphan_metadata_extractor import OrphanMetadataExtractor
from src.zotero.parent_item_creator import ParentItemCreator

logger = get_logger(__name__)


def load_config() -> dict:
    """Load configuration from config.yaml."""
    import yaml

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_orphan_pdfs(db_path: str, collection_id: int) -> list[dict]:
    """Get all orphan PDF attachments from a collection.

    Args:
        db_path: Path to Zotero database.
        collection_id: Collection ID to scan.

    Returns:
        List of orphan attachment records.
    """
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row

    cursor = conn.execute(
        """
        SELECT
            i.itemID,
            i.key,
            att.path
        FROM collectionItems ci
        JOIN items i ON ci.itemID = i.itemID
        JOIN itemAttachments att ON att.itemID = i.itemID
        WHERE ci.collectionID = ?
          AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
          AND att.contentType = 'application/pdf'
          AND att.parentItemID IS NULL
        ORDER BY i.itemID
        """,
        (collection_id,),
    )

    orphans = []
    for row in cursor:
        orphans.append({
            "item_id": row["itemID"],
            "key": row["key"],
            "path": row["path"],
        })

    conn.close()
    return orphans


def resolve_pdf_path(orphan: dict, storage_path: Path) -> Path | None:
    """Resolve the actual PDF file path.

    Args:
        orphan: Orphan record with path info.
        storage_path: Zotero storage directory.

    Returns:
        Resolved Path or None if not found.
    """
    path_str = orphan["path"]
    if path_str.startswith("storage:"):
        filename = path_str[8:]
        pdf_path = storage_path / orphan["key"] / filename
    else:
        pdf_path = Path(path_str)

    if pdf_path.exists():
        return pdf_path
    return None


def process_orphans(
    orphans: list[dict],
    storage_path: Path,
    extractor: OrphanMetadataExtractor,
    enricher: MetadataEnricher | None,
    use_llm: bool = False,
) -> list[dict]:
    """Process orphan PDFs to extract and enrich metadata.

    Args:
        orphans: List of orphan records.
        storage_path: Zotero storage directory.
        extractor: Metadata extractor.
        enricher: Metadata enricher (optional).
        use_llm: Whether to use LLM for extraction.

    Returns:
        List of processed records with metadata.
    """
    results = []
    total = len(orphans)

    for i, orphan in enumerate(orphans):
        pdf_path = resolve_pdf_path(orphan, storage_path)
        if pdf_path is None:
            results.append({
                "orphan": orphan,
                "pdf_path": None,
                "extracted": None,
                "enriched": None,
                "status": "file_not_found",
            })
            continue

        # Extract metadata
        extracted = extractor.extract_metadata(
            pdf_path,
            attachment_item_id=orphan["item_id"],
            attachment_key=orphan["key"],
        )

        # Use LLM for low-confidence extractions
        if use_llm and extracted.confidence < 0.5:
            extracted = extractor.extract_with_llm(pdf_path)
            extracted.attachment_item_id = orphan["item_id"]
            extracted.attachment_key = orphan["key"]
            extracted.pdf_path = pdf_path

        # Enrich metadata
        enriched = None
        if enricher:
            enriched = enricher.enrich(extracted)

        results.append({
            "orphan": orphan,
            "pdf_path": pdf_path,
            "extracted": extracted,
            "enriched": enriched,
            "status": "processed",
        })

        # Progress
        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{total} PDFs...")

    return results


def generate_report(results: list[dict], output_path: Path) -> None:
    """Generate a CSV report of processed PDFs.

    Args:
        results: Processing results.
        output_path: Path for CSV output.
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "attachment_key",
            "filename",
            "status",
            "extraction_source",
            "extraction_confidence",
            "enrichment_source",
            "enrichment_confidence",
            "doi",
            "title",
            "authors",
            "year",
        ])

        for result in results:
            orphan = result["orphan"]
            extracted = result["extracted"]
            enriched = result["enriched"]
            pdf_path = result["pdf_path"]

            filename = pdf_path.name if pdf_path else "NOT_FOUND"
            doi = ""
            title = ""
            authors = ""
            year = ""
            extraction_source = ""
            extraction_confidence = ""
            enrichment_source = ""
            enrichment_confidence = ""

            if extracted:
                extraction_source = extracted.source.value
                extraction_confidence = f"{extracted.confidence:.2f}"
                title = extracted.title or ""
                authors = "; ".join(extracted.authors) if extracted.authors else ""
                year = str(extracted.publication_year) if extracted.publication_year else ""
                doi = extracted.doi or ""

            if enriched:
                enrichment_source = enriched.enrichment_source or ""
                enrichment_confidence = f"{enriched.enrichment_confidence:.2f}"
                if enriched.best_doi:
                    doi = enriched.best_doi
                if enriched.best_title:
                    title = enriched.best_title
                if enriched.best_authors:
                    authors = "; ".join(enriched.best_authors)
                if enriched.best_year:
                    year = str(enriched.best_year)

            writer.writerow([
                orphan["key"],
                filename[:100],
                result["status"],
                extraction_source,
                extraction_confidence,
                enrichment_source,
                enrichment_confidence,
                doi,
                title[:200],
                authors[:200],
                year,
            ])

    logger.info(f"Report saved to {output_path}")


def print_summary(results: list[dict]) -> None:
    """Print summary of processing results."""
    total = len(results)
    by_status = {}
    by_extraction_source = {}
    by_enrichment_source = {}
    doi_count = 0

    for result in results:
        status = result["status"]
        by_status[status] = by_status.get(status, 0) + 1

        if result["extracted"]:
            source = result["extracted"].source.value
            by_extraction_source[source] = by_extraction_source.get(source, 0) + 1

        if result["enriched"]:
            source = result["enriched"].enrichment_source or "none"
            by_enrichment_source[source] = by_enrichment_source.get(source, 0) + 1
            if result["enriched"].best_doi:
                doi_count += 1

    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total PDFs: {total}")
    print()

    print("By Status:")
    for status, count in sorted(by_status.items()):
        print(f"  {status}: {count}")
    print()

    print("By Extraction Source:")
    for source, count in sorted(by_extraction_source.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count}")
    print()

    print("By Enrichment Source:")
    for source, count in sorted(by_enrichment_source.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count}")
    print()

    print(f"PDFs with DOI: {doi_count} ({doi_count * 100 / total:.1f}%)")
    print("=" * 60)


def create_parent_items(
    results: list[dict],
    db_path: str,
    collection_id: int,
    dry_run: bool = True,
) -> list[dict]:
    """Create parent items for processed PDFs.

    Args:
        results: Processing results.
        db_path: Path to Zotero database.
        collection_id: Collection ID.
        dry_run: If True, don't commit changes.

    Returns:
        List of creation results.
    """
    creation_results = []

    with ParentItemCreator(db_path, dry_run=dry_run) as creator:
        for result in results:
            if result["status"] != "processed":
                continue

            enriched = result["enriched"]
            if enriched is None:
                # Use extracted as base
                from src.zotero.metadata_enricher import EnrichedMetadata
                enriched = EnrichedMetadata(original=result["extracted"])

            created = creator.create_parent_item(
                enriched,
                collection_id=collection_id,
                tag="auto-created",
            )
            creation_results.append({
                "orphan": result["orphan"],
                "created": created,
            })

        if not dry_run:
            creator.commit()

    return creation_results


def main():
    parser = argparse.ArgumentParser(description="Process orphan PDF attachments in Zotero")
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
        "--report-only",
        action="store_true",
        help="Generate report without creating parent items",
    )
    parser.add_argument(
        "--collection-id",
        type=int,
        default=71,
        help="Zotero collection ID (default: 71 for HSD_PDFs-only)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for low-confidence extractions",
    )
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Skip API enrichment (faster)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create database backup before applying changes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for report (default: data/query_results/orphan_pdfs_report.csv)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.dry_run and not args.apply and not args.report_only:
        parser.print_help()
        print("\nError: Must specify --dry-run, --apply, or --report-only")
        sys.exit(1)

    if args.apply and args.dry_run:
        print("Error: Cannot use both --apply and --dry-run")
        sys.exit(1)

    # Load config
    config = load_config()
    db_path = config["zotero"]["database_path"]
    storage_path = Path(config["zotero"]["storage_path"])

    # Set up output path
    output_path = args.output or Path("data/query_results/orphan_pdfs_report.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create backup if requested
    if args.backup and args.apply:
        backup_path = Path(db_path).with_suffix(f".backup_{datetime.now():%Y%m%d_%H%M%S}.sqlite")
        shutil.copy2(db_path, backup_path)
        logger.info(f"Created backup: {backup_path}")

    # Get orphan PDFs
    logger.info(f"Scanning collection {args.collection_id} for orphan PDFs...")
    orphans = get_orphan_pdfs(db_path, args.collection_id)
    logger.info(f"Found {len(orphans)} orphan PDFs")

    # Initialize extractors
    extractor = OrphanMetadataExtractor()
    enricher = None if args.skip_enrichment else MetadataEnricher()

    # Process PDFs
    logger.info("Extracting metadata...")
    results = process_orphans(
        orphans,
        storage_path,
        extractor,
        enricher,
        use_llm=args.use_llm,
    )

    # Generate report
    generate_report(results, output_path)
    print_summary(results)

    if args.report_only:
        print(f"\nReport saved to: {output_path}")
        return

    # Create parent items
    if args.dry_run:
        print("\n[DRY RUN] Previewing parent item creation...")
    else:
        print("\n[APPLY] Creating parent items in database...")

    creation_results = create_parent_items(
        results,
        db_path,
        args.collection_id,
        dry_run=args.dry_run,
    )

    # Report creation results
    success_count = sum(1 for r in creation_results if r["created"].success)
    fail_count = len(creation_results) - success_count

    print(f"\nParent items created: {success_count}")
    print(f"Failed: {fail_count}")

    if args.dry_run:
        print("\n[DRY RUN] No changes were committed. Use --apply to commit changes.")
    else:
        print("\nChanges have been committed to the database.")
        print("Open Zotero to verify the results.")


if __name__ == "__main__":
    main()
