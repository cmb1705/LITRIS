#!/usr/bin/env python
"""Import a folder of PDFs into a Zotero library.

Scans a folder for PDF files, extracts metadata (from PDF properties and
filenames), optionally enriches via CrossRef/Semantic Scholar, and writes
items to Zotero using the best available backend (Web API or SQLite).

Usage:
    python scripts/import_pdf_folder.py /path/to/pdfs
    python scripts/import_pdf_folder.py /path/to/pdfs --enrich
    python scripts/import_pdf_folder.py /path/to/pdfs --dry-run --collection "Imported"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.references.pdffolder_adapter import PDFFolderReferenceDB
from src.utils.logging_config import get_logger
from src.zotero.orphan_metadata_extractor import ExtractedMetadata, MetadataSource
from src.zotero.writer import PaperWriteRequest, ZoteroWriter

logger = get_logger(__name__)


def paper_to_write_request(
    paper,
    collection_name: str | None = None,
) -> PaperWriteRequest:
    """Convert a PaperMetadata from PDFFolderReferenceDB to a PaperWriteRequest.

    Args:
        paper: PaperMetadata from the PDF folder adapter.
        collection_name: Optional collection to add the item to.

    Returns:
        PaperWriteRequest ready for ZoteroWriter.
    """
    authors = [a.full_name for a in paper.authors]
    return PaperWriteRequest(
        title=paper.title or "Untitled",
        authors=authors,
        year=paper.publication_year,
        abstract=paper.abstract,
        item_type=paper.item_type if paper.item_type != "document" else "journalArticle",
        pdf_path=paper.pdf_path,
        collection_name=collection_name or (paper.collections[0] if paper.collections else None),
        tags=paper.tags + ["litris-imported"],
    )


def enrich_request(request: PaperWriteRequest) -> PaperWriteRequest:
    """Enrich a write request with metadata from external APIs.

    Args:
        request: Basic write request.

    Returns:
        Enriched write request with additional metadata.
    """
    from src.zotero.metadata_enricher import MetadataEnricher

    extracted = ExtractedMetadata(
        title=request.title if request.title != "Untitled" else None,
        authors=request.authors,
        publication_year=request.year,
        source=MetadataSource.FILENAME_PARSED,
        confidence=0.6,
        pdf_path=request.pdf_path,
    )

    enricher = MetadataEnricher()
    enriched = enricher.enrich(extracted)

    if enriched.enrichment_confidence > 0.5:
        return PaperWriteRequest.from_enriched(enriched)

    return request


def main():
    parser = argparse.ArgumentParser(
        description="Import a folder of PDFs into Zotero",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Path to folder containing PDF files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Scan subfolders recursively (default: True)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Do not scan subfolders",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Enrich metadata via CrossRef/Semantic Scholar APIs",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Zotero collection name to add imported items to",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be imported without writing",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: auto-detect)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of PDFs to import",
    )

    args = parser.parse_args()

    if not args.folder.exists():
        print(f"Folder not found: {args.folder}")
        return 1

    if not args.folder.is_dir():
        print(f"Not a directory: {args.folder}")
        return 1

    # Load config
    config = Config.load(config_path=args.config)

    # Scan folder
    adapter = PDFFolderReferenceDB(args.folder, recursive=args.recursive)
    paper_count = adapter.get_paper_count()
    print(f"Found {paper_count} PDFs in {args.folder}")

    if paper_count == 0:
        print("No PDFs found.")
        return 0

    # Convert to write requests
    requests = []
    for paper in adapter.get_all_papers():
        req = paper_to_write_request(paper, collection_name=args.collection)

        if args.enrich:
            req = enrich_request(req)

        requests.append(req)

        if args.limit and len(requests) >= args.limit:
            break

    print(f"Prepared {len(requests)} items for import")

    # Preview mode
    if args.dry_run:
        print("\n--- DRY RUN (no changes will be made) ---\n")
        for i, req in enumerate(requests, 1):
            authors = ", ".join(req.authors[:3])
            if len(req.authors) > 3:
                authors += " et al."
            year = f" ({req.year})" if req.year else ""
            print(f"  {i}. {req.title}{year}")
            if authors:
                print(f"     Authors: {authors}")
            if req.doi:
                print(f"     DOI: {req.doi}")
            if req.collection_name:
                print(f"     Collection: {req.collection_name}")
            print()
        return 0

    # Write to Zotero
    writer = ZoteroWriter(config.zotero)
    try:
        results = writer.write_batch(requests)
    finally:
        writer.close()

    # Report results
    succeeded = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"\nImport complete: {len(succeeded)} succeeded, {len(failed)} failed")

    if succeeded:
        print("\nSuccessfully imported:")
        for r in succeeded:
            print(f"  [{r.backend}] {r.item_key}: {r.title}")

    if failed:
        print("\nFailed:")
        for r in failed:
            print(f"  [{r.backend}] {r.title}: {r.error}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
