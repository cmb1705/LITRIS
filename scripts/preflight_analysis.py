#!/usr/bin/env python
"""Pre-flight analysis for literature index builds.

Analyzes Zotero database and existing index to show what will be processed,
estimate costs, and identify potential issues before running the build.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.utils.deduplication import analyze_doi_overlap, normalize_doi
from src.utils.file_utils import safe_read_json
from src.zotero.database import ZoteroDatabase


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze what will be processed before running build_index.py"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=["api", "cli", "batch_api"],
        default="cli",
        help="Extraction mode for cost estimation (default: cli)",
    )
    parser.add_argument(
        "--show-duplicates",
        action="store_true",
        help="Show detailed list of duplicate papers",
    )
    parser.add_argument(
        "--show-new",
        action="store_true",
        help="Show detailed list of genuinely new papers",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        help="Export analysis to CSV file",
    )
    return parser.parse_args()


def estimate_batch_cost(num_papers: int, avg_text_length: int = 20000) -> dict:
    """Estimate Batch API cost.

    Args:
        num_papers: Number of papers to process.
        avg_text_length: Average text length per paper in chars.

    Returns:
        Cost estimate dictionary.
    """
    # Rough estimate: 4 chars per token
    input_tokens_per_paper = avg_text_length // 4 + 500  # +500 for prompt
    output_tokens_per_paper = 2000  # Extraction output

    total_input = input_tokens_per_paper * num_papers
    total_output = output_tokens_per_paper * num_papers

    # Batch API pricing (50% discount)
    # Claude Opus: $15/MTok input, $75/MTok output (regular)
    # With 50% batch discount: $7.50/MTok input, $37.50/MTok output
    input_cost_per_million = 7.50
    output_cost_per_million = 37.50

    input_cost = (total_input / 1_000_000) * input_cost_per_million
    output_cost = (total_output / 1_000_000) * output_cost_per_million

    return {
        "num_papers": num_papers,
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "input_cost": round(input_cost, 2),
        "output_cost": round(output_cost, 2),
        "total_cost": round(input_cost + output_cost, 2),
        "cost_per_paper": round((input_cost + output_cost) / max(num_papers, 1), 4),
    }


def load_existing_index_stats(index_dir: Path) -> dict:
    """Load statistics from existing index."""
    stats = {
        "exists": False,
        "paper_count": 0,
        "extraction_count": 0,
        "papers_with_doi": 0,
        "last_updated": None,
    }

    papers_file = index_dir / "papers.json"
    if papers_file.exists():
        stats["exists"] = True
        data = safe_read_json(papers_file, default={})

        if isinstance(data, dict) and "papers" in data:
            papers = data["papers"]
            stats["paper_count"] = len(papers)
            stats["papers_with_doi"] = sum(
                1 for p in papers if normalize_doi(p.get("doi"))
            )
            stats["last_updated"] = data.get("generated_at")

    extractions_file = index_dir / "extractions.json"
    if extractions_file.exists():
        data = safe_read_json(extractions_file, default={})
        if isinstance(data, dict) and "extractions" in data:
            stats["extraction_count"] = len(data["extractions"])
        elif isinstance(data, dict):
            stats["extraction_count"] = len(data)

    return stats


def export_to_csv(
    filepath: Path,
    new_papers: list,
    duplicates: list,
    without_pdf: list,
):
    """Export analysis to CSV file."""
    import csv

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "paper_id", "title", "authors", "year", "doi", "status", "has_pdf"
        ])

        for paper in new_papers:
            writer.writerow([
                paper.paper_id,
                paper.title,
                paper.author_string,
                paper.publication_year,
                paper.doi or "",
                "new",
                "yes" if paper.pdf_path else "no",
            ])

        for paper in duplicates:
            writer.writerow([
                paper.paper_id,
                paper.title,
                paper.author_string,
                paper.publication_year,
                paper.doi or "",
                "duplicate",
                "yes" if paper.pdf_path else "no",
            ])

        for paper in without_pdf:
            writer.writerow([
                paper.paper_id,
                paper.title,
                paper.author_string,
                paper.publication_year,
                paper.doi or "",
                "no_pdf",
                "no",
            ])

    print(f"\nExported analysis to: {filepath}")


def main():
    """Main entry point."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("LITRIS Pre-flight Analysis")
    print("=" * 70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load configuration
    try:
        config = Config.load(args.config)
    except Exception as e:
        print(f"\nError loading configuration: {e}")
        return 1

    # Setup paths
    index_dir = project_root / "data" / "index"

    # Connect to Zotero
    print("\n[1/4] Connecting to Zotero database...")
    try:
        zotero = ZoteroDatabase(
            config.get_zotero_db_path(),
            config.get_storage_path(),
        )
        print(f"  Database: {config.get_zotero_db_path()}")
        print(f"  Storage: {config.get_storage_path()}")
    except Exception as e:
        print(f"  Error: {e}")
        return 1

    # Get papers from Zotero
    print("\n[2/4] Loading papers from Zotero...")
    all_papers = list(zotero.get_all_papers())
    papers_with_pdf = [p for p in all_papers if p.pdf_path]
    papers_without_pdf = [p for p in all_papers if not p.pdf_path]

    print(f"  Total items: {len(all_papers)}")
    print(f"  With PDFs: {len(papers_with_pdf)}")
    print(f"  Without PDFs: {len(papers_without_pdf)}")

    # DOI coverage
    with_doi = [p for p in papers_with_pdf if normalize_doi(p.doi)]
    without_doi = [p for p in papers_with_pdf if not normalize_doi(p.doi)]
    print(f"  With DOIs: {len(with_doi)}")
    print(f"  Without DOIs: {len(without_doi)}")

    # Load existing index stats
    print("\n[3/4] Analyzing existing index...")
    existing_stats = load_existing_index_stats(index_dir)

    if existing_stats["exists"]:
        print(f"  Papers indexed: {existing_stats['paper_count']}")
        print(f"  Extractions: {existing_stats['extraction_count']}")
        print(f"  With DOIs: {existing_stats['papers_with_doi']}")
        if existing_stats["last_updated"]:
            print(f"  Last updated: {existing_stats['last_updated']}")
    else:
        print("  No existing index found (will create new)")

    # DOI overlap analysis
    print("\n[4/4] Analyzing DOI overlap...")
    analysis = analyze_doi_overlap(papers_with_pdf, index_dir)

    duplicate_papers = analysis.get("duplicate_papers", [])
    new_papers = analysis.get("new_papers_filtered", [])

    print(f"  Duplicates by DOI: {analysis['duplicates_by_doi']}")
    print(f"  Genuinely new: {len(new_papers)}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nPapers to process (with --dedupe-by-doi): {len(new_papers)}")
    print(f"Papers to skip (DOI duplicates): {len(duplicate_papers)}")
    print(f"Papers skipped (no PDF): {len(papers_without_pdf)}")

    # Cost estimation
    if new_papers:
        print("\n" + "-" * 70)
        print("COST ESTIMATION (Batch API)")
        print("-" * 70)

        estimate = estimate_batch_cost(len(new_papers))
        print(f"\n  Papers to extract: {estimate['num_papers']}")
        print(f"  Est. input tokens: {estimate['estimated_input_tokens']:,}")
        print(f"  Est. output tokens: {estimate['estimated_output_tokens']:,}")
        print(f"  Input cost: ${estimate['input_cost']:.2f}")
        print(f"  Output cost: ${estimate['output_cost']:.2f}")
        print(f"  Total cost: ${estimate['total_cost']:.2f}")
        print(f"  Cost per paper: ${estimate['cost_per_paper']:.4f}")

        print("\n  Note: Batch API offers 50% discount, processed within 24 hours")

    # Show duplicates if requested
    if args.show_duplicates and duplicate_papers:
        print("\n" + "-" * 70)
        print(f"DUPLICATE PAPERS ({len(duplicate_papers)})")
        print("-" * 70)
        for i, p in enumerate(duplicate_papers, 1):
            print(f"\n  {i}. {p.title[:60]}...")
            print(f"     Authors: {p.author_string}")
            print(f"     DOI: {p.doi}")
            print(f"     Year: {p.publication_year}")

    # Show new papers if requested
    if args.show_new and new_papers:
        print("\n" + "-" * 70)
        print(f"GENUINELY NEW PAPERS ({len(new_papers)})")
        print("-" * 70)
        for i, p in enumerate(new_papers[:50], 1):  # Limit to 50
            print(f"\n  {i}. {p.title[:60]}...")
            print(f"     Authors: {p.author_string}")
            print(f"     DOI: {p.doi or 'N/A'}")
        if len(new_papers) > 50:
            print(f"\n  ... and {len(new_papers) - 50} more")

    # Export if requested
    if args.export_csv:
        export_to_csv(
            args.export_csv,
            new_papers,
            duplicate_papers,
            papers_without_pdf,
        )

    # Recommended command
    print("\n" + "=" * 70)
    print("RECOMMENDED COMMANDS")
    print("=" * 70)

    if new_papers:
        print("\n  # Dry run to verify:")
        print(f"  python scripts/build_index.py --dedupe-by-doi --dry-run")

        print("\n  # For Batch API extraction:")
        print(f"  python scripts/batch_extract.py submit --dedupe-by-doi")

        print("\n  # For CLI extraction (uses subscription):")
        print(f"  python scripts/build_index.py --dedupe-by-doi --mode cli")
    else:
        print("\n  No new papers to process. Index is up to date.")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
