#!/usr/bin/env python
"""Export search results and literature review data in various formats."""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.indexing.embeddings import EmbeddingGenerator
from src.indexing.structured_store import StructuredStore
from src.indexing.vector_store import SearchResult, VectorStore
from src.utils.logging_config import setup_logging


def escape_bibtex(text: str) -> str:
    """Escape special BibTeX characters.

    Args:
        text: Text to escape.

    Returns:
        Escaped text safe for BibTeX.
    """
    if not text:
        return ""
    # Order matters - escape backslash first
    replacements = [
        ('\\', '\\textbackslash{}'),
        ('{', '\\{'),
        ('}', '\\}'),
        ('%', '\\%'),
        ('_', '\\_'),
        ('^', '\\^{}'),
        ('&', '\\&'),
        ('#', '\\#'),
        ('$', '\\$'),
        ('~', '\\textasciitilde{}'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def sanitize_csv_field(field: str) -> str:
    """Sanitize field for CSV export to prevent formula injection.

    Args:
        field: Field value to sanitize.

    Returns:
        Sanitized field safe for CSV.
    """
    if not field:
        return ""
    field = str(field)
    # Prefix potentially dangerous formulas with a single quote
    if field and field[0] in ('=', '+', '-', '@', '\t', '\r'):
        return "'" + field
    return field


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export search results and literature data"
    )

    subparsers = parser.add_subparsers(dest="command", help="Export command")

    # Search export
    search_parser = subparsers.add_parser("search", help="Export search results")
    search_parser.add_argument(
        "-q", "--query",
        type=str,
        required=True,
        help="Search query",
    )
    search_parser.add_argument(
        "-n", "--num-results",
        type=int,
        default=20,
        help="Number of results (default: 20)",
    )
    search_parser.add_argument(
        "-f", "--format",
        choices=["json", "csv", "bibtex", "markdown"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    search_parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file (default: stdout)",
    )
    search_parser.add_argument(
        "--year-min",
        type=int,
        help="Minimum publication year",
    )
    search_parser.add_argument(
        "--year-max",
        type=int,
        help="Maximum publication year",
    )

    # Full export
    full_parser = subparsers.add_parser("full", help="Export all papers")
    full_parser.add_argument(
        "-f", "--format",
        choices=["json", "csv", "bibtex"],
        default="json",
        help="Output format (default: json)",
    )
    full_parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output file",
    )
    full_parser.add_argument(
        "--include-extractions",
        action="store_true",
        help="Include extraction data",
    )

    # Summary export
    summary_parser = subparsers.add_parser("summary", help="Export summary stats")
    summary_parser.add_argument(
        "-f", "--format",
        choices=["json", "markdown"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    summary_parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file (default: stdout)",
    )

    # BibTeX export
    bibtex_parser = subparsers.add_parser("bibtex", help="Export to BibTeX")
    bibtex_parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output .bib file",
    )
    bibtex_parser.add_argument(
        "--papers",
        type=str,
        nargs="+",
        help="Specific paper IDs to export (default: all)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def format_bibtex_entry(paper: dict) -> str:
    """Format a paper as a BibTeX entry.

    Args:
        paper: Paper dictionary from structured store.

    Returns:
        BibTeX entry string.
    """
    # Determine entry type
    item_type = paper.get("item_type", "article")
    bibtex_type = {
        "journalArticle": "article",
        "book": "book",
        "bookSection": "inbook",
        "conferencePaper": "inproceedings",
        "thesis": "phdthesis",
        "report": "techreport",
    }.get(item_type, "misc")

    # Generate citation key
    authors = paper.get("authors", [])
    first_author = authors[0] if authors else {}
    author_name = first_author.get("last_name", "Unknown")
    year = paper.get("publication_year", "n.d.")
    cite_key = f"{author_name}{year}"
    # Clean key
    cite_key = "".join(c for c in cite_key if c.isalnum())

    lines = [f"@{bibtex_type}{{{cite_key},"]

    # Title (escape special characters)
    title = escape_bibtex(paper.get("title", "Unknown Title"))
    lines.append(f'  title = {{{title}}},')

    # Authors (escape special characters)
    if authors:
        author_strs = []
        for a in authors:
            first = escape_bibtex(a.get("first_name", ""))
            last = escape_bibtex(a.get("last_name", ""))
            if first and last:
                author_strs.append(f"{last}, {first}")
            elif last:
                author_strs.append(last)
        if author_strs:
            lines.append(f'  author = {{{" and ".join(author_strs)}}},')

    # Year
    if year and year != "n.d.":
        lines.append(f'  year = {{{year}}},')

    # Journal/Booktitle (escape special characters)
    journal = paper.get("journal")
    if journal:
        journal = escape_bibtex(journal)
        if bibtex_type == "inproceedings":
            lines.append(f'  booktitle = {{{journal}}},')
        else:
            lines.append(f'  journal = {{{journal}}},')

    # Volume/Issue/Pages
    if volume := paper.get("volume"):
        lines.append(f'  volume = {{{volume}}},')
    if issue := paper.get("issue"):
        lines.append(f'  number = {{{issue}}},')
    if pages := paper.get("pages"):
        lines.append(f'  pages = {{{pages}}},')

    # DOI
    if doi := paper.get("doi"):
        lines.append(f'  doi = {{{doi}}},')

    # Abstract (escape special characters)
    if abstract := paper.get("abstract"):
        abstract = escape_bibtex(abstract)
        lines.append(f'  abstract = {{{abstract}}},')

    lines.append("}")

    return "\n".join(lines)


def format_search_result_markdown(
    result: SearchResult,
    paper: dict | None,
    extraction: dict | None,
) -> str:
    """Format a search result as markdown.

    Args:
        result: Search result.
        paper: Paper metadata.
        extraction: Extraction data.

    Returns:
        Markdown formatted string.
    """
    lines = []

    # Title and basic info
    title = paper.get("title", "Unknown Title") if paper else result.metadata.get("title", "Unknown")
    author = paper.get("author_string", "") if paper else ""
    year = paper.get("publication_year", "") if paper else result.metadata.get("year", "")

    lines.append(f"### {title}")
    if author:
        lines.append(f"**Authors:** {author}")
    if year:
        lines.append(f"**Year:** {year}")
    lines.append(f"**Relevance Score:** {result.score:.3f}")
    lines.append("")

    # Matched content
    lines.append(f"> {result.text[:500]}..." if len(result.text) > 500 else f"> {result.text}")
    lines.append("")

    # Key extraction info
    if extraction:
        ext = extraction.get("extraction", extraction)
        if thesis := ext.get("thesis_statement"):
            lines.append(f"**Thesis:** {thesis}")
        if conclusions := ext.get("conclusions"):
            lines.append(f"**Conclusions:** {conclusions}")
        if keywords := ext.get("keywords"):
            lines.append(f"**Keywords:** {', '.join(keywords)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def export_search_results(
    query: str,
    num_results: int,
    output_format: str,
    output_path: Path | None,
    store: StructuredStore,
    vector_store: VectorStore,
    embedding_gen: EmbeddingGenerator,
    year_min: int | None = None,
    year_max: int | None = None,
):
    """Export search results in specified format.

    Args:
        query: Search query.
        num_results: Number of results.
        output_format: Output format.
        output_path: Output file path.
        store: Structured store.
        vector_store: Vector store.
        embedding_gen: Embedding generator.
        year_min: Minimum year filter.
        year_max: Maximum year filter.
    """
    # Perform search
    results = vector_store.search_by_text(
        query,
        embedding_gen,
        top_k=num_results,
        year_min=year_min,
        year_max=year_max,
    )

    if not results:
        print("No results found.")
        return

    # Collect paper data
    papers_data = []
    for result in results:
        paper = store.get_paper(result.paper_id)
        extraction = store.get_extraction(result.paper_id)
        papers_data.append({
            "result": result,
            "paper": paper,
            "extraction": extraction,
        })

    # Format output
    if output_format == "json":
        output = json.dumps(
            [
                {
                    "paper_id": d["result"].paper_id,
                    "score": d["result"].score,
                    "matched_text": d["result"].text,
                    "paper": d["paper"],
                    "extraction": d["extraction"],
                }
                for d in papers_data
            ],
            indent=2,
            default=str,
        )
    elif output_format == "csv":
        rows = []
        for d in papers_data:
            paper = d["paper"] or {}
            rows.append({
                "paper_id": sanitize_csv_field(d["result"].paper_id),
                "score": d["result"].score,
                "title": sanitize_csv_field(paper.get("title", "")),
                "authors": sanitize_csv_field(paper.get("author_string", "")),
                "year": paper.get("publication_year", ""),
                "journal": sanitize_csv_field(paper.get("journal", "")),
                "doi": sanitize_csv_field(paper.get("doi", "")),
            })

        from io import StringIO
        buffer = StringIO()
        if rows:
            writer = csv.DictWriter(buffer, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        output = buffer.getvalue()
    elif output_format == "bibtex":
        entries = []
        for d in papers_data:
            if d["paper"]:
                entries.append(format_bibtex_entry(d["paper"]))
        output = "\n\n".join(entries)
    else:  # markdown
        lines = [
            f"# Search Results: \"{query}\"",
            "",
            f"*{len(results)} results found*",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
            "---",
            "",
        ]
        for d in papers_data:
            lines.append(format_search_result_markdown(
                d["result"], d["paper"], d["extraction"]
            ))
        output = "\n".join(lines)

    # Write output
    if output_path:
        output_path.write_text(output, encoding="utf-8")
        print(f"Exported {len(results)} results to {output_path}")
    else:
        print(output)


def export_all_papers(
    output_format: str,
    output_path: Path,
    store: StructuredStore,
    include_extractions: bool = False,
):
    """Export all papers in specified format.

    Args:
        output_format: Output format.
        output_path: Output file path.
        store: Structured store.
        include_extractions: Include extraction data.
    """
    papers = store.load_papers()

    if output_format == "json":
        if include_extractions:
            extractions = store.load_extractions()
            data = {
                "papers": list(papers.values()),
                "extractions": extractions,
                "exported_at": datetime.now().isoformat(),
            }
        else:
            data = {
                "papers": list(papers.values()),
                "exported_at": datetime.now().isoformat(),
            }
        output_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    elif output_format == "csv":
        rows = []
        for paper in papers.values():
            rows.append({
                "paper_id": sanitize_csv_field(paper.get("paper_id", "")),
                "zotero_key": sanitize_csv_field(paper.get("zotero_key", "")),
                "title": sanitize_csv_field(paper.get("title", "")),
                "authors": sanitize_csv_field(paper.get("author_string", "")),
                "year": paper.get("publication_year", ""),
                "journal": sanitize_csv_field(paper.get("journal", "")),
                "doi": sanitize_csv_field(paper.get("doi", "")),
                "item_type": sanitize_csv_field(paper.get("item_type", "")),
                "collections": sanitize_csv_field("; ".join(paper.get("collections", []))),
                "tags": sanitize_csv_field("; ".join(paper.get("tags", []))),
            })

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    elif output_format == "bibtex":
        entries = [format_bibtex_entry(paper) for paper in papers.values()]
        output_path.write_text("\n\n".join(entries), encoding="utf-8")

    print(f"Exported {len(papers)} papers to {output_path}")


def export_summary(
    output_format: str,
    output_path: Path | None,
    store: StructuredStore,
):
    """Export summary statistics.

    Args:
        output_format: Output format.
        output_path: Output file path.
        store: Structured store.
    """
    summary = store.generate_summary()

    if output_format == "json":
        output = json.dumps(summary, indent=2, default=str)
    else:  # markdown
        lines = [
            "# Literature Review Index Summary",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
            "## Overview",
            "",
            f"- **Total Papers:** {summary.get('total_papers', 0)}",
            f"- **Total Extractions:** {summary.get('total_extractions', 0)}",
            "",
            "## Papers by Type",
            "",
        ]

        for item_type, count in summary.get("papers_by_type", {}).items():
            lines.append(f"- {item_type}: {count}")

        lines.extend([
            "",
            "## Papers by Year",
            "",
        ])

        for year, count in sorted(summary.get("papers_by_year", {}).items()):
            lines.append(f"- {year}: {count}")

        lines.extend([
            "",
            "## Top Collections",
            "",
        ])

        for coll, count in summary.get("papers_by_collection", {}).items():
            lines.append(f"- {coll}: {count}")

        if disciplines := summary.get("top_disciplines"):
            lines.extend([
                "",
                "## Top Disciplines",
                "",
            ])
            for disc, count in disciplines.items():
                lines.append(f"- {disc}: {count}")

        if recent := summary.get("recent_papers"):
            lines.extend([
                "",
                "## Recently Added",
                "",
            ])
            for paper in recent:
                lines.append(f"- {paper.get('title', 'Unknown')} ({paper.get('year', 'n.d.')})")

        output = "\n".join(lines)

    if output_path:
        output_path.write_text(output, encoding="utf-8")
        print(f"Exported summary to {output_path}")
    else:
        print(output)


def export_bibtex(
    output_path: Path,
    store: StructuredStore,
    paper_ids: list[str] | None = None,
):
    """Export papers to BibTeX format.

    Args:
        output_path: Output .bib file path.
        store: Structured store.
        paper_ids: Specific paper IDs to export.
    """
    papers = store.load_papers()

    if paper_ids:
        papers = {pid: papers[pid] for pid in paper_ids if pid in papers}

    entries = [format_bibtex_entry(paper) for paper in papers.values()]
    output_path.write_text("\n\n".join(entries), encoding="utf-8")
    print(f"Exported {len(entries)} entries to {output_path}")


def main():
    """Main entry point."""
    args = parse_args()

    if not args.command:
        print("Please specify a command: search, full, summary, or bibtex")
        print("Use --help for more information")
        return 1

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)

    # Load configuration
    try:
        config = Config.load(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Setup paths
    index_dir = project_root / "data" / "index"
    if not index_dir.exists():
        logger.error(
            f"Index directory not found: {index_dir}\n"
            "Please run build_index.py first."
        )
        return 1

    # Initialize stores
    store = StructuredStore(index_dir)

    if args.command == "search":
        # Need vector store and embeddings for search
        chroma_dir = index_dir / "chroma"
        embedding_gen = EmbeddingGenerator(model_name=config.embeddings.model)
        vector_store = VectorStore(chroma_dir)

        export_search_results(
            query=args.query,
            num_results=args.num_results,
            output_format=args.format,
            output_path=args.output,
            store=store,
            vector_store=vector_store,
            embedding_gen=embedding_gen,
            year_min=args.year_min,
            year_max=args.year_max,
        )

    elif args.command == "full":
        export_all_papers(
            output_format=args.format,
            output_path=args.output,
            store=store,
            include_extractions=args.include_extractions,
        )

    elif args.command == "summary":
        export_summary(
            output_format=args.format,
            output_path=args.output,
            store=store,
        )

    elif args.command == "bibtex":
        export_bibtex(
            output_path=args.output,
            store=store,
            paper_ids=args.papers,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
