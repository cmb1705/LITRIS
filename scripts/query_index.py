#!/usr/bin/env python
"""Query the literature review index."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.query.retrieval import (
    OutputFormat,
    format_paper_detail,
    format_results,
    format_summary,
    save_results,
)
from src.query.search import SearchEngine
from src.utils.logging_config import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query the literature review index"
    )

    # Query options
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Natural language search query",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)",
    )

    # Filter options
    parser.add_argument(
        "--chunk-types",
        nargs="+",
        choices=["abstract", "thesis", "contribution", "methodology",
                 "findings", "claims", "limitations", "future_work", "full_summary"],
        help="Filter by chunk types",
    )
    parser.add_argument(
        "--year-min",
        type=int,
        help="Minimum publication year",
    )
    parser.add_argument(
        "--year-max",
        type=int,
        help="Maximum publication year",
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        help="Filter by collection names",
    )
    parser.add_argument(
        "--item-types",
        nargs="+",
        help="Filter by item types (journalArticle, book, etc.)",
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        choices=["json", "markdown", "brief"],
        default="brief",
        help="Output format (default: brief)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to file",
    )
    parser.add_argument(
        "--include-extraction",
        action="store_true",
        help="Include extraction data in output",
    )

    # Special commands
    parser.add_argument(
        "--paper",
        type=str,
        help="Get details for a specific paper ID",
    )
    parser.add_argument(
        "--similar",
        type=str,
        help="Find papers similar to given paper ID",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show index summary statistics",
    )
    parser.add_argument(
        "--list-collections",
        action="store_true",
        help="List all collections in the index",
    )

    # Config
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


def main():
    """Main entry point."""
    args = parse_args()

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
    chroma_dir = index_dir / "chroma"
    results_dir = project_root / "data" / "query_results"

    # Check index exists
    if not index_dir.exists():
        logger.error(f"Index directory not found: {index_dir}")
        logger.error("Run scripts/build_index.py first to create the index")
        return 1

    # Initialize search engine
    logger.info("Initializing search engine...")
    try:
        engine = SearchEngine(
            index_dir=index_dir,
            chroma_dir=chroma_dir,
            embedding_model=config.embeddings.model,
        )
    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}")
        return 1

    # Handle special commands
    if args.summary:
        summary = engine.get_summary()
        output = format_summary(summary)
        print(output)
        return 0

    if args.list_collections:
        collections = engine.get_collections()
        print("Collections in index:")
        for coll in sorted(collections):
            print(f"  - {coll}")
        return 0

    if args.paper:
        paper_data = engine.get_paper(args.paper)
        if not paper_data:
            logger.error(f"Paper not found: {args.paper}")
            return 1

        output = format_paper_detail(
            paper_data.get("paper", {}),
            paper_data.get("extraction"),
        )
        print(output)
        return 0

    if args.similar:
        logger.info(f"Finding papers similar to: {args.similar}")
        results = engine.search_similar_papers(
            paper_id=args.similar,
            top_k=args.top_k,
        )

        if not results:
            logger.warning("No similar papers found")
            return 0

        output = format_results(
            results,
            f"Similar to: {args.similar}",
            args.output,
            args.include_extraction,
        )
        print(output)

        if args.save:
            save_results(
                results,
                f"Similar to: {args.similar}",
                results_dir,
                args.output,
                "similar",
                args.include_extraction,
            )

        return 0

    # Regular search
    if not args.query:
        logger.error("No query specified. Use -q/--query or --help for options")
        return 1

    logger.info(f"Searching for: {args.query}")
    results = engine.search(
        query=args.query,
        top_k=args.top_k,
        chunk_types=args.chunk_types,
        year_min=args.year_min,
        year_max=args.year_max,
        collections=args.collections,
        item_types=args.item_types,
        include_extraction=args.include_extraction,
    )

    if not results:
        logger.warning("No results found")
        return 0

    # Format output
    output = format_results(
        results,
        args.query,
        args.output,
        args.include_extraction,
    )
    print(output)

    # Save if requested
    if args.save:
        filepath = save_results(
            results,
            args.query,
            results_dir,
            args.output,
            "results",
            args.include_extraction,
        )
        logger.info(f"Results saved to {filepath}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
