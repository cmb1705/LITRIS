#!/usr/bin/env python
"""Generate a heuristic gap analysis report from the indexed corpus."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.gap_detection import (  # noqa: E402
    GapDetectionConfig,
    load_gap_report,
    save_gap_report,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a gap analysis report from the indexed literature."
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=project_root / "data" / "index",
        help="Path to index directory (default: data/index)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "out" / "experiments" / "gap_analysis",
        help="Directory to save outputs (default: data/out/experiments/gap_analysis)",
    )
    parser.add_argument(
        "--output-format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        help="Filter analysis to specific collections",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=10,
        help="Maximum number of gap items per section (default: 10)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum count threshold for underrepresentation (default: 2)",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.2,
        help="Quantile threshold for underrepresentation (default: 0.2)",
    )
    parser.add_argument(
        "--include-abstracts",
        action="store_true",
        help="Include abstracts when estimating coverage",
    )
    parser.add_argument(
        "--sparse-year-max-count",
        type=int,
        default=1,
        help="Max papers to label a year as sparse (default: 1)",
    )
    parser.add_argument(
        "--future-min-mentions",
        type=int,
        default=1,
        help="Minimum mentions to include a future direction (default: 1)",
    )
    parser.add_argument(
        "--future-max-coverage",
        type=int,
        default=1,
        help="Max coverage papers to flag a future direction gap (default: 1)",
    )
    return parser.parse_args()


def main() -> int:
    """Run gap analysis and save report."""
    args = parse_args()

    if not args.index_dir.exists():
        print(f"Error: index directory not found: {args.index_dir}")
        return 1

    config = GapDetectionConfig(
        max_items=args.max_items,
        min_count=args.min_count,
        quantile=args.quantile,
        include_abstracts=args.include_abstracts,
        sparse_year_max_count=args.sparse_year_max_count,
        future_direction_min_mentions=args.future_min_mentions,
        future_direction_max_coverage=args.future_max_coverage,
    )

    report = load_gap_report(
        index_dir=args.index_dir,
        config=config,
        collections=args.collections,
    )
    output_path = save_gap_report(report, args.output_dir, args.output_format)
    print(f"Gap analysis saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
