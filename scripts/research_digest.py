#!/usr/bin/env python
"""Generate a research digest from newly indexed papers.

Usage:
  python scripts/research_digest.py
  python scripts/research_digest.py --max-papers 20 --output-format json
  python scripts/research_digest.py --dry-run  # Don't mark papers as processed
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.research_digest import (  # noqa: E402
    DigestConfig,
    format_digest_json,
    format_digest_markdown,
    generate_digest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a research digest from newly indexed papers.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=project_root / "data" / "index",
        help="Path to the index directory (default: data/index)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "out" / "digests",
        help="Directory to save digest output (default: data/out/digests)",
    )
    parser.add_argument(
        "--output-format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=10,
        help="Maximum papers to include (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate digest without marking papers as processed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.index_dir.exists():
        print(f"Error: Index directory not found: {args.index_dir}")
        return 1

    config = DigestConfig(max_papers=args.max_papers)

    if args.verbose:
        print(f"Generating digest from {args.index_dir}...")

    digest = generate_digest(
        args.index_dir,
        config=config,
        mark_processed=not args.dry_run,
    )

    if args.verbose:
        print(f"Found {digest.new_paper_count} new papers")
        print(f"Included {len(digest.highlights)} highlights")

    if not digest.highlights:
        print("No new papers to report.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_format == "markdown":
        content = format_digest_markdown(digest)
        output_file = args.output_dir / "digest.md"
    else:
        content = format_digest_json(digest)
        output_file = args.output_dir / "digest.json"

    with open(output_file, "w") as f:
        f.write(content)

    print(f"Digest saved to: {output_file}")
    if args.dry_run:
        print("(dry run — papers not marked as processed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
