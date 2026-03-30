#!/usr/bin/env python
"""Deprecated compatibility wrapper for update-mode indexing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.indexing.orchestrator import IndexOrchestrator
from src.utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse legacy updater arguments and forward them to the orchestrator."""
    parser = argparse.ArgumentParser(
        description="Deprecated wrapper around build_index update mode",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only detect changes and print the resolved plan",
    )
    parser.add_argument(
        "--new-only",
        action="store_true",
        help="Only process new items, skip modified and deleted",
    )
    parser.add_argument(
        "--delete-only",
        action="store_true",
        help="Only process deleted items",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip extraction for compatible pending work",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embeddings and mark pending work",
    )
    parser.add_argument(
        "--mode",
        choices=["api", "cli"],
        default=None,
        help="Extraction mode override",
    )
    parser.add_argument(
        "--summary-model",
        type=str,
        default=None,
        help="Override model for summary extraction fields",
    )
    parser.add_argument(
        "--methodology-model",
        type=str,
        default=None,
        help="Override model for methodology extraction fields",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers to process after change detection",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def _augment_args(args: argparse.Namespace) -> argparse.Namespace:
    args.sync_mode = "update"
    args.explain_plan = bool(args.detect_only)
    args.dry_run = bool(args.detect_only)
    args.source = "zotero"
    args.source_path = None
    args.paper = []
    args.skip_paper = []
    args.rebuild_embeddings = False
    args.show_skipped = False
    args.show_failed = False
    args.reset_checkpoint = False
    args.classify_only = False
    args.reclassify = False
    args.index_all = False
    args.collection = None
    args.show_doi_overlap = False
    args.retry_failed = False
    args.resume = False
    args.parallel = None
    args.no_cache = False
    args.clear_cache = False
    args.use_subscription = False
    args.estimate_cost = False
    args.gap_fill = False
    args.gap_fill_threshold = 0.85
    args.gap_fill_provider = None
    args.provider = None
    args.model = None
    args.skip_similarity = False
    args.dedupe_by_doi = False
    return args


def main() -> int:
    """Entrypoint for the deprecated wrapper."""
    args = _augment_args(parse_args())
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)

    logger.warning(
        "scripts/update_index.py is deprecated; forwarding to build orchestrator in update mode",
    )

    try:
        config = Config.load(args.config)
    except Exception as exc:
        logger.error(f"Failed to load configuration: {exc}")
        return 1

    orchestrator = IndexOrchestrator(project_root=project_root, logger=logger)
    try:
        return orchestrator.run(args, config)
    except ValueError as exc:
        logger.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
