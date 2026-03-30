#!/usr/bin/env python
"""Maintenance wrapper for RAPTOR and similarity refresh."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.indexing.orchestrator import IndexOrchestrator
from src.utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh RAPTOR summaries and similarity pairs",
    )
    parser.add_argument(
        "--mode",
        choices=["template", "llm"],
        default="template",
        help="Generation mode: template (default) or llm",
    )
    parser.add_argument(
        "--provider",
        default="google",
        help="LLM provider for llm mode (default: google)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model for llm mode",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if cached",
    )
    parser.add_argument(
        "--skip-similarity",
        action="store_true",
        help="Refresh RAPTOR only and leave similarity pending",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)

    try:
        config = Config.load(args.config)
    except Exception as exc:
        logger.error(f"Failed to load configuration: {exc}")
        return 1

    orchestrator = IndexOrchestrator(project_root=PROJECT_ROOT, logger=logger)
    return orchestrator.refresh_derived_artifacts(
        config=config,
        mode=args.mode,
        provider=args.provider,
        model=args.model,
        force=args.force,
        skip_similarity=args.skip_similarity,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
