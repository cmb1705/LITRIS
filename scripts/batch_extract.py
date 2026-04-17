#!/usr/bin/env python
"""Compatibility wrapper for index-scoped semantic batch extraction.

This entrypoint is retained for operators who still call ``batch_extract.py``.
New batch functionality lives in ``scripts/dimensions.py batch ...`` and the
shared implementation in ``src/indexing/semantic_jobs.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.indexing.semantic_jobs import (
    collect_semantic_batch,
    get_semantic_batch_status,
    list_pending_semantic_batches,
    plan_semantic_batch,
    semantic_batch_state_dir,
    submit_semantic_batch,
    wait_for_semantic_batch,
)
from src.utils.logging_config import setup_logging


def _emit_payload(payload: dict[str, Any]) -> None:
    print(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False).strip())


def _build_plan(args: argparse.Namespace, config: Config):
    provider = args.provider or config.extraction.provider
    if args.model:
        config.extraction.model = args.model
    return plan_semantic_batch(
        index_dir=args.index_dir,
        config=config,
        provider=provider,
        paper_ids=args.paper,
        exclude_paper_ids=args.skip_paper,
        limit=args.limit,
        include_existing=args.include_existing,
        allow_abstract_fallback=args.allow_abstract_fallback,
        live_text_fallback=args.live_text_fallback,
        profile_reference=args.dimension_profile,
    )


def cmd_estimate(args: argparse.Namespace, config: Config, logger) -> int:
    plan = _build_plan(args, config)
    _emit_payload(
        {
            "index_dir": str(args.index_dir),
            "provider": plan.provider,
            "model": plan.model,
            "profile_id": plan.profile.profile_id,
            "selected_count": len(plan.selected),
            "selected_paper_ids": [item.paper.paper_id for item in plan.selected],
            "skipped_existing": plan.skipped_existing,
            "skipped_missing_text": plan.skipped_missing_text,
            "estimated_cost": plan.estimated_cost,
        }
    )
    return 0


def cmd_submit(args: argparse.Namespace, config: Config, logger) -> int:
    plan = _build_plan(args, config)
    if args.dry_run:
        _emit_payload(plan.to_dict())
        return 0

    manifest = submit_semantic_batch(plan, config=config)
    _emit_payload(
        {
            "batch_id": manifest["batch_id"],
            "index_dir": str(args.index_dir),
            "provider": manifest["provider"],
            "model": manifest["model"],
            "profile_id": manifest["profile_snapshot"]["profile_id"],
            "paper_count": len(manifest["paper_ids"]),
            "state_dir": str(semantic_batch_state_dir(args.index_dir)),
        }
    )
    return 0


def cmd_status(args: argparse.Namespace, config: Config, logger) -> int:
    manifest, status = get_semantic_batch_status(
        index_dir=args.index_dir,
        batch_id=args.batch_id,
        config=config,
    )
    _emit_payload(
        {
            "batch_id": manifest["batch_id"],
            "provider": manifest["provider"],
            "model": manifest["model"],
            "status": status.status,
            "completed_requests": status.completed_requests,
            "total_requests": status.total_requests,
            "failed_requests": getattr(status, "failed_requests", 0),
        }
    )
    return 0


def cmd_wait(args: argparse.Namespace, config: Config, logger) -> int:
    manifest, status = wait_for_semantic_batch(
        index_dir=args.index_dir,
        batch_id=args.batch_id,
        config=config,
        poll_interval=args.poll_interval,
        progress_callback=lambda state: logger.info(
            "Batch %s: %s/%s (%s)",
            args.batch_id,
            state.completed_requests,
            state.total_requests,
            state.status,
        ),
    )
    _emit_payload(
        {
            "batch_id": manifest["batch_id"],
            "status": status.status,
            "completed_requests": status.completed_requests,
            "total_requests": status.total_requests,
            "failed_requests": getattr(status, "failed_requests", 0),
        }
    )
    return 0


def cmd_collect(args: argparse.Namespace, config: Config, logger) -> int:
    _emit_payload(
        collect_semantic_batch(
            index_dir=args.index_dir,
            batch_id=args.batch_id,
            config=config,
            logger_override=logger,
        )
    )
    return 0


def cmd_pending(args: argparse.Namespace, config: Config, logger) -> int:
    _emit_payload(
        {
            "index_dir": str(args.index_dir),
            "pending": list_pending_semantic_batches(
                index_dir=args.index_dir,
                config=config,
                provider=args.provider,
            ),
        }
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for semantic batch extraction",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=project_root / "data" / "index")
    parser.add_argument("--dimension-profile", type=str, default=None)
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "anthropic"],
        help="Batch API provider (default: openai)",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate = subparsers.add_parser("estimate", help="Estimate batch cost")
    estimate.add_argument("--paper", action="append", default=[])
    estimate.add_argument("--skip-paper", action="append", default=[])
    estimate.add_argument("--limit", type=int, default=None)
    estimate.add_argument("--include-existing", action="store_true")
    estimate.add_argument("--allow-abstract-fallback", action="store_true")
    estimate.add_argument("--live-text-fallback", action="store_true")

    submit = subparsers.add_parser("submit", help="Submit batch")
    submit.add_argument("--paper", action="append", default=[])
    submit.add_argument("--skip-paper", action="append", default=[])
    submit.add_argument("--limit", type=int, default=None)
    submit.add_argument("--include-existing", action="store_true")
    submit.add_argument("--allow-abstract-fallback", action="store_true")
    submit.add_argument("--live-text-fallback", action="store_true")
    submit.add_argument("--dry-run", action="store_true")

    status = subparsers.add_parser("status", help="Check batch status")
    status.add_argument("batch_id")

    wait = subparsers.add_parser("wait", help="Wait for completion")
    wait.add_argument("batch_id")
    wait.add_argument("--poll-interval", type=int, default=60)

    collect = subparsers.add_parser("collect", help="Collect results")
    collect.add_argument("batch_id")

    subparsers.add_parser("pending", help="List pending batches")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logging(level="DEBUG" if args.verbose else "INFO")
    config = Config.load(args.config)
    commands = {
        "estimate": cmd_estimate,
        "submit": cmd_submit,
        "status": cmd_status,
        "wait": cmd_wait,
        "collect": cmd_collect,
        "pending": cmd_pending,
    }
    return commands[args.command](args, config, logger)


if __name__ == "__main__":
    sys.exit(main() or 0)
