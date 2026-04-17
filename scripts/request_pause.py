#!/usr/bin/env python
"""Request or inspect a graceful pause for the active extraction run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.run_control import (
    clear_control_request,
    default_control_path,
    read_control_request,
    write_pause_request,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Request a graceful pause for the active LITRIS extraction run.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=project_root / "data" / "index",
        help="Index directory containing the cooperative run-control file.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear an existing pause request instead of writing one.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print the current request status and exit.",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default=None,
        help="Optional reason recorded with the pause request.",
    )
    return parser.parse_args()


def main() -> int:
    """Entrypoint for pause request management."""
    args = parse_args()
    control_path = default_control_path(args.index_dir)

    if args.status:
        request = read_control_request(control_path)
        if request is None:
            print(f"No run-control request at {control_path}")
        else:
            print(
                f"Run-control request at {control_path}: {request.action} ({request.requested_at})"
            )
            if request.reason:
                print(f"Reason: {request.reason}")
        return 0

    if args.clear:
        if clear_control_request(control_path):
            print(f"Cleared pause request at {control_path}")
        else:
            print(f"No pause request to clear at {control_path}")
        return 0

    request = write_pause_request(control_path, reason=args.reason)
    print(f"Requested graceful pause at {control_path} ({request.requested_at})")
    if request.reason:
        print(f"Reason: {request.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
