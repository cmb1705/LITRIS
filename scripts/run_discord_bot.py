#!/usr/bin/env python3
"""Run the LITRIS Discord bot.

Usage:
    # Set token via environment variable
    export DISCORD_BOT_TOKEN=your-token-here
    python scripts/run_discord_bot.py

    # Or pass token directly
    python scripts/run_discord_bot.py --token your-token-here
"""

import argparse
import logging
import sys

# Ensure project root is in path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from src.discord_bot.bot import run_bot


def main() -> None:
    """Parse arguments and start the Discord bot."""
    parser = argparse.ArgumentParser(description="Run the LITRIS Discord bot")
    parser.add_argument(
        "--token",
        help="Discord bot token (default: DISCORD_BOT_TOKEN env var)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    run_bot(token=args.token)


if __name__ == "__main__":
    main()
