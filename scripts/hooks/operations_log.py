#!/usr/bin/env python3
"""Log tool operations to data/logs/operations.log."""

import os
import sys
from datetime import datetime

tool = os.environ.get("CLAUDE_TOOL_NAME", "")
path = os.environ.get("CLAUDE_FILE_PATH", "")

if path:
    log_dir = os.path.join("data", "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp} | {tool} | {path}\n"
    with open(os.path.join(log_dir, "operations.log"), "a") as f:
        f.write(entry)

sys.exit(0)
