#!/usr/bin/env python3
"""Run ruff check and format on modified Python files."""
import os
import subprocess
import sys

path = os.environ.get("CLAUDE_FILE_PATH", "")
if not path or not path.endswith(".py") or not os.path.exists(path):
    sys.exit(0)

# Auto-fix fixable violations (suppress output)
subprocess.run(["ruff", "check", "--fix", path], capture_output=True)

# Auto-format (suppress output)
subprocess.run(["ruff", "format", path], capture_output=True)

# Report remaining violations -- output visible, exit code signals pass/fail
result = subprocess.run(["ruff", "check", path])
sys.exit(result.returncode)
