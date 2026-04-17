#!/usr/bin/env python3
"""Lint markdown files using markdownlint-cli2."""

import os
import re
import subprocess
import sys

path = os.environ.get("CLAUDE_FILE_PATH", "")
if not path or not path.endswith(".md"):
    sys.exit(0)

# Skip config-like markdown files
if re.search(r"CLAUDE\.md$|CLAUDE_SUPPLEMENTAL\.md$|ralph-loop\.local\.md$", path):
    sys.exit(0)

subprocess.run(["npx", "--yes", "markdownlint-cli2", path], stderr=subprocess.DEVNULL)
sys.exit(0)
