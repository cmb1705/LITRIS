#!/usr/bin/env python3
"""Run pytest for test files related to a modified src/ module."""
import os
import re
import subprocess
import sys
from pathlib import Path

path = os.environ.get("CLAUDE_FILE_PATH", "")
if not path:
    sys.exit(0)
if not re.search(r"src[/\\].*\.py$", path):
    sys.exit(0)
if "__pycache__" in path or path.endswith("__init__.py"):
    sys.exit(0)

# Extract module name from src/<module>/... or src/<module>.py
relative = re.sub(r".*src[/\\]", "", path)
parts = re.split(r"[/\\]", relative)
module = parts[0].replace(".py", "")

# File stem for direct match
stem = Path(path).stem

# Build candidates (most specific first)
candidates = []
if stem != module:
    candidates.append(f"tests/test_{stem}.py")
candidates.append(f"tests/test_{module}.py")

for candidate in candidates:
    if os.path.exists(candidate):
        result = subprocess.run(["pytest", candidate, "-v", "--tb=short", "-x"])
        sys.exit(result.returncode)

sys.exit(0)
