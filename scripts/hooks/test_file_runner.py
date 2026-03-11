#!/usr/bin/env python3
"""Run pytest on test files that were directly edited."""
import os
import re
import subprocess
import sys

path = os.environ.get("CLAUDE_FILE_PATH", "")
if not path:
    sys.exit(0)

if re.search(r"tests[/\\]test_.*\.py$", path):
    result = subprocess.run(["pytest", path, "-v", "--tb=short"])
    sys.exit(result.returncode)
