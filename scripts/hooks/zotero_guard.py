#!/usr/bin/env python3
"""Block edits to Zotero storage directories."""
import os
import re
import sys

path = os.environ.get("CLAUDE_FILE_PATH", "")
if not path:
    sys.exit(0)

zotero_path = os.environ.get("ZOTERO_STORAGE_PATH", "")
if zotero_path:
    pattern = re.escape(zotero_path)
else:
    pattern = "Zotero"

if re.search(pattern, path):
    print("BLOCKED: Cannot modify Zotero directory", file=sys.stderr)
    sys.exit(1)
