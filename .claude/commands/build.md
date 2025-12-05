---
description: Build the literature index from Zotero library
allowed-tools: Bash(python:*)
---

Build the literature index from the Zotero library.

Execute the build_index.py script with appropriate options:
- If no arguments provided, run a test build with --limit 10
- Pass any provided arguments to the script

Example usage:
- /build - Run test build (10 papers)
- /build --full - Full library build
- /build --resume - Resume from checkpoint

After build completes, report:
1. Number of papers processed
2. Number of extractions completed
3. Any failures encountered
4. Estimated cost (if available)
