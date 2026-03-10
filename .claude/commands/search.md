---
description: Search the literature index
allowed-tools: Bash(python:*), Read
argument-hint: <query>
---

Search the literature index using semantic search.

Query: $ARGUMENTS

Execute the query_index.py script with the provided query.

Results are automatically saved to `data/query_results/` as markdown files with:

- Timestamp and date-based filename (e.g., `2025-12-06_graph-neural-networks.md`)
- Query text in blockquote format at top
- Ranked list of matching papers (title, authors, year, score)
- Matching excerpts from each paper

If filters are mentioned in the query (e.g., "from 2020", "about methods"), apply appropriate filters.

Output options:

- `--output markdown` (default) - Save as .md file
- `--output pdf` or `--pdf` - Save as PDF file
- `--output brief` - Compact console output
- `--no-save` - Skip auto-saving

Convert existing markdown to PDF:

- `--convert-to-pdf path/to/file.md`
