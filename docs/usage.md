# Usage Guide

This guide covers the main workflows for using the Literature Review Index System.

## Prerequisites

Before starting, ensure you have:
1. Completed the installation steps in README.md
2. Configured `config.yaml` with your Zotero paths
3. A Zotero library with PDF attachments

## Building the Index

### Initial Build

Build the full index from your Zotero library:

```bash
# Test with a small sample first
python scripts/build_index.py --limit 10

# Full build
python scripts/build_index.py
```

### Build Options

| Option | Description |
|--------|-------------|
| `--limit N` | Process only N papers |
| `--mode cli/api` | Override extraction mode |
| `--resume` | Resume from checkpoint |
| `--dry-run` | Show what would be processed |
| `--estimate-cost` | Estimate API costs |
| `--skip-extraction` | Skip LLM analysis |
| `--skip-embeddings` | Skip vector store |
| `--retry-failed` | Retry failed papers |

### CLI Mode (Recommended)

CLI mode uses Claude Code CLI for free extraction with a Max subscription:

```bash
# Verify CLI is set up
claude --version

# Build using CLI mode
python scripts/build_index.py --mode cli
```

### Batch API Mode

For faster bulk processing with API costs:

```bash
# Set API key
export ANTHROPIC_API_KEY=your_key_here

# Build using Batch API
python scripts/build_index.py --mode api
```

## Querying the Index

### Basic Search

```bash
# Simple query
python scripts/query_index.py -q "network analysis methods"

# With more results
python scripts/query_index.py -q "citation networks" -n 20
```

### Filtered Search

```bash
# Filter by year
python scripts/query_index.py -q "methodology" --year-min 2020

# Filter by collection
python scripts/query_index.py -q "research methods" --collection "PhD/Dissertation"
```

### Output Formats

```bash
# JSON output
python scripts/query_index.py -q "query" -f json

# Markdown output
python scripts/query_index.py -q "query" -f markdown

# CSV output
python scripts/query_index.py -q "query" -f csv
```

## Incremental Updates

Keep your index synchronized with Zotero changes:

```bash
# Check for changes without applying
python scripts/update_index.py --detect-only

# Apply all updates
python scripts/update_index.py

# Only add new papers
python scripts/update_index.py --new-only

# Only remove deleted papers
python scripts/update_index.py --delete-only
```

## Exporting Results

### Search Results

```bash
# Export search to markdown
python scripts/export_results.py search -q "your query" -f markdown -o results.md

# Export to BibTeX
python scripts/export_results.py search -q "your query" -f bibtex -o results.bib
```

### Full Library

```bash
# Export all papers to JSON
python scripts/export_results.py full -f json -o library.json

# Export to BibTeX
python scripts/export_results.py bibtex -o library.bib

# Export with extraction data
python scripts/export_results.py full -f json --include-extractions -o full_export.json
```

### Summary Statistics

```bash
# View summary
python scripts/export_results.py summary

# Export summary to file
python scripts/export_results.py summary -f markdown -o summary.md
```

## Working with Extractions

The system extracts structured information from each paper:

- **Thesis statement**: Main argument
- **Research questions**: Explicit questions addressed
- **Methodology**: Approach, design, data sources
- **Key findings**: Results with evidence types
- **Key claims**: Arguments with support types
- **Conclusions**: Summary of conclusions
- **Limitations**: Acknowledged limitations
- **Future directions**: Suggested research
- **Keywords**: Searchable terms
- **Discipline tags**: Academic fields

### Viewing Extraction Data

```bash
# Export with extractions
python scripts/export_results.py full --include-extractions -o data.json
```

The extraction data is stored in `data/index/extractions.json`.

## Data Locations

| Path | Contents |
|------|----------|
| `data/index/papers.json` | Paper metadata |
| `data/index/extractions.json` | LLM extractions |
| `data/index/summary.json` | Statistics |
| `data/index/metadata.json` | Build metadata |
| `data/index/chroma/` | Vector embeddings |
| `data/cache/pdf_text/` | Cached PDF text |

## Configuration Reference

### config.yaml

```yaml
zotero:
  database_path: "D:/Zotero/zotero.sqlite"
  storage_path: "D:/Zotero/storage"

extraction:
  mode: "cli"  # or "batch_api"
  model: "claude-opus-4-5-20251101"
  max_tokens: 100000
  timeout: 120

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384

storage:
  chroma_path: "data/chroma"
  cache_path: "data/cache"
  collection_name: "literature_review"

processing:
  batch_size: 10
  ocr_enabled: false
  min_text_length: 100
```

## Next Steps

- See [Query Guide](query_guide.md) for search examples
- See [Troubleshooting](troubleshooting.md) for common issues
