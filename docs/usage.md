# Usage Guide

This guide covers the main workflows for using the Literature Review Index System.

## Prerequisites

Before starting, ensure you have:

1. Completed the installation steps in README.md
2. Configured `config.yaml` with your reference source paths
3. A reference source with papers (Zotero, BibTeX, PDF folder, or Mendeley)

## Building the Index

### Initial Build

Build the full index from your reference library:

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

CLI mode uses your existing subscription for free extraction:

```bash
# Anthropic (Claude Max subscription)
claude --version  # Verify installed
python scripts/build_index.py --mode cli --provider anthropic

# OpenAI (ChatGPT Plus/Pro subscription)
codex --version  # Verify installed
python scripts/build_index.py --mode cli --provider openai
```

### API Mode

For faster bulk processing with pay-per-use:

```bash
# Anthropic
export ANTHROPIC_API_KEY=your_key_here
python scripts/build_index.py --mode api --provider anthropic

# OpenAI
export OPENAI_API_KEY=your_key_here
python scripts/build_index.py --mode api --provider openai

# Google Gemini
export GOOGLE_API_KEY=your_key_here
python scripts/build_index.py --mode api --provider google
```

### Local LLM Mode

For offline extraction using local models:

```bash
# Ollama (requires Ollama server running)
python scripts/build_index.py --provider ollama --model llama3

# llama.cpp (requires model file)
python scripts/build_index.py --provider llamacpp --model-path ./models/llama-3.gguf
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

LITRIS uses `config.yaml` for all settings. See `config.example.yaml` for the full template.

### Configuration Versioning

The configuration file includes a schema version that enables automatic migration:

```yaml
version: "1.2.0"  # Do not modify manually
```

When you upgrade LITRIS, old config files are automatically migrated to the current schema.
A backup is created before migration (e.g., `config.yaml.bak.1.1.0`).

### Multi-Provider LLM Configuration

```yaml
extraction:
  provider: "anthropic"  # "anthropic", "openai", or "google"
  mode: "cli"            # "cli" (subscription) or "api" (pay-per-use)
  model: ""              # Leave empty for provider default
  max_tokens: 100000
  timeout: 120
```

Provider defaults:

- **Anthropic**: claude-opus-4-5-20251101
- **OpenAI**: gpt-5.2
- **Google**: gemini-3-pro

### Model Overrides by Document Type

Optimize cost by using different models for different document types:

```yaml
extraction:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"  # Default model
  model_overrides:
    journal_article: "claude-sonnet-4-20250514"
    book: "claude-opus-4-5-20251101"           # Better for books
    thesis: "claude-opus-4-5-20251101"         # Better for theses
    conference_paper: "gemini-2.5-flash"       # Cost-effective
    preprint: "gemini-2.5-flash"               # Cost-effective
```

### Full Configuration Example

```yaml
version: "1.2.0"

zotero:
  database_path: "D:/Zotero/zotero.sqlite"
  storage_path: "D:/Zotero/storage"

extraction:
  provider: "anthropic"
  mode: "cli"
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

## Alternative Reference Sources

LITRIS supports multiple reference sources beyond Zotero:

```bash
# BibTeX file
python scripts/build_index.py --provider bibtex --bibtex-path refs.bib --pdf-dir ./papers/

# PDF folder (no reference manager)
python scripts/build_index.py --provider pdffolder --folder-path ./papers/

# Mendeley Desktop
python scripts/build_index.py --provider mendeley --db-path mendeley.sqlite

# EndNote XML export
python scripts/build_index.py --provider endnote --xml-path library.xml

# Paperpile BibTeX export
python scripts/build_index.py --provider paperpile --bibtex-path paperpile.bib
```

See [guides/alternative-sources.md](guides/alternative-sources.md) for detailed setup.

## Web UI

Launch the Streamlit search workbench:

```bash
python -m streamlit run scripts/web_ui.py
```

The Web UI provides:

- **Semantic search** with natural language queries
- **Filters** for year range, collections, item types, and chunk types
- **Quick filters** for common searches (Recent, Methods, etc.)
- **Sort options** by relevance, year, or title
- **Detail panel** with full extraction and PDF access
- **Similarity network** visualization using PyVis
- **Export** to CSV, BibTeX, and PDF formats
- **One-click citation copy** in APA, MLA, Chicago, or BibTeX

Keyboard shortcuts: `j`/`k` navigate results, `o` opens detail panel, `Esc` closes it.

## Next Steps

- See [Query Guide](query_guide.md) for search examples
- See [Troubleshooting](troubleshooting.md) for common issues
- See [guides/openai-integration.md](guides/openai-integration.md) for OpenAI/Codex setup
- See [guides/gemini-integration.md](guides/gemini-integration.md) for Google Gemini setup
- See [guides/alternative-sources.md](guides/alternative-sources.md) for BibTeX/PDF folder/Mendeley
