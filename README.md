# Literature Review Index System

Build a semantic search index from a Zotero library, enabling AI-assisted literature review, citation support, and research gap identification.

## Overview

This system extracts structured insights from academic papers in a Zotero library using LLM analysis, then indexes them for semantic search. It enables:

- **Literature discovery** - Find papers relevant to research questions
- **Citation support** - Retrieve metadata and key claims for academic writing
- **Research front mapping** - Understand the current state of a discipline
- **Gap analysis** - Identify underexplored areas and future directions

## Architecture

```
Zotero SQLite ──> Metadata Extraction ──> Paper Records
      │
      v
PDF Storage ──> Text Extraction ──> LLM Analysis ──> Structured Extractions
      │
      v
Embedding Generation ──> ChromaDB Vector Store ──> Semantic Search
```

## Features

- **Zotero Integration**: Read-only access to Zotero SQLite database and PDF storage
- **PDF Processing**: PyMuPDF extraction with OCR fallback for scanned documents
- **LLM Extraction**: Claude-powered extraction of thesis, methods, findings, limitations
- **Semantic Search**: Vector similarity search with metadata filtering
- **Incremental Updates**: Detect and process new/modified papers without full rebuild

## Prerequisites

- Python 3.10+
- Zotero desktop with local library
- Anthropic API key (Claude access)
- Optional: Tesseract OCR for scanned PDFs

## Installation

```bash
# Clone repository
git clone git@github.com:cmb1705/Lit_Review.git
cd Lit_Review

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Edit config.yaml with your Zotero paths
```

## Quick Start

```bash
# Test build with 10 papers
python scripts/build_index.py --limit 10

# Query the index
python scripts/query_index.py -q "network analysis methods"

# Full library build
python scripts/build_index.py
```

## Project Structure

```
Lit_Review/
├── .claude/              # Claude Code configuration
│   ├── agents/           # Specialized AI agents
│   ├── skills/           # Domain knowledge
│   └── commands/         # Slash commands
├── src/                  # Source code
│   ├── zotero/           # Zotero database reader
│   ├── extraction/       # PDF and LLM extraction
│   ├── indexing/         # Embeddings and storage
│   └── query/            # Search interface
├── scripts/              # CLI tools
├── data/                 # Index and cache (gitignored)
├── proposals/            # Technical specification and tasks
└── tests/                # Test suite
```

## Documentation

| Document | Description |
|----------|-------------|
| [STATE.md](STATE.md) | Implementation progress tracker |
| [CLAUDE.md](CLAUDE.md) | Project memory for Claude Code |
| [Technical Specification](proposals/technical_specification.md) | Full system design |
| [Project Plan](proposals/project_plan.md) | Phased TODO lists |
| [Task Files](proposals/tasks/) | Detailed implementation guides |

## Configuration

Edit `config.yaml` to customize:

```yaml
zotero:
  database_path: "D:/Zotero/zotero.sqlite"
  storage_path: "D:/Zotero/storage"

extraction:
  model: "claude-3-opus-20240229"

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
```

## Usage Examples

### Search the Literature

```bash
# Basic search
python scripts/query_index.py -q "citation network analysis"

# Filter by year
python scripts/query_index.py -q "research methods" --year-min 2020

# Output as markdown
python scripts/query_index.py -q "methodology" -f markdown
```

### Update the Index

```bash
# Check for changes
python scripts/update_index.py --detect-only

# Apply updates
python scripts/update_index.py
```

## Development Status

**Current Phase**: Planning Complete

See [STATE.md](STATE.md) for detailed progress tracking.

| Phase | Status |
|-------|--------|
| 0: Setup | Not Started |
| 1: Foundation | Not Started |
| 2: Semantic Search | Not Started |
| 3: Robustness | Not Started |
| 4: Incremental Updates | Not Started |
| 5: Refinement | Not Started |

## Cost Estimates

| Operation | Estimated Cost |
|-----------|---------------|
| Test build (10 papers) | ~$3 |
| Full build (500 papers) | ~$135 |
| Incremental updates | Variable |

Costs assume Claude Opus. Using Sonnet reduces costs by ~90%.

## License

MIT

## Acknowledgments

- [Zotero](https://www.zotero.org/) for reference management
- [Anthropic Claude](https://www.anthropic.com/) for LLM extraction
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [sentence-transformers](https://www.sbert.net/) for embeddings
