# Literature Review Indexing System (LITRIS)

Build a semantic search index from a Zotero library, enabling AI-assisted literature review, citation support, and research gap identification.

## Overview

LITRIS extracts structured insights from academic papers in a Zotero library using LLM analysis, then indexes them for semantic search. It enables:

- **Literature discovery** - Find papers relevant to research questions
- **Citation support** - Retrieve metadata and key claims for academic writing
- **Research front mapping** - Understand the current state of a discipline
- **Gap analysis** - Identify underexplored areas and future directions

## Architecture

```
Zotero SQLite --> Metadata Extraction --> Paper Records
      |
      v
PDF Storage --> Text Extraction --> LLM Analysis --> Structured Extractions
      |
      v
Embedding Generation --> ChromaDB Vector Store --> Semantic Search
```

## Features

- **Zotero Integration**: Read-only access to Zotero SQLite database and PDF storage
- **PDF Processing**: PyMuPDF extraction with OCR fallback for scanned documents
- **Dual Extraction Modes**: CLI (free with Max subscription) or Batch API (paid, 50% discount)
- **Semantic Search**: Vector similarity search with metadata filtering
- **Incremental Updates**: Detect and process new/modified papers without full rebuild
- **Multi-attachment Support**: Handles papers with multiple PDF attachments

## Collaborative Research Model

LITRIS enables a human-AI research collaboration where the index serves as compressed domain knowledge:

```
Research Question (Human)
        |
        v
LITRIS Index (236 papers)
        |
        v
[Extractions: thesis, methodology, findings, limitations]
        |
        +--> AI scans 20-30 papers' core arguments
        +--> Identifies patterns, contradictions, gaps
        +--> Selects 3-5 papers for deep reading
        |
        v
Full PDF Text (targeted retrieval)
        |
        +--> Verifies specific claims
        +--> Extracts detailed methods
        +--> Finds nuanced arguments
        |
        v
Synthesis + Gap Analysis
        |
        +--> What does the index cover well?
        +--> What's missing from the corpus?
        +--> Web search for external papers
        |
        v
Research Directions (Human + AI)
```

### Workflow

1. **Human frames the question** - Guides inquiry, determines search direction
2. **AI searches the index** - Semantic search across extractions
3. **AI synthesizes findings** - Compares arguments, methods, results across papers
4. **AI flags papers for deep reading** - When extraction detail is insufficient
5. **Human guides deeper exploration** - Pivots, narrows, or expands scope
6. **AI identifies gaps** - Compares corpus to external literature
7. **Together propose directions** - Novel research questions, methodological innovations

### MCP Integration

Direct tool access for Claude Code enables seamless research collaboration:

| Tool | Purpose |
|------|---------|
| `litris_search` | Semantic search with filters (year, collection, chunk type) |
| `litris_get_paper` | Full extraction + PDF path for deep reading |
| `litris_similar` | Find papers similar to a given paper |
| `litris_summary` | Index coverage and statistics |
| `litris_collections` | List available Zotero collections |

**Setup**: Configure `.mcp.json` in project root and enable in `.claude/settings.json`:

```json
// .mcp.json
{
  "mcpServers": {
    "litris": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/LITRIS"
    }
  }
}
```

See [docs/mcp_troubleshooting.md](docs/mcp_troubleshooting.md) for setup details.

## Prerequisites

- Python 3.10+
- Zotero desktop with local library
- **For CLI mode**: Claude Code CLI with Max subscription (`claude login`)
- **For Batch API mode**: Anthropic API key
- Optional: Tesseract OCR for scanned PDFs

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/LITRIS.git
cd LITRIS

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure paths
cp config.yaml.example config.yaml
# Edit config.yaml with your Zotero paths
```

## Quick Start

```bash
# Test build with 10 papers
python scripts/build_index.py --limit 10 --mode cli

# Query the index
python scripts/query_index.py -q "network analysis methods"

# Full library build
python scripts/build_index.py --mode cli
```

## Project Structure

```
LITRIS/
├── .claude/              # Claude Code configuration
│   ├── agents/           # Specialized AI agents
│   ├── skills/           # Domain knowledge
│   └── commands/         # Slash commands
├── src/                  # Source code
│   ├── zotero/           # Zotero database reader
│   ├── analysis/         # PDF and LLM extraction
│   ├── indexing/         # Embeddings and storage
│   └── query/            # Search interface
├── scripts/              # CLI tools
├── data/                 # Index and cache (gitignored)
├── proposals/            # Technical specifications
└── tests/                # Test suite
```

## Configuration

Create a `config.yaml` file (see `config.yaml.example`):

```yaml
zotero:
  database_path: "/path/to/zotero.sqlite"
  storage_path: "/path/to/zotero/storage"

extraction:
  mode: "cli"  # "cli" (free) or "batch_api" (paid, 50% discount)
  model: "claude-opus-4-5-20251101"

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
```

### Extraction Mode Comparison

| Mode | Cost | Speed | Best For |
|------|------|-------|----------|
| `cli` | Free (Max subscription) | ~30s/paper | Budget, incremental updates |
| `batch_api` | ~$0.14/paper | ~1hr for 500 | Speed, bulk builds |

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

### Build Options

```bash
# Parallel extraction (faster)
python scripts/build_index.py --mode cli --parallel 5

# Skip embeddings (extraction only)
python scripts/build_index.py --mode cli --skip-embeddings

# Rebuild embeddings only
python scripts/build_index.py --skip-extraction

# Retry failed extractions
python scripts/build_index.py --mode cli --retry-failed

# Show papers without PDFs
python scripts/build_index.py --show-skipped
```

### Update the Index

```bash
# Check for changes
python scripts/update_index.py --detect-only

# Apply updates
python scripts/update_index.py
```

## Documentation

| Document | Description |
|----------|-------------|
| [STATE.md](STATE.md) | Implementation progress tracker |
| [CLAUDE.md](CLAUDE.md) | Project memory for Claude Code |
| [Technical Specification](proposals/completed/technical_specification.md) | Full system design (Phase 1) |
| [MCP Technical Specification](proposals/mcp_technical_specification.md) | MCP integration design |

## Cost Estimates

| Operation | CLI Mode | Batch API Mode |
|-----------|----------|----------------|
| Test build (10 papers) | $0 | ~$1.35 |
| Full build (500 papers) | $0 | ~$67.50 |
| Incremental updates | $0 | Variable |

- **CLI mode**: Free with Max subscription (rate limited)
- **Batch API mode**: Pay-per-token with 50% batch discount

## License

MIT

## Acknowledgments

- [Zotero](https://www.zotero.org/) for reference management
- [Anthropic Claude](https://www.anthropic.com/) for LLM extraction
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [sentence-transformers](https://www.sbert.net/) for embeddings
