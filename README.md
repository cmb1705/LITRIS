# Literature Review Indexing System (LITRIS)

Build a semantic search index from your academic library, enabling AI-assisted literature review, citation support, and research gap identification.

## Overview

LITRIS extracts structured insights from academic papers using LLM analysis, then indexes them for semantic search. Works with Zotero, Mendeley, EndNote, Paperpile, BibTeX files, or plain PDF folders. It enables:

- **Literature discovery** - Find papers relevant to research questions
- **Citation support** - Retrieve metadata and key claims for academic writing
- **Research front mapping** - Understand the current state of a discipline
- **Gap analysis** - Identify underexplored areas and future directions
- **AI-assisted synthesis** - Complex multi-part queries via Claude Code integration

## Architecture

```text
Reference Source --> Metadata Extraction --> Paper Records
(Zotero/Mendeley/EndNote/Paperpile/BibTeX/PDF Folder)
      |
      v
PDF Storage --> Text Extraction --> LLM Analysis --> Structured Extractions
      |
      v
Embedding Generation --> ChromaDB Vector Store --> Semantic Search
```

## Features

- **Multi-Provider LLM Support**: Anthropic Claude, OpenAI GPT, Google Gemini, Ollama, and llama.cpp
- **Multiple Reference Sources**: Zotero, BibTeX, PDF folders, Mendeley, EndNote, and Paperpile
- **Flexible Extraction Modes**: CLI (free with subscriptions) or API (pay-per-use)
- **PDF Processing**: PyMuPDF extraction with OCR fallback for scanned documents
- **Semantic Search**: Vector similarity search with metadata filtering
- **Incremental Updates**: Detect and process new/modified papers without full rebuild
- **Config Migration**: Automatic schema versioning and migration

## Collaborative Research Model

LITRIS enables a human-AI research collaboration where the index serves as compressed domain knowledge:

```
Research Question (Human)
        |
        v
LITRIS Index (your papers)
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
| `litris_save_query` | Save search results to query_results/ folder |

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

### Claude Code Slash Commands

When using Claude Code in this project, these slash commands are available:

| Command | Description |
|---------|-------------|
| `/search <query>` | Search the literature index with semantic search |
| `/build` | Build/update the index from Zotero library |
| `/review-paper <id>` | Review extraction quality for a specific paper |

## Prerequisites

- Python 3.10+
- Reference source (one of):
  - Zotero desktop with local library
  - BibTeX file (`.bib`) with optional PDF folder
  - Folder of PDF files
  - Mendeley Desktop
- LLM access (one of):
  - **Claude CLI**: Free with Max subscription (`claude login`)
  - **Codex CLI**: Free with ChatGPT Plus/Pro (`codex login`)
  - **API Key**: Anthropic, OpenAI, or Google Gemini
  - **Local LLM**: Ollama or llama.cpp (free, runs on your hardware)
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

## Docker

For a containerized workflow, use the provided `Dockerfile` and `docker-compose.yml`.

Step 1: Copy the config template and update Zotero paths (or rely on Compose env overrides).

```bash
cp config.yaml.example config.yaml
```

Step 2: Copy the environment template and set host paths for Docker mounts.

```bash
cp .env.example .env
```

Set these in `.env`:

```text
ZOTERO_DB_HOST=/path/to/zotero.sqlite
ZOTERO_STORAGE_HOST=/path/to/zotero/storage
```

Step 3: Build the image.

```bash
docker compose build
```

Run commands via Compose:

```bash
docker compose run --rm litris python scripts/build_index.py --dry-run
docker compose run --rm litris python scripts/query_index.py -q "network analysis methods"
```

Run the Web UI:

```bash
docker compose up web
# Open http://localhost:8501 in your browser
```

Note: The container targets API or batch modes. CLI modes require installing and authenticating
the Claude Code or Codex CLI inside the image or running on the host. See
`docs/troubleshooting.md` for CLI setup steps.

## Quick Start

```bash
# Preview what will be indexed
python scripts/build_index.py --dry-run

# Test build with 10 papers (using Claude subscription)
python scripts/build_index.py --limit 10 --use-subscription

# Query the index (results auto-saved to data/query_results/)
python scripts/query_index.py -q "network analysis methods"

# Full library build
python scripts/build_index.py --use-subscription
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
├── docs/                 # Documentation and specifications
└── tests/                # Test suite
```

## Configuration

Create a `config.yaml` file (see `config.example.yaml`):

```yaml
zotero:
  database_path: "/path/to/zotero.sqlite"
  storage_path: "/path/to/zotero/storage"

extraction:
  provider: "anthropic"  # or "openai", "google", "ollama", "llamacpp"
  mode: "cli"            # "cli" (subscription) or "api" (pay-per-use)
  model: ""              # Leave empty for provider default

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
```

### LLM Provider Options

| Provider | CLI Mode | API Mode | Default Model |
| -------- | -------- | -------- | ------------- |
| Anthropic | Claude Max | API key | claude-opus-4-5-20251101 |
| OpenAI | ChatGPT Plus/Pro | API key | gpt-5.2 |
| Google | N/A | API key | gemini-3-pro |
| Ollama | N/A | Local server | llama3 |
| llama.cpp | N/A | Local file | llama-3 |

See [docs/guides/openai-integration.md](docs/guides/openai-integration.md), [docs/guides/gemini-integration.md](docs/guides/gemini-integration.md), and [docs/guides/local-llm-integration.md](docs/guides/local-llm-integration.md) for setup details.

### Reference Source Options

| Source | Command | Use Case |
| ------ | ------- | -------- |
| Zotero | `--provider zotero` (default) | Full-featured reference manager |
| BibTeX | `--provider bibtex --bibtex-path file.bib` | Any ref manager export |
| PDF Folder | `--provider pdffolder --folder-path ./papers/` | No ref manager needed |
| Mendeley | `--provider mendeley --db-path mendeley.sqlite` | Mendeley Desktop users |

See [docs/guides/alternative-sources.md](docs/guides/alternative-sources.md) for details.

## Usage Examples

### Search the Literature

```bash
# Basic search (auto-saves to data/query_results/)
python scripts/query_index.py -q "citation network analysis"

# Filter by year
python scripts/query_index.py -q "research methods" --year-min 2020

# Filter by chunk type (thesis, methodology, findings, etc.)
python scripts/query_index.py -q "machine learning" --chunk-types methodology findings

# Export as PDF
python scripts/query_index.py -q "emerging topics" --pdf

# Brief console output (no file save)
python scripts/query_index.py -q "quick check" --output brief --no-save

# Convert existing markdown report to PDF
python scripts/query_index.py --convert-to-pdf data/query_results/report.md

# Get paper details by ID
python scripts/query_index.py --paper "PAPER_ID"

# Find similar papers
python scripts/query_index.py --similar "PAPER_ID"

# Index statistics
python scripts/query_index.py --summary

```

### Web UI

Launch the local search workbench with Streamlit:

```bash
python -m streamlit run scripts/web_ui.py
```

The UI expects an existing index at `data/index` (run `scripts/build_index.py` if needed)
and saves exports to `data/query_results/`.

**Features:**

| Feature | Description |
|---------|-------------|
| Semantic search | Query the index with natural language |
| Filters | Year range, collections, item types, chunk types |
| Quick filters | One-click common filters (Recent, Methods, etc.) |
| Quick filter from metadata | Click year/type/collection in results to filter |
| Sort options | Sort by relevance, year (newest/oldest), or title (A-Z/Z-A) |
| Detail panel | Full extraction view with PDF access |
| On-demand extraction | Load extraction only for focused paper (performance) |
| One-click citation copy | Copy citation in APA, MLA, Chicago, or BibTeX format |
| Similarity network | Interactive PyVis visualization of related papers |
| Export | CSV, BibTeX, and PDF export formats |

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `Enter` | Execute search |
| `j` / `Down` | Next result |
| `k` / `Up` | Previous result |
| `o` | Open detail panel |
| `Esc` | Close detail panel |

Install visualization dependencies for similarity network:

```bash
pip install pyvis networkx
```

### AI-Assisted Literature Queries

Beyond basic search, LITRIS enables complex multi-part research queries when paired with Claude Code. The AI can:

- Search the index semantically across multiple dimensions
- Synthesize findings across papers
- Compare your library to external literature via web search
- Generate structured reports with citations and recommendations

### Example: GNN vs LSTM for Citation Forecasting

**Query:**
> Part 1: Help me understand how Graph Neural Networks (GNNs) can be used to forecast growth in citation networks for particular research fronts.
>
> Part 2: How does this compare to current research that forecasts that growth using LSTM-based RNNs?
>
> Part 3: After answering the questions, provide a top-10 list of papers to review that cover these topics with a relevance score for each.
>
> Part 4: Compare the works in LITRIS to other relevant web searches of scientometric and infometric academic journals and provide a list of 5 additional papers to read on the topic.

**Output:** [GNN vs LSTM Citation Forecasting Report](docs/examples/2025-12-06_gnn-lstm-citation-forecasting.md)

The generated report includes:

- Comparative analysis of GNN and LSTM approaches
- Top-10 ranked papers from the index with relevance scores
- Gap analysis comparing local library to external literature
- Five additional recommended papers from web search
- Raw search results for transparency

### Build Options

```bash
# Preview what will be processed
python scripts/build_index.py --dry-run

# Use Claude subscription (free with Max/Pro)
python scripts/build_index.py --use-subscription

# Skip DOI duplicates (when adding from new Zotero database)
python scripts/build_index.py --use-subscription --dedupe-by-doi

# Analyze DOI overlap before building
python scripts/build_index.py --show-doi-overlap

# Rebuild embeddings only (skip extraction)
python scripts/build_index.py --skip-extraction

# Show papers without PDFs
python scripts/build_index.py --show-skipped

# Limit processing to N papers
python scripts/build_index.py --limit 50 --use-subscription
```

### Incremental Updates

The build script automatically detects new papers and only processes those not already in the index:

```bash
# Just run build again - it skips already-extracted papers
python scripts/build_index.py --use-subscription
```

## Documentation

| Document | Description |
|----------|-------------|
| [STATE.md](STATE.md) | Implementation progress tracker |
| [CLAUDE.md](CLAUDE.md) | Project memory for Claude Code |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines and development setup |
| [Technical Specification](docs/proposals/completed/technical_specification.md) | Full system design (Phase 1) |
| [MCP Technical Specification](docs/proposals/mcp_technical_specification.md) | MCP integration design |
| [API Documentation](docs/sphinx/_build/html/index.html) | Sphinx-generated API reference |

### Continuous Integration

Pull requests trigger GitHub Actions CI (`.github/workflows/ci.yml`) which runs:

- **pytest** across Python 3.10-3.12 on Ubuntu, Windows, and macOS
- **mypy** type checking on src/
- **pip-audit** security vulnerability scanning

Pre-commit hooks (`.pre-commit-config.yaml`) enforce code quality on every commit.

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
- [Anthropic Claude](https://www.anthropic.com/), [OpenAI](https://openai.com/), and [Google Gemini](https://ai.google.dev/) for LLM extraction
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [sentence-transformers](https://www.sbert.net/) for embeddings
