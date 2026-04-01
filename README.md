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
PDF Storage --> Text Extraction --> Profile-Driven LLM Analysis --> Dimensioned Extractions
      |                               (legacy profile: 6 passes / 40 dimensions)
      v
Embedding Generation --> ChromaDB Vector Store --> Semantic Search
      |
      v
LLM Council (optional) --> Multi-Provider Consensus --> Aggregated Extraction
```

## Features

- **Multi-Provider LLM Support**: Anthropic Claude, OpenAI GPT, Google Gemini, Ollama, and llama.cpp
- **Multiple Reference Sources**: Zotero, BibTeX, PDF folders, Mendeley, EndNote, and Paperpile
- **Flexible Extraction Modes**: CLI (free with subscriptions) or API (pay-per-use)
- **Portable Dimension Profiles**: Swap, extend, disable, diff, and backfill semantic dimensions per index
- **PDF Processing**: PyMuPDF extraction with OCR fallback for scanned documents
- **Semantic Search**: Vector similarity search with metadata filtering
- **RRF Search**: Reciprocal Rank Fusion combining semantic and keyword search for better recall
- **Agentic Search**: Multi-step iterative search with query refinement for complex questions
- **Citation Graph**: Build and analyze citation relationships between papers
- **Similarity Graph**: Visualize paper similarity networks using embeddings
- **Gap Detection**: Identify underexplored topics, sparse years, and future research directions
- **Topic Clustering**: UMAP + HDBSCAN clustering to discover topic groups in the corpus
- **Research Digest**: Generate summaries of newly indexed papers
- **Research Questions**: LLM-generated research questions from gap analysis reports
- **LLM Council**: Multi-provider consensus extraction with pluggable aggregation strategies (longest, quality-weighted, union merge), optional synthesis judge round, and generic query support
- **Quality Rating**: Automated quality scoring of paper extractions
- **Deep Review**: In-depth analysis of individual papers with full-text retrieval
- **Discord Bot**: Search your index from Discord with `/litris` commands
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
| `litris_search_rrf` | Reciprocal Rank Fusion search combining semantic and keyword matching |
| `litris_search_agentic` | Multi-step iterative search with automatic query refinement |
| `litris_deep_review` | In-depth analysis of a paper with full-text retrieval |
| `litris_get_paper` | Full extraction + PDF path for deep reading |
| `litris_similar` | Find papers similar to a given paper |
| `litris_clusters` | Topic clustering via UMAP + HDBSCAN on paper embeddings |
| `litris_summary` | Index coverage and statistics |
| `litris_collections` | List available Zotero collections |
| `litris_save_query` | Save search results to query_results/ folder |
| `litris_search_dimension` | Search within a specific semantic dimension using canonical ids, legacy `qNN` aliases, or roles |
| `litris_search_group` | Search across a group of dimensions by analysis pass |

**Setup**: Configure `.mcp.json` in project root and enable in `.claude/settings.json`:

```json
// .mcp.json
{
  "mcpServers": {
    "litris": {
      "command": "python",
      "args": ["-m", "src.mcp"],
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
| `/build` | Build/update the index from your reference library |
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
cp config.example.yaml config.yaml
# Edit config.yaml with your reference source paths
```

## Docker

For a containerized workflow, use the provided `Dockerfile` and `docker-compose.yml`.

Step 1: Copy the config template and update Zotero paths (or rely on Compose env overrides).

```bash
cp config.example.yaml config.yaml
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
# Preview the resolved sync plan
python scripts/build_index.py --explain-plan --dry-run

# Test build with 10 papers (using Claude subscription)
python scripts/build_index.py --limit 10 --use-subscription

# First run after upgrading to unified sync
python scripts/build_index.py --sync-mode full --use-subscription

# Normal day-to-day sync (default: --sync-mode auto)
python scripts/build_index.py --use-subscription

# Query the index (results auto-saved to data/query_results/)
python scripts/query_index.py -q "network analysis methods"
```

## Project Structure

```
LITRIS/
├── .claude/              # Claude Code configuration
│   ├── agents/           # Specialized AI agents (PI, pipeline, query, etc.)
│   ├── skills/           # Domain knowledge (extraction, search, citation)
│   └── commands/         # Slash commands (/build, /search, /review-paper)
├── src/                  # Source code
│   ├── analysis/         # LLM clients, 6-pass extraction, council, schemas
│   ├── extraction/       # PDF text extraction and cleaning
│   ├── indexing/         # Embeddings, ChromaDB, RAPTOR summaries
│   ├── mcp/              # MCP server for Claude Code integration
│   ├── query/            # Search engine (semantic, RRF, agentic, deep review)
│   └── zotero/           # Zotero database reader and models
├── scripts/              # CLI tools (build, query, compare, gap analysis)
├── data/                 # Index, cache, logs, query results (gitignored)
├── docs/                 # Documentation, proposals, guides
└── tests/                # Test suite (~980 tests)
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

dimensions:
  active_profile: "legacy_semantic_v1"
  profile_paths: []
  approval_required: true
  suggestion_sample_size: 25

embeddings:
  backend: "ollama"
  model: "qwen3-embedding:8b-q8_0"
  batch_size: auto      # or a fixed integer, e.g. 64
```

Use `--embedding-batch-size auto` to probe Ollama upward from a safe starting
size on the current machine and choose the largest successful batch that stays
within the per-request latency budget, or pass a fixed integer to override it
for a run.

### LLM Provider Options

| Provider | CLI Mode | API Mode | Default Model |
| -------- | -------- | -------- | ------------- |
| Anthropic | Claude Max | API key | claude-opus-4-6 |
| OpenAI | ChatGPT Plus/Pro | API key | gpt-5.4 |
| Google | N/A | API key | gemini-2.5-flash |
| Ollama | N/A | Local server | llama3 |
| llama.cpp | N/A | Local file | llama-3 |

See [docs/guides/openai-integration.md](docs/guides/openai-integration.md), [docs/guides/gemini-integration.md](docs/guides/gemini-integration.md), and [docs/guides/local-llm-integration.md](docs/guides/local-llm-integration.md) for setup details.

### Portable Dimension Profiles

Every index now carries an active semantic profile snapshot in
`dimension_profile.json`. The built-in fallback profile,
`legacy_semantic_v1`, preserves the historical `q01` through `q40`
aliases and `dim_qNN` chunk types so older indexes keep working without
forced re-extraction.

Use `scripts/dimensions.py` to migrate storage, diff profiles, suggest
additions, approve proposals, and backfill an existing corpus. The
operator workflow and storage format are documented in
[docs/dimension_profiles.md](docs/dimension_profiles.md).

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

### LLM Council

The LLM Council enables multi-provider consensus extraction where multiple LLMs independently analyze the same paper and their results are aggregated into a higher-quality consensus.

```bash
# Compare individual extractions vs council consensus
python scripts/council_comparison.py --keys PAPER_KEY1 PAPER_KEY2 --save

# Use API mode instead of CLI
python scripts/council_comparison.py --keys PAPER_KEY1 --mode api --save

# Use quality-weighted strategy instead of longest
python scripts/council_comparison.py --keys PAPER_KEY1 --strategy quality_weighted --save
```

**Aggregation strategies:**

| Strategy | Behavior |
|----------|----------|
| `longest` (default) | Selects longest response per field |
| `quality_weighted` | Scores by sentence structure, citations, numbers, multiplied by provider weight |
| `union` | Merges unique sentences across providers, deduplicating by 60% word overlap |

Per-field strategy overrides allow mixing strategies (e.g., `union` for key_claims, `quality_weighted` for thesis).

**Synthesis round** (optional): A judge LLM reviews all provider outputs and produces a merged best answer, resolving disagreements by comparing evidence quality.

### Build Options

```bash
# Preview what will be processed
python scripts/build_index.py --explain-plan --dry-run

# Day-to-day sync (auto chooses update vs full)
python scripts/build_index.py --use-subscription

# Skip DOI duplicates (when adding from new Zotero database)
python scripts/build_index.py --use-subscription --dedupe-by-doi

# Analyze DOI overlap before building
python scripts/build_index.py --show-doi-overlap

# Force a full rebuild of the vector store
python scripts/build_index.py --sync-mode full --use-subscription

# Rebuild embeddings from existing extractions
python scripts/build_index.py --sync-mode full --skip-extraction

# Show papers without PDFs
python scripts/build_index.py --show-skipped

# Limit processing to N papers
python scripts/build_index.py --limit 50 --use-subscription

# Build only papers from a specific collection
python scripts/build_index.py --collection "Machine Learning" --use-subscription

# Force update-only mode (Zotero only; fails if compatibility checks require full rebuild)
python scripts/build_index.py --sync-mode update --use-subscription

# Build or update with a custom dimension profile
python scripts/build_index.py --dimension-profile ./profiles/sts_policy.yaml --use-subscription

# Refresh a specific paper by Zotero key or paper_id
python scripts/build_index.py --paper ABC123_DEF456 --use-subscription
```

### Sync Modes

`scripts/build_index.py` is now the primary indexing entrypoint. The default
`--sync-mode auto` chooses the safest path automatically:

- `auto`: Use incremental sync when the index is compatible, otherwise upgrade to a full rebuild. Provider/model drift is logged as an advisory and does not re-extract unchanged papers.
- `full`: Rebuild the vector store for the full scope. Existing extractions are reused unless they are missing or incompatible.
- `update`: Force incremental sync for compatible Zotero indexes only. If compatibility checks fail, the command exits with an error instead of silently rebuilding.

`auto` and `update` preserve existing extractions even if you switch extraction providers or models. In that case, only new, modified, pending, or explicitly targeted papers use the currently requested provider. Use `--sync-mode full` when you intentionally want a corpus-wide re-extraction.

`--gap-fill` only runs against papers extracted in the current `build_index.py` invocation. For a corpus-wide low-coverage sweep, use `python scripts/run_gap_fill.py --threshold 0.90`. By default, that script auto-selects the opposite provider of each paper's original extraction; pass `--provider` to force an override.

For the first run after upgrading to the unified sync flow, run an explicit full rebuild once:

```bash
python scripts/build_index.py --sync-mode full --use-subscription
```

The legacy `scripts/update_index.py` command still exists as a deprecated compatibility wrapper, but new workflows should use `scripts/build_index.py`.

### Profile Migration And Backfill

For an existing index, migrate the extraction store once, diff the target
profile, then dry-run backfill before touching the full corpus:

```bash
python scripts/dimensions.py migrate-store --index-dir data/index --dry-run
python scripts/dimensions.py migrate-store --index-dir data/index
python scripts/dimensions.py diff --index-dir data/index --new-profile \
  ./profiles/sts_policy.yaml
python scripts/dimensions.py backfill --index-dir data/index \
  --dimension-profile ./profiles/sts_policy.yaml --dry-run
```

`backfill` now re-extracts only the sections touched by `added`,
`reworded`, and `replaced` dimensions. `disabled` dimensions do not
trigger LLM work; LITRIS retargets the stored records to the new profile
and refreshes embeddings so retired chunks drop out of active search.
Retired answers remain in storage by default.

### Gap Analysis

Identify underexplored topics, sparse publication years, and future research directions:

```bash
# Generate gap analysis report
python scripts/gap_analysis.py

# Limit to specific collections
python scripts/gap_analysis.py --collections "Network Science"

# Custom output path
python scripts/gap_analysis.py --output data/gap_report.json
```

### Research Questions

Generate LLM-powered research questions from gap analysis reports:

```bash
# Generate questions from an existing gap report
python scripts/research_questions.py

# Specify provider and scope
python scripts/research_questions.py --provider anthropic --scope narrow
```

### Research Digest

Summarize newly indexed papers:

```bash
# Generate digest of recent additions
python scripts/research_digest.py

# Limit papers and output as JSON
python scripts/research_digest.py --max-papers 20 --output-format json

# Dry run (don't mark papers as processed)
python scripts/research_digest.py --dry-run
```

### Discord Bot

Search your LITRIS index from Discord:

```bash
# Set token and run
export DISCORD_BOT_TOKEN=your-token-here
python scripts/run_discord_bot.py

# Or pass token directly
python scripts/run_discord_bot.py --token your-token-here
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

Extraction uses a 6-pass pipeline (each paper analyzed 6 times for different dimension groups). Costs reflect the full pipeline.

| Operation | CLI Mode | API (Opus 4.6) | Batch API (50% off) |
|-----------|----------|-----------------|---------------------|
| Test build (10 papers) | $0 | ~$9 | ~$4.50 |
| Full build (500 papers) | $0 | ~$450 | ~$225 |
| Incremental updates | $0 | Variable | Variable |

- **CLI mode**: Free with Max/Pro subscription (rate limited)
- **API mode**: Opus 4.6: $5/$25 per MTok; GPT-5.4: $2.50/$15 per MTok
- **Batch API mode**: 50% discount on all Anthropic models
- **Gemini**: Significantly cheaper -- 2.5 Flash at $0.30/$2.50 per MTok

## License

MIT

## Acknowledgments

- [Zotero](https://www.zotero.org/) for reference management
- [Anthropic Claude](https://www.anthropic.com/), [OpenAI](https://openai.com/), and [Google Gemini](https://ai.google.dev/) for LLM extraction
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [sentence-transformers](https://www.sbert.net/) for embeddings
