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
| `--classify-only` | Run classification pre-pass only (no extraction) |
| `--index-all` | Extract all papers regardless of classification |
| `--source SOURCE` | Reference source: zotero, bibtex, pdffolder, mendeley, endnote, paperpile |

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

## RRF Search (Multi-Query Fusion)

Reciprocal Rank Fusion (RRF) improves recall by automatically generating query
reformulations via LLM, running each through vector search, and fusing results
using the RRF formula:

```
score(d) = sum(1 / (k + rank_i(d)))
```

This surfaces papers that would be missed by a single query phrasing.

### Usage via MCP

The `litris_search_rrf` MCP tool exposes RRF search directly:

```
litris_search_rrf(query="network analysis methods", n_results=10)
```

### Programmatic Usage

```python
from src.query.rrf import generate_query_variants, rrf_score

# Generate query variants via LLM
variants = generate_query_variants("network analysis", n_variants=4)
# Returns: ["network analysis", "graph theory methods", ...]

# Run each variant through your search, collect ranked paper IDs
rankings = [search(v) for v in variants]

# Fuse results
fused = rrf_score(rankings, k=60)
# Returns: [("paper_id_1", 0.048), ("paper_id_2", 0.032), ...]
```

### Parameters

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `n_variants` | 4 | Number of LLM-generated query reformulations |
| `k` | 60 | RRF constant (from the original RRF paper) |
| `provider` | anthropic | LLM provider for reformulation |

## Agentic Search (Multi-Round with Gap Analysis)

Agentic search implements an iterative search loop that identifies topical gaps
in results and generates follow-up queries to fill them:

1. Initial query runs vector search
2. LLM analyzes results for missing sub-topics, methods, or perspectives
3. LLM generates follow-up queries to fill gaps
4. Re-searches with follow-up queries and merges results
5. Repeats for up to `max_rounds` gap-analysis rounds

This is particularly useful for comprehensive literature reviews where a single
query cannot capture all relevant dimensions of a topic.

### Usage via MCP

The `litris_search_agentic` MCP tool exposes agentic search:

```
litris_search_agentic(query="social network analysis in education", n_results=15, max_rounds=2)
```

### Programmatic Usage

```python
from src.query.agentic import analyze_gaps, AgenticSearchResult

# Analyze search results for gaps
gap_analysis = analyze_gaps(
    query="network analysis",
    results=initial_results,
    provider="anthropic",
)

print(gap_analysis.gaps)
# ["No papers on temporal network analysis", ...]

print(gap_analysis.follow_up_queries)
# ["temporal dynamics in network evolution", ...]
```

### Parameters

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `max_rounds` | 2 | Maximum gap-analysis rounds |
| `n_results` | 10 | Results per search round |
| `provider` | anthropic | LLM provider for gap analysis |

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

LITRIS supports multiple reference sources beyond Zotero via the `--source` flag:

```bash
# BibTeX file
python scripts/build_index.py --source bibtex --source-path refs.bib

# PDF folder (no reference manager)
python scripts/build_index.py --source pdffolder --source-path ./papers/

# Mendeley Desktop
python scripts/build_index.py --source mendeley --source-path mendeley.sqlite

# EndNote XML export
python scripts/build_index.py --source endnote --source-path library.xml

# Paperpile BibTeX export
python scripts/build_index.py --source paperpile --source-path paperpile.bib
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

## Gap Analysis

Identify underrepresented topics, methodologies, and temporal gaps in your literature:

```bash
# Generate gap analysis report
python scripts/gap_analysis.py --index-dir data/index

# Output as JSON
python scripts/gap_analysis.py --output-format json

# Filter to specific collections
python scripts/gap_analysis.py --collections "PhD/Core" "Methods"
```

### Gap Analysis Options

| Option | Description |
| ------ | ----------- |
| `--max-items N` | Maximum gaps per section (default: 10) |
| `--min-count N` | Minimum count threshold (default: 2) |
| `--quantile Q` | Quantile threshold (default: 0.2) |
| `--include-abstracts` | Include abstracts in coverage estimation |

Output saved to `data/out/experiments/gap_analysis/`.

## Research Question Generation

Generate research questions from gap analysis using LLMs:

```bash
# Generate from existing gap report
python scripts/research_questions.py --gap-report data/out/experiments/gap_analysis/gap_report.json

# Run gap analysis first, then generate
python scripts/research_questions.py --index-dir data/index

# Preview prompts without LLM call
python scripts/research_questions.py --gap-report report.json --dry-run
```

### Generation Options

| Option | Description |
| ------ | ----------- |
| `--provider` | LLM provider: anthropic, openai, google (default: anthropic) |
| `--model` | Model override (default: provider's default) |
| `--count N` | Questions per gap (default: 3) |
| `--scope` | Question scope: narrow, moderate, broad (default: moderate) |
| `--styles` | Allowed styles: exploratory, causal, comparative, evaluative, descriptive |
| `--max-gaps N` | Maximum gaps to process (default: 5) |
| `--dry-run` | Show prompts without calling LLM |
| `--verbose` | Show detailed progress |

### Example Workflows

```bash
# Generate narrow-scope causal questions
python scripts/research_questions.py \
  --gap-report report.json \
  --scope narrow \
  --styles causal comparative \
  --count 5

# Use OpenAI with specific model
python scripts/research_questions.py \
  --gap-report report.json \
  --provider openai \
  --model gpt-5.4

# JSON output for programmatic use
python scripts/research_questions.py \
  --index-dir data/index \
  --output-format json
```

Output saved to `data/out/experiments/research_questions/`.

## LLM Council (Multi-Provider Consensus)

The LLM Council enables consensus-based extraction by querying multiple providers
in parallel and aggregating their responses. This improves extraction robustness
by combining insights from different models.

### Programmatic Usage

```python
from src.analysis.llm_council import LLMCouncil, CouncilConfig, ProviderConfig

# Configure providers with reliability weights
config = CouncilConfig(
    providers=[
        ProviderConfig(name="anthropic", weight=1.2),
        ProviderConfig(name="openai", weight=1.0),
        ProviderConfig(name="google", weight=0.8),
    ],
    min_responses=2,  # Need at least 2 responses for consensus
    fallback_to_single=True,  # Use single response if threshold not met
    parallel=True,  # Query providers simultaneously
)

# Create council and extract
council = LLMCouncil(config)
result = council.extract(
    paper_id="paper123",
    title="Paper Title",
    authors="Author Name",
    year=2024,
    item_type="article",
    text="Full paper text...",
)

if result.success:
    consensus = result.consensus
    print(f"Consensus confidence: {result.consensus_confidence:.2f}")
    print(f"Providers responded: {len(result.provider_responses)}")
    print(f"Total cost: ${result.total_cost:.4f}")
```

### Consensus Strategies

The council uses field-specific strategies to build consensus:

| Field Type | Strategy | Description |
| ---------- | -------- | ----------- |
| Text (thesis, conclusions) | Longest | Select most detailed response |
| Lists (keywords, limitations) | Union | Combine all unique values |
| Nested (methodology, findings) | Merge | Deduplicate and combine |
| Numeric (confidence) | Weighted average | Average weighted by provider reliability |

### Configuration Options

| Option | Description |
| ------ | ----------- |
| `providers` | List of ProviderConfig with name, weight, timeout |
| `min_responses` | Minimum successful responses for consensus (default: 2) |
| `fallback_to_single` | Use single response if min not met (default: True) |
| `parallel` | Query providers simultaneously (default: True) |
| `timeout` | Overall timeout in seconds (default: 180) |

### Provider Weights

Provider weights affect consensus when combining extractions:

- Higher weight = more influence on combined fields
- Default weight: 1.0
- Adjust based on provider reliability for your domain

```python
# Example: Trust Anthropic more for academic papers
ProviderConfig(name="anthropic", weight=1.5)
ProviderConfig(name="openai", weight=1.0)
```

## Citation Graph

LITRIS can build a citation graph from your indexed papers, showing how papers in
your library reference each other. The graph is generated automatically during
`build_index.py` (unless skipped), or you can access it through the Web UI's
Citation Network tab.

The citation graph is built from reference lists extracted during indexing. Papers
are linked when one cites another, and duplicate papers are merged by DOI.

### Controlling Graph Generation During Build

```bash
# Normal build includes citation graph
python scripts/build_index.py

# Skip citation graph generation
python scripts/build_index.py --skip-similarity
```

### Viewing the Citation Graph

The Web UI provides an interactive citation network visualization (requires
`pyvis` and `networkx`). Launch the UI and navigate to the **Citation Network**
tab to explore connections between papers in your library.

## Similarity Graph

LITRIS pre-computes a topical similarity graph alongside the citation graph.
It uses cosine similarity on paper centroid embeddings (the mean of each
paper's chunk embeddings) to identify related papers that may not directly
cite each other.

The similarity graph is generated automatically during `build_index.py`
(unless skipped with `--skip-similarity`). Output is saved to
`data/index/similarity_graph.json`.

### Configuration

The similarity graph uses the following defaults:

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `min_similarity` | 0.75 | Minimum cosine similarity to create an edge |
| `max_edges_per_node` | 10 | Maximum edges per paper (keeps top-N) |
| `use_paper_centroid` | True | Use mean of chunk embeddings as paper vector |
| `batch_size` | 100 | Papers processed per batch |

### Programmatic Usage

```python
from src.analysis.similarity_graph import (
    SimilarityConfig,
    load_and_build_similarity_graph,
)

config = SimilarityConfig(min_similarity=0.8, max_edges_per_node=5)
graph = load_and_build_similarity_graph("data/index", config)

print(f"Edges: {len(graph['edges'])}")
print(f"Nodes: {len(graph['nodes'])}")
```

### Jaccard Validation

Pass `compute_jaccard=True` to add a token-level Jaccard score to each edge.
This helps identify cross-disciplinary connections where cosine similarity is
high but vocabulary overlap is low.

## Collection Filtering for Builds

Use `--collection` to restrict a build to papers belonging to a specific Zotero
collection. The filter matches collection names by substring (case-sensitive).

```bash
# Build only papers in collections containing "Methods"
python scripts/build_index.py --collection "Methods"

# Build only PhD-related papers
python scripts/build_index.py --collection "PhD"

# Combine with other options
python scripts/build_index.py --collection "Core" --limit 20 --dry-run
```

## Research Digest

Generate a digest summarizing newly indexed papers. The digest tracks which
papers have already been reported, so each run only covers new additions.

```bash
# Generate a digest of recent papers
python scripts/research_digest.py

# Include up to 20 papers, output as JSON
python scripts/research_digest.py --max-papers 20 --output-format json

# Preview without marking papers as processed
python scripts/research_digest.py --dry-run

# Verbose progress output
python scripts/research_digest.py --verbose
```

### Digest Options

| Option | Description |
| ------ | ----------- |
| `--index-dir PATH` | Path to index directory (default: data/index) |
| `--output-dir PATH` | Directory to save digest (default: data/out/digests) |
| `--output-format` | `markdown` or `json` (default: markdown) |
| `--max-papers N` | Maximum papers to include (default: 10) |
| `--dry-run` | Generate without marking papers as processed |
| `--verbose` | Show detailed progress |

## Discord Bot

LITRIS includes a Discord bot that exposes search functionality via slash
commands with embed formatting and button-based pagination.

### Prerequisites

1. Install the `discord.py` package: `pip install discord.py`
2. Create a Discord application at https://discord.com/developers/applications
3. Add a bot to the application and copy the token
4. Invite the bot to your server with the `applications.commands` scope

### Running the Bot

```bash
# Set token via environment variable
export DISCORD_BOT_TOKEN=your-token-here
python scripts/run_discord_bot.py

# Or pass token directly
python scripts/run_discord_bot.py --token your-token-here

# With debug logging
python scripts/run_discord_bot.py --log-level DEBUG
```

### Bot Options

| Option | Description |
| ------ | ----------- |
| `--token` | Discord bot token (default: `DISCORD_BOT_TOKEN` env var) |
| `--log-level` | Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO) |

### Slash Commands

| Command | Description |
| ------- | ----------- |
| `/search <query>` | Semantic search across indexed papers |
| `/paper <id>` | Get full details for a specific paper |
| `/similar <id>` | Find papers similar to a given paper |
| `/summary` | Get index statistics |

Search results are paginated with Previous/Next buttons.

## Web UI (Streamlit)

The Streamlit-based web interface provides a full-featured search workbench.

### Launching

```bash
python -m streamlit run scripts/web_ui.py
```

### Tabs

The UI is organized into four tabs:

- **Search** -- Semantic search with filters for year range, collections, item
  types, and chunk types. Includes quick-filter presets (Recent, Methods, etc.),
  sort options (relevance, year, title), a detail panel with full extraction data
  and PDF access, and one-click citation copy in APA, MLA, Chicago, or BibTeX.
  Results can be exported to CSV, BibTeX, or PDF.

- **Index Summary** -- Full overview of your indexed library with statistics on
  paper counts, collection distribution, and extraction coverage.

- **Research Questions** -- Generate research questions directly from the UI.
  Runs gap analysis on your index and uses an LLM to propose questions based on
  identified gaps.

- **Citation Network** -- Interactive graph visualization of citation
  relationships between papers in your library (requires `pyvis` and
  `networkx`).

### Keyboard Shortcuts

| Key | Action |
| --- | ------ |
| `j` / `k` | Navigate results up/down |
| `o` | Open detail panel |
| `Esc` | Close detail panel |

## Additional Build Options

The following `build_index.py` options were added in recent releases:

| Option | Description |
| ------ | ----------- |
| `--collection SUBSTR` | Filter to papers in collections matching this substring |
| `--parallel N` | Number of parallel extraction workers (CLI mode only) |
| `--summary-model MODEL` | Override model for summary extraction fields |
| `--methodology-model MODEL` | Override model for methodology extraction fields |
| `--no-cache` | Disable extraction caching (re-extract all papers) |
| `--clear-cache` | Clear extraction cache before running |
| `--use-subscription` | Use Claude subscription (Max/Pro) instead of API billing |
| `--dedupe-by-doi` | Skip papers with DOIs already in index |
| `--show-doi-overlap` | Analyze DOI overlap without processing |
| `--skip-similarity` | Skip pre-computed similarity pair generation |
| `--rebuild-embeddings` | Rebuild embeddings even if they exist |
| `--paper PAPER_ID` | Process only this paper (repeatable) |
| `--skip-paper PAPER_ID` | Skip specific paper ID (repeatable) |
| `--show-failed` | Show list of failed papers from previous run |
| `--show-skipped` | Show list of papers without PDFs |
| `--classify-only` | Run classification pre-pass only (no LLM extraction) |
| `--index-all` | Extract all papers regardless of classification |
| `--reclassify` | Force re-classification of all papers |
| `--source SOURCE` | Reference source (default: zotero) |
| `--source-path PATH` | Path for non-Zotero sources |

## Document Classification

LITRIS can classify papers by document type before extraction, filtering out
non-academic content (slides, syllabi, etc.) to save LLM costs.

### Classification Pre-Pass

Run classification without extraction to preview what would be extracted:

```bash
# Classify all papers and generate a report
python scripts/build_index.py --classify-only

# Force re-classification of all papers
python scripts/build_index.py --classify-only --reclassify

# Classify only papers in a specific collection
python scripts/build_index.py --classify-only --collection "ML Papers"
```

Classification results are saved to `data/index/classification_index.json`.

### Academic-Only Gating

By default, `build_index.py` filters to extractable papers (academic content
with sufficient text). Use `--index-all` to bypass this filter:

```bash
# Default: extract only academic papers
python scripts/build_index.py --limit 10

# Extract everything regardless of classification
python scripts/build_index.py --limit 10 --index-all
```

If no classification index exists, papers are classified inline during the
build. Running `--classify-only` first is recommended for large libraries.

### Extraction Cascade

Text extraction uses a multi-tier cascade, trying higher-fidelity methods
first and falling back to simpler ones:

1. **Companion** (.md file alongside PDF) -- pre-extracted markdown
2. **arXiv HTML** -- structured HTML from arXiv papers
3. **ar5iv** -- accessible HTML rendering of arXiv papers
4. **Marker** -- ML-based PDF to markdown conversion
5. **PyMuPDF** -- direct PDF text extraction
6. **OCR** -- Tesseract fallback for scanned documents

Cascade behavior is configured in `config.yaml` under `processing:`:

```yaml
processing:
  cascade_enabled: true
  companion_dir: null      # Optional directory for companion .md files
  arxiv_enabled: true
  marker_enabled: true
```

## Hook Scripts

LITRIS includes hook scripts in `scripts/hooks/` that integrate with Claude Code
for automated quality checks during development. These scripts are
cross-platform Python replacements for the original PowerShell hooks.

| Hook | Trigger | Purpose |
| ---- | ------- | ------- |
| `zotero_guard.py` | PreToolUse (Write/Edit) | Blocks writes to Zotero directories |
| `ruff_lint.py` | PostToolUse (Write/Edit) | Runs ruff on modified Python files |
| `lint_markdown.py` | PostToolUse (Write/Edit) | Lints modified Markdown files |
| `run_related_tests.py` | PostToolUse (Write/Edit) | Auto-runs tests for changed `src/` modules |
| `verify_task_completion.py` | PreToolUse (Edit) | Verifies task completion before edits |
| `operations_log.py` | PostToolUse (Bash/Write/Edit) | Logs operations to `operations.log` |
| `test_file_runner.py` | PostToolUse (Write/Edit) | Runs test files after modification |

Each script reads `CLAUDE_FILE_PATH` and `CLAUDE_TOOL_NAME` from environment
variables set by Claude Code. They are configured in `.claude/settings.json`.

## Next Steps

- See [Query Guide](query_guide.md) for search examples
- See [Troubleshooting](troubleshooting.md) for common issues
- See [guides/openai-integration.md](guides/openai-integration.md) for OpenAI/Codex setup
- See [guides/gemini-integration.md](guides/gemini-integration.md) for Google Gemini setup
- See [guides/alternative-sources.md](guides/alternative-sources.md) for BibTeX/PDF folder/Mendeley
