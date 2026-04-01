# Literature Review Indexing System (LITRIS)

LITRIS builds a searchable literature index from your academic library. It
extracts structured paper summaries with LLMs, stores canonical full-text
snapshots for retrieval, generates embeddings for semantic search, and exposes
the resulting index to local scripts, a Streamlit UI, and MCP tools.

## Overview

LITRIS is designed for a workflow where you:

1. ingest references from Zotero, BibTeX, PDF folders, or other supported
   sources
2. extract and clean paper text
3. run semantic analysis into profile-defined dimensions
4. build embeddings and a Chroma-backed search index
5. query the index directly or through an MCP-connected coding agent

Current capabilities include:

- semantic search, RRF search, and agentic search
- portable semantic dimension profiles with migration, diff, suggestion, and
  backfill tooling
- canonical full-text snapshot storage for verbatim retrieval
- citation graph and similarity graph generation
- gap analysis, research digest generation, and research-question generation
- MCP tools for search, deep review, paper lookup, and full-text context lookup

## Architecture

```text
Reference Source --> Paper Records --> Full-Text Extraction --> Semantic Extraction
(Zotero/BibTeX/PDF Folder/etc.)          |                    (profile-driven)
                                         v
                               Canonical Full-Text Snapshots
                                         |
                                         v
                         Embedding Generation --> Chroma Vector Store
                                         |
                                         v
                     Search / MCP / Web UI / Digest / Gap Analysis / Review
```

## Text Extraction And Storage

The text-extraction stack is layered. Depending on the paper and what is
available, LITRIS can use:

- companion Markdown sidecars
- arXiv HTML
- ar5iv
- OpenDataLoader
- PyMuPDF
- Marker
- OCR fallback for low-quality PDFs

Canonical full-text snapshots are stored in `data/index/fulltext/`, with
metadata in `data/index/fulltext_manifest.json`. These snapshots are preserved
as full cleaned text for retrieval. LLM input may be truncated for provider
limits, but the stored snapshot is not truncated for search or quote retrieval.

## Current Workflow Support

The main entrypoints do not all expose the same provider surface area.

Primary indexing:

- `scripts/build_index.py` currently exposes Anthropic and OpenAI for the main
  extraction workflow
- CLI mode supports Claude Code CLI and Codex CLI where applicable
- Anthropic batch mode is available through `--mode batch_api`

Dimension tooling:

- `scripts/dimensions.py backfill` currently exposes Anthropic, OpenAI, and
  Google
- `scripts/dimensions.py suggest` can use Anthropic, OpenAI, Google, Ollama,
  or llama.cpp where configured

Embeddings:

- the default backend is `sentence-transformers`
- Ollama embeddings are optional, not required

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/LITRIS.git
cd LITRIS

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
copy config.example.yaml config.yaml
```

Minimum prerequisites:

- Python 3.10+
- one supported reference source
- one supported extraction path
  - Claude CLI for Anthropic subscription mode
  - Codex CLI for OpenAI CLI mode
  - or API access for supported providers
- optional OCR tooling if you want scanned-PDF recovery

## Configuration

The default embedding setup is CPU-friendly and does not require Ollama.

```yaml
extraction:
  provider: "anthropic"
  mode: "cli"
  model: ""

dimensions:
  active_profile: "legacy_semantic_v1"
  profile_paths: []
  approval_required: true
  suggestion_sample_size: 25
  suggestion_max_proposals: 5
  suggestion_neighbor_count: 3
  suggestion_use_llm: true

embeddings:
  backend: "sentence-transformers"
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
  batch_size: auto
```

If you want a stronger local embedding model and are willing to manage Ollama,
you can switch the embedding backend to `ollama` and point `model` at an Ollama
embedding model such as Qwen3. That is optional. The baseline
`sentence-transformers/all-MiniLM-L6-v2` model remains the default because it is
lightweight, broadly available, and good enough for most users.

See [config.example.yaml](config.example.yaml) for the full configuration
surface.

## Quick Start

Anthropic subscription example:

```bash
python scripts/build_index.py --explain-plan --dry-run
python scripts/build_index.py --limit 10 --use-subscription
python scripts/query_index.py -q "network analysis methods"
```

OpenAI CLI example:

```bash
python scripts/build_index.py --provider openai --mode cli --limit 10
```

Useful build flags:

- `--sync-mode auto`: default safe mode
- `--sync-mode full`: full vector-store rebuild
- `--sync-mode update`: fail if an incremental update is not safe
- `--paper`: target one or more specific papers
- `--refresh-text`: re-run source text extraction instead of reusing stored
  canonical fulltext
- `--skip-extraction`: reuse existing semantic records
- `--skip-embeddings`: avoid rebuilding the vector store
- `--estimate-cost`: estimate extraction cost instead of running

## Reference Sources

`scripts/build_index.py` supports these source selectors:

| Source | Example |
| ------ | ------- |
| Zotero | `--source zotero` |
| BibTeX | `--source bibtex --source-path ./library.bib` |
| PDF folder | `--source pdffolder --source-path ./papers/` |
| Mendeley | `--source mendeley --source-path ./mendeley.sqlite` |
| EndNote | `--source endnote --source-path ./MyLibrary.enl` |
| Paperpile | `--source paperpile --source-path ./paperpile.json` |

See [docs/guides/alternative-sources.md](docs/guides/alternative-sources.md)
for source-specific setup notes.

## Search And Retrieval

Basic search:

```bash
python scripts/query_index.py -q "citation network analysis"
python scripts/query_index.py -q "research methods" --year-min 2020
python scripts/query_index.py -q "graph methods" --chunk-types abstract dim_q07
python scripts/query_index.py --paper PAPER_ID
python scripts/query_index.py --similar PAPER_ID
python scripts/query_index.py --summary
```

Search output is saved to `data/query_results/` by default unless you pass
`--no-save`.

## Portable Dimension Profiles

Each index now carries an active semantic profile snapshot in
`data/index/dimension_profile.json`. Existing indexes without a profile snapshot
fall back to the built-in `legacy_semantic_v1` profile, which preserves the
historical `q01` through `q40` aliases and `dim_qNN` chunk types.

The profile management workflow lives in `scripts/dimensions.py`:

```bash
python scripts/dimensions.py migrate-store --index-dir data/index --dry-run
python scripts/dimensions.py migrate-store --index-dir data/index
python scripts/dimensions.py diff --index-dir data/index --new-profile ./profiles/custom.yaml
python scripts/dimensions.py backfill --index-dir data/index \
  --dimension-profile ./profiles/custom.yaml --dry-run
```

Important current behavior:

- `backfill` re-extracts only the sections touched by added, reworded, or
  replaced dimensions
- disable-only profile changes do not trigger LLM work
- `--fulltext-only` captures canonical full-text snapshots without calling an
  LLM provider
- `--resume` resumes interrupted backfill and full-text runs using checkpoints
- `--refresh-text` re-runs full-text extraction only for the targeted scope

See [docs/dimension_profiles.md](docs/dimension_profiles.md) and
[docs/guides/portable-dimensions-live-test.md](docs/guides/portable-dimensions-live-test.md)
for the operator workflow.

## MCP Integration

LITRIS exposes MCP tools for agent-driven research workflows:

| Tool | Purpose |
| ---- | ------- |
| `litris_search` | semantic search with filters |
| `litris_search_rrf` | reciprocal-rank-fusion search |
| `litris_search_agentic` | iterative search with follow-up rounds |
| `litris_deep_review` | integrated topic review |
| `litris_get_paper` | paper metadata plus extraction payload |
| `litris_get_fulltext_context` | verbatim context lookup from stored full text |
| `litris_similar` | similar-paper lookup |
| `litris_clusters` | topic clustering |
| `litris_summary` | index statistics |
| `litris_collections` | collection listing |
| `litris_save_query` | save a query report |
| `litris_search_dimension` | search a single semantic dimension |
| `litris_search_group` | search a profile section/group |

Example `.mcp.json`:

```json
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

See [docs/mcp_troubleshooting.md](docs/mcp_troubleshooting.md) for setup and
diagnostics.

## Web UI

Launch the local search workbench with Streamlit:

```bash
python -m streamlit run scripts/web_ui.py
```

The UI expects an existing index in `data/index`. It supports:

- semantic search
- metadata and chunk-type filters
- paper detail panels
- citation copying
- export to CSV, BibTeX, and PDF
- similarity-network visualization when visualization dependencies are installed

## Additional Analysis Tools

Examples:

```bash
python scripts/gap_analysis.py
python scripts/research_questions.py --provider anthropic --scope narrow
python scripts/research_digest.py --dry-run
python scripts/council_comparison.py --keys PAPER_KEY1 PAPER_KEY2 --save
python scripts/run_discord_bot.py
```

The legacy `scripts/update_index.py` wrapper still exists for compatibility,
but new workflows should use `scripts/build_index.py`.

## Docker

The repo includes `Dockerfile` and `docker-compose.yml` for containerized API
workflows. CLI-based extraction modes are typically simpler on the host because
they require local CLI authentication.

Basic flow:

```bash
copy config.example.yaml config.yaml
copy .env.example .env
docker compose build
docker compose run --rm litris python scripts/build_index.py --dry-run
docker compose run --rm litris python scripts/query_index.py \
  -q "network analysis methods"
```

## Development

Relevant files:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/dimension_profiles.md](docs/dimension_profiles.md)
- [docs/guides/openai-integration.md](docs/guides/openai-integration.md)
- [docs/guides/gemini-integration.md](docs/guides/gemini-integration.md)
- [docs/guides/local-llm-integration.md](docs/guides/local-llm-integration.md)
- [docs/sphinx/_build/html/index.html](docs/sphinx/_build/html/index.html)

CI currently runs:

- `ruff`
- `pytest`
- `mypy`
- `pip-audit`

Pre-commit hooks are defined in
[.pre-commit-config.yaml](.pre-commit-config.yaml).

## Cost And Provider Notes

Hard-coded price tables go stale quickly. Use
`python scripts/build_index.py --estimate-cost` against your current config
instead of relying on static README pricing.

## License

MIT
