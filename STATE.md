# LITRIS - Build State

**Project:** LITRIS (Literature Review Indexing System)
**Current Phase:** Phase 5 Complete
**Status:** Production Ready

---

## Quick Reference

| Document | Location | Purpose |
|----------|----------|---------|
| Project Memory | [CLAUDE.md](CLAUDE.md) | Instructions for Claude Code |
| Technical Specification | [docs/proposals/mcp_technical_specification.md](docs/proposals/mcp_technical_specification.md) | MCP system design |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) | Common issues and solutions |
| Usage Guide | [docs/usage.md](docs/usage.md) | Workflows and configuration |

---

## Current Status

LITRIS is fully operational with comprehensive integrations:

- **332 papers** indexed from Zotero library
- **3,746 embedding chunks** in vector store
- Semantic search with metadata filtering
- Incremental update support
- MCP tools for Claude Code integration
- Citation verification agent for academic writing support
- **Streamlit Web UI** with Citation Network and Research Questions tabs
- **Multi-provider LLM support** (Anthropic, OpenAI, Google, Ollama, llama.cpp)
- **Multiple reference sources** (Zotero, BibTeX, PDF Folder, Mendeley, EndNote, Paperpile)
- **Docker containerization** for portable deployments
- **Citation graph** with DOI and title matching, node deduplication
- **Similarity graph** with pre-computed topical similarity
- **Gap detection** with confidence scoring pipeline
- **Research questions** generation from literature analysis
- **Clustering** for automatic paper grouping by theme
- **LLM Council** for multi-provider consensus with cost limits
- **Research digest** generation with paper starvation fix
- **Quality rating** integrated into extraction schema
- **Discord bot** for interactive search and formatting
- **RRF search** (Reciprocal Rank Fusion) for hybrid retrieval
- **Agentic search** for multi-step research queries

All five implementation phases complete. System is production-ready for research workflows.

---

## Index Statistics

| Metric | Value |
|--------|-------|
| Total Papers | ~1,746 |
| Total Chunks | ~19,166 (~17,074 in ChromaDB after dedup) |
| Collections | 20+ Zotero collections |
| Item Types | Journal articles, books, preprints, conference papers |
| Year Range | 1945-2025 |
| Top Disciplines | Scientometrics, Network Science, Machine Learning |
| Citation Edges | ~1,479 (title-match based) |
| Similarity Pairs | ~30,960 across 1,548 papers |
| Embedding Model | Qwen3-Embedding-8B Q8_0 via Ollama |

---

## MCP Integration (Complete)

All six MCP tools are implemented and working:

| Tool | Purpose | Status |
|------|---------|--------|
| `litris_search` | Semantic search with filters | Complete |
| `litris_get_paper` | Retrieve full paper extraction | Complete |
| `litris_similar` | Find similar papers | Complete |
| `litris_summary` | Index statistics | Complete |
| `litris_collections` | List Zotero collections | Complete |
| `litris_save_query` | Save search results to file | Complete |
| `litris_search_rrf` | Multi-query RRF search | Complete |
| `litris_search_agentic` | Multi-round search with gap analysis | Complete |
| `litris_deep_review` | Integrated literature review synthesis | Complete |
| `litris_clusters` | Topic clustering via UMAP + HDBSCAN | Complete |

---

## Claude Code Integration

### Custom Agents

| Agent | Purpose |
|-------|---------|
| `citation-verifier` | Verify citations, check APA formatting, cross-reference LITRIS |
| `code-reviewer` | Review code quality and standards |
| `literature-analyst` | Analyze papers and synthesize findings |

### Slash Commands

| Command | Purpose |
|---------|---------|
| `/search <query>` | Search the literature index |
| `/build` | Build/update the index from reference library |
| `/review-paper <id>` | Review extraction quality |

### Hooks

| Hook | Purpose |
|------|---------|
| PreToolUse (Write/Edit) | Block writes to Zotero directory |
| PostToolUse (Write/Edit) | Auto-format Python files with Ruff |
| PostToolUse (Bash/Write/Edit) | Log operations |
| SubagentStop | Validate citation-verifier output |

---

## File Structure

```text
LITRIS/
+-- src/
|   +-- zotero/           # Zotero database reader
|   +-- references/       # Multi-source adapter factory (Zotero, BibTeX, Mendeley, etc.)
|   +-- extraction/       # PDF extraction cascade, text cleaning, OCR
|   +-- analysis/         # LLM extraction, citation graph, clustering, gap detection, research digest
|   +-- indexing/         # Embeddings and storage
|   +-- query/            # Search interface (semantic, RRF, agentic)
|   +-- mcp/              # MCP server and tools
|   +-- discord_bot/      # Discord bot integration
+-- .claude/
|   +-- agents/           # Custom agents (citation-verifier, etc.)
|   +-- commands/         # Slash commands
|   +-- settings.json     # Hooks configuration
+-- scripts/              # CLI tools
+-- docs/                 # Documentation
+-- tests/                # Test suite
+-- data/                 # Index and cache (gitignored)
```

---

## Implementation History

### Phase 1: Core LITRIS (Complete)

- Zotero database integration (read-only)
- PDF text extraction with OCR fallback
- LLM-based paper analysis (CLI and Batch API modes)
- Semantic search with ChromaDB
- Incremental update support
- DOI-based deduplication

### Phase 2: MCP Integration (Complete)

- FastMCP server implementation
- Five semantic search tools
- Claude Code configuration
- Comprehensive test suite
- Documentation and troubleshooting guides

### Phase 3: Research Workflow Support (Complete)

- Citation verification agent with anti-hallucination protocol
- SubagentStop hook for output validation
- Query result auto-save and PDF export
- Cross-reference capabilities against index

### Phase 4: Advanced Analysis (Complete)

- **Citation graph**: DOI and title matching pipeline with node deduplication
- **Similarity graph**: Pre-computed topical similarity network (PyVis visualization)
- **Gap detection**: Confidence scoring pipeline for underexplored research areas
- **Research questions**: Automatic generation from literature analysis
- **Clustering**: Automatic paper grouping by research themes
- **Quality rating**: Integrated into extraction schema and search results

### Phase 5: Multi-Provider and Deployment (Complete)

- **LLM Council**: Multi-provider consensus with per-provider cost limits and timeout handling
- **Research digest**: Generation module with paper starvation fix and atomic state writes
- **Discord bot**: Interactive search with rich formatting
- **RRF search**: Reciprocal Rank Fusion for hybrid retrieval
- **Agentic search**: Multi-step research query execution
- **Web UI tabs**: Citation Network and Research Questions tabs in Streamlit
- **Reference list parsing**: Citation graph ground truth from parsed references
- **Cross-platform hooks**: Python-based pre-commit hooks (replaces PowerShell)
- **Collection filtering**: `--collection` flag for selective Zotero indexing

---

## Remaining Work

### Requires User Infrastructure

- **NAS deployment** (li-33k): Docker/systemd deployment to NAS hardware — needs NAS access
- **Open WebUI + MCP integration** (li-6s7): Blocked by NAS deployment

### Future Expansion Ideas

### Integration Opportunities

- **Zotero plugin**: Direct integration with Zotero desktop
- **Browser extension**: One-click paper addition and extraction
- **API service**: REST API for programmatic access
- **Trend forecasting**: Use temporal patterns to predict emerging topics
- **Collaborative annotations**: Share extractions and notes across researchers
- **Author profiles**: Aggregate papers by author for collaboration analysis

### Completed (January 2026)

- **Web interface**: Streamlit-based search workbench with export
- **Multi-database support**: Zotero, BibTeX, PDF Folder, Mendeley, EndNote, Paperpile
- **Similarity network visualization**: PyVis-powered interactive paper similarity graphs
- **Local LLM support**: Ollama and llama.cpp for offline extraction
- **CI/CD pipeline**: GitHub Actions with pre-commit hooks

### Completed (February 2026)

- **Non-publication filtering**: Skip fragments, forms, and notes based on word/page count and section markers
- **OCR fallback on fail**: Automatically retry with OCR when initial text extraction produces poor results
- **Skipped items reporting**: Generate JSON/CSV reports for items skipped during extraction
- **OpenAI/Codex CLI support**: Full provider comparison with robust enum normalization for GPT responses
- **Provider comparison script**: Side-by-side extraction quality comparison between Anthropic and OpenAI
- **Improved prompt schema**: Explicit enum rules to improve LLM compliance with expected values
- **Windows compatibility**: Fixed Unicode encoding in Codex CLI, WinGet Poppler path detection

### Completed (March 2026)

- **Citation graph pipeline**: DOI and title matching with node deduplication
- **Pre-computed similarity graph**: Topical similarity with containment matching
- **Gap detection scoring**: Confidence-scored research gap identification
- **Research questions tab**: Web UI tab for generated research questions
- **Citation network tab**: Web UI tab for interactive citation graph
- **LLM Council**: Multi-provider consensus with cost limits and integration tests
- **Research digest**: Generation with paper starvation fix and silent failure hardening
- **Cross-platform hooks**: Python scripts replacing PowerShell for Linux compatibility
- **Package updates**: Requirements lock updated to latest versions
- **Extraction cascade**: Multi-tier PDF extraction (Companion, arXiv, Marker, PyMuPDF, OCR)
- **Document classification**: Persistent classification index with extractability gating
- **Adapter factory**: Unified reference source interface via `--source` flag
- **Academic-only gating**: Automatic filtering of non-academic content before LLM extraction

---

## Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| mcp | MCP SDK for Python | 1.23.1+ |
| chromadb | Vector store | 0.4+ |
| sentence-transformers | Embeddings | 2.2+ |
| anthropic | Claude API | 0.39+ |
| pymupdf | PDF extraction | 1.23+ |

---

Last Updated: 2026-03-14
