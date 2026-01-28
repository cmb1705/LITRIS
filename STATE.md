# LITRIS - Build State

**Project:** LITRIS (Literature Review Indexing System)
**Current Phase:** Complete (Active Development)
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
- **Streamlit Web UI** for interactive search and exploration
- **Multi-provider LLM support** (Anthropic, OpenAI, Google, Ollama, llama.cpp)
- **Multiple reference sources** (Zotero, BibTeX, PDF Folder, Mendeley, EndNote, Paperpile)
- **Docker containerization** for portable deployments

All implementation phases complete. System is production-ready for research workflows.

---

## Index Statistics

| Metric | Value |
|--------|-------|
| Total Papers | 332 |
| Total Chunks | 3,746 |
| Collections | 20+ Zotero collections |
| Item Types | Journal articles, books, preprints, conference papers |
| Year Range | 1945-2025 |
| Top Disciplines | Scientometrics, Network Science, Machine Learning |

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
| `/build` | Build/update the index from Zotero |
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
|   +-- analysis/         # PDF and LLM extraction
|   +-- indexing/         # Embeddings and storage
|   +-- query/            # Search interface
|   +-- mcp/              # MCP server and tools
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

---

## Future Expansion Ideas

### Near-term Enhancements

- **Full-text search**: Add keyword search alongside semantic search
- **Citation graph**: Build paper citation network from DOI references
- **Author profiles**: Aggregate papers by author for collaboration analysis
- **Topic modeling**: Automatic clustering of papers by research themes

### Advanced Features

- **Research gap detection**: Identify underexplored intersections in the literature
- **Trend forecasting**: Use temporal patterns to predict emerging topics
- **Collaborative annotations**: Share extractions and notes across researchers

### Integration Opportunities

- **Zotero plugin**: Direct integration with Zotero desktop
- **Browser extension**: One-click paper addition and extraction
- **API service**: REST API for programmatic access

### Recently Completed (January 2026)

- **Web interface**: Streamlit-based search workbench with export
- **Multi-database support**: Zotero, BibTeX, PDF Folder, Mendeley, EndNote, Paperpile
- **Similarity network visualization**: PyVis-powered interactive paper similarity graphs
- **Local LLM support**: Ollama and llama.cpp for offline extraction
- **CI/CD pipeline**: GitHub Actions with pre-commit hooks

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

Last Updated: 2026-01-28
