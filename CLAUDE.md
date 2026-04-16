# Project Memory: Literature Review Indexing System (LITRIS)

> Extended reference: [CLAUDE_SUPPLEMENTAL.md](CLAUDE_SUPPLEMENTAL.md)

## Project Overview

**Purpose**: AI-assisted literature review system for Zotero libraries with LLM extraction and semantic search.

**Name**: LITRIS (Literature Review Indexing System)

## Development Environment

- **Python**: 3.10+ with .venv
- **Dependencies**: anthropic, pymupdf, chromadb, sentence-transformers, pydantic

## Quick Start

```bash
pip install -r requirements.txt
python scripts/build_index.py --limit 5  # Test extraction on 5 papers
python scripts/query_index.py -q "network analysis"  # Search the index
pytest tests/ -v --tb=short  # Run test suite
```

## Git Workflow

- No attribution lines ("Generated with Claude Code", "Co-Authored-By")
- Branch naming: `feature/`, `fix/`, `docs/`
- Clear, concise commit messages

## Code Quality Standards

- **No emojis** in code, comments, docs, or commits
- PEP 8 with type hints; docstrings for public APIs
- Ruff linting (auto-applied via hook)
- Markdown: language specifiers, blank lines around blocks

## Data Protection

### NEVER Modify

- Zotero database and storage directories (read-only)
- `data/raw/` - Immutable if created

### Zotero Database Rules

- **ALWAYS** open read-only: `file:{path}?mode=ro`
- **NEVER** INSERT, UPDATE, DELETE, or modify storage files

### Allowed Modifications

- `data/index/`, `data/cache/`, `data/logs/`, `data/query_results/`

## Agent Workflow

- **Start new projects in plan mode** - When beginning work on a new project or major feature, enter plan mode first to design the approach before implementation
- Bias for action on clear tasks
- Use beads (`bd create`) for 3+ step operations - enables tracking across all agents and developers
- Verify outputs before reporting completion
- Be direct; report exact paths and metrics

## Task Tracking

- Use beads (`bd`) for multi-step tracking (preferred)
- Use TodoWrite only for quick single-session visibility
- Task specs in `docs/proposals/tasks/*.md`

## Key Files

| File | Purpose |
|------|---------|
| STATE.md | Status tracker |
| config.yaml | Configuration |
| docs/proposals/*.md | Specs and plans |
| Local Claude Code config repo | Machine-specific shared config (settings, plugins, skills) |

## Custom Agents

| Agent | Purpose |
|-------|---------|
| principal-investigator | Research direction, major decisions |
| literature-analyst | Paper analysis, extraction validation |
| pipeline-engineer | Infrastructure, performance |
| query-specialist | Search optimization |
| code-reviewer | Code quality |
| citation-verifier | Citation accuracy, APA formatting, anti-hallucination |
| extraction-reviewer | LLM extraction quality vs source PDFs |
| extraction-comparator | Multi-provider extraction quality diffs |

## Custom Skills

| Skill | Purpose |
|-------|---------|
| academic-extraction | Paper structure, extraction guidelines |
| citation-formatting | APA/MLA/Chicago/BibTeX |
| semantic-search | Query formulation, results interpretation |
| verification | Task completion checklists, quality gates |
| provider-benchmark | Side-by-side Anthropic vs OpenAI extraction comparison |
| index-health | Unified index quality check (validation + gaps + preflight) |
| smoketest | Targeted integration smoketests by keyword |
| thematic-comparison | Cross-paper thematic analysis with verbatim quotes |

## Custom Commands

| Command | Purpose |
|---------|---------|
| /build | Run index pipeline |
| /search | Execute search |
| /review-paper | Analyze extraction |

## Hooks

| Hook | Action |
|------|--------|
| PreToolUse (Write/Edit) | Block Zotero directory writes |
| PostToolUse (Write/Edit) | Ruff on .py files |
| PostToolUse (Write/Edit) | Auto-run related tests on src/ changes |
| PostToolUse (Write/Edit) | Lint .md files via markdownlint-cli2 |
| PreToolUse (Edit) | Verify task completion before edits |
| PostToolUse (Bash/Write/Edit) | Log to operations.log |

## Report Standards

All LITRIS reports (saved via `litris_save_query`, `litris_deep_review`, or manual write to `data/query_results/`) must be self-contained documents that any reader can understand without prior context.

### Required Structure

```markdown
# [Report Title]

**Generated**: YYYY-MM-DD | **Index**: [paper count] papers, [chunk count] chunks | ...

---

## Search Query

> **Part 1:** [exact original query text]
>
> **Part 2:** [exact original query text]

---

## [Analysis sections...]
```

### Rules

1. **Include the exact query**: The full original query/prompt must appear verbatim in a "Search Query" section immediately after the metadata header, before any analysis. Use blockquotes with part labels.
2. **Metadata header**: Include generation date, index size, and search method summary.
3. **Paper references**: Include LITRIS paper_id (e.g., `ZLBNMNC5`) alongside author-year citations for traceability back to the index.
4. **Self-contained**: A reader unfamiliar with the original conversation must be able to understand what questions the report answers and why.

### Reference

See `data/query_results/2026-01-22_hgt-citation-network-anomaly-detection.md` for the canonical example.

## MCP Integration

LITRIS exposes semantic search as MCP tools for Claude Code.

### MCP Tools

| Tool | Purpose |
|------|---------|
| litris_search | Semantic search with filters |
| litris_search_rrf | Multi-query RRF search for improved recall |
| litris_search_agentic | Multi-round search with gap analysis |
| litris_deep_review | Integrated literature review synthesis |
| litris_clusters | Topic clustering via UMAP + HDBSCAN |
| litris_get_paper | Full paper details |
| litris_similar | Find similar papers |
| litris_summary | Index statistics |
| litris_collections | List collections |
| litris_save_query | Save query results to data/query_results/ |

### Tool Naming

When invoked from Claude Code, tools are prefixed: `mcp__litris__litris_search`

### Server Startup

```bash
python -m src.mcp
```

### Configuration

MCP server is defined in `.mcp.json` (project root):

```json
{
  "mcpServers": {
    "litris": {
      "command": "python",
      "args": ["-m", "src.mcp"],
      "cwd": "path/to/litris"
    }
  }
}
```

Enable in `.claude/settings.json`:

```json
{
  "enableAllProjectMcpServers": true
}
```

### Logging

- Log file: `data/logs/mcp_server.log`
- Log level: Set via `MCP_LOG_LEVEL` env var (default: INFO)

## Security

- API keys in environment variables only
- Never commit .env or config.yaml with real paths
- Validate paths; block protected directories

## Development Notes

### Windows Compatibility

- **Plugin marketplace**: Canonical source is `anthropics-claude-plugins-official` (not `claude-plugins-official`); all `enabledPlugins` references must use exact directory name
- **Subprocess encoding**: Always use `encoding="utf-8"` and `errors="replace"` in `subprocess.run()` to handle Unicode characters (e.g., ligatures like '\ufb01')
- **Hook syntax**: Use `2>NUL` instead of `2>/dev/null || true` in Windows/PowerShell hooks
- **Hook caching**: Claude Code caches hooks; restart required after `.claude/settings.json` changes
- **PATH propagation**: Newly installed CLI tools need process restart; use full path as workaround (e.g., `"C:\Program Files\GitHub CLI\gh.exe"`)

### Testing

- **Mock aliased imports**: When module uses `from x import Y as _Y`, mock `module._Y` not `module.Y` (see `test_embeddings.py`)
- **Mock deferred imports at the factory**: When a function does `from src.analysis.llm_factory import create_llm_client` inside its body, mock `src.analysis.llm_factory.create_llm_client`, NOT the calling module
- **CI status**: Use `gh run list` and `gh run view <id>` to check GitHub Actions

### Forward References and Circular Imports

- Use `from __future__ import annotations` + `TYPE_CHECKING` guard for type annotations that would cause circular imports
- Pattern: import at top under `if TYPE_CHECKING:`, use bare name in annotations (no quotes needed with `__future__` annotations)
- Used in: `search.py` (AgenticSearchResult), `embeddings.py` (RaptorSummaries)

### CLI Extraction Architecture

- `cli_executor.extract(prompt, text)` - Separates prompt (-p flag) from text (stdin); used for paper extraction
- `call_with_prompt(combined)` - Sends everything as single prompt; different behavior
- When debugging extraction issues, verify which method is being called

### Multi-Provider Support

- OpenAI/GPT responses may include explanatory text in enum values (e.g., `"citation (literature support)"`)
- Enum normalization in `section_extractor.py` handles this automatically
- Provider comparison: `scripts/compare_providers.py --key <zotero_key>`

---

**See [CLAUDE_SUPPLEMENTAL.md](CLAUDE_SUPPLEMENTAL.md) for:**
Pipeline Architecture, Extraction Schema, API Usage Details, Common Operations,
Verification Standards, Error Handling, Domain Context, Performance Targets,
Development Workflow (8-step), Implementation Standards

<!-- BEADS-RALPH-INTEGRATION-START -->
## Beads Integration

This project uses **beads** for persistent issue tracking with **ralph-loop** compatibility.

### Context Recovery

After compaction, `bd prime` runs automatically via SessionStart hook. If context seems missing:

```bash
bd prime
```

### Core Workflow

1. **Start**: `bd ready` - Check available work
2. **Claim**: `bd update <id> --status=in_progress`
3. **Complete**: `bd close <id>`
4. **Sync**: `bd sync --flush-only`

### TodoWrite vs Beads

- **Beads** (preferred): Persistent tracking across agents and developers; use for 3+ step operations
- **TodoWrite**: Quick single-session visibility for simple immediate tasks only

### Session Close Protocol

Before stopping:
1. Update beads status for tasks worked on
2. Create issues for discovered work
3. Run `bd dolt push`
4. Verify with `bd ready`
<!-- BEADS-RALPH-INTEGRATION-END -->
