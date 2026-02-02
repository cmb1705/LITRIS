# Project Memory: Literature Review Indexing System (LITRIS)

> Extended reference: [CLAUDE_SUPPLEMENTAL.md](CLAUDE_SUPPLEMENTAL.md)

## Project Overview

**Purpose**: AI-assisted literature review system for Zotero libraries with LLM extraction and semantic search.

**Name**: LITRIS (Literature Review Indexing System)

## Development Environment

- **Python**: 3.10+ with .venv
- **Dependencies**: anthropic, pymupdf, chromadb, sentence-transformers, pydantic

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

- Track progress in [STATE.md](STATE.md)
- Use TodoWrite for complex tasks
- Task specs in `docs/proposals/tasks/*.md`

## Key Files

| File | Purpose |
|------|---------|
| STATE.md | Status tracker |
| config.yaml | Configuration |
| docs/proposals/*.md | Specs and plans |

## Custom Agents

| Agent | Purpose |
|-------|---------|
| principal-investigator | Research direction, major decisions |
| literature-analyst | Paper analysis, extraction validation |
| pipeline-engineer | Infrastructure, performance |
| query-specialist | Search optimization |
| code-reviewer | Code quality |

## Custom Skills

| Skill | Purpose |
|-------|---------|
| academic-extraction | Paper structure, extraction guidelines |
| citation-formatting | APA/MLA/Chicago/BibTeX |
| semantic-search | Query formulation, results interpretation |

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
| PostToolUse (Bash/Write/Edit) | Log to operations.log |

## MCP Integration

LITRIS exposes semantic search as MCP tools for Claude Code.

### MCP Tools

| Tool | Purpose |
|------|---------|
| litris_search | Semantic search with filters |
| litris_get_paper | Full paper details |
| litris_similar | Find similar papers |
| litris_summary | Index statistics |
| litris_collections | List collections |
| litris_save_query | Save query results to query_results/ |

### Tool Naming

When invoked from Claude Code, tools are prefixed: `mcp__litris__litris_search`

### Server Startup

```bash
python -m src.mcp.server
```

### Configuration

MCP server is defined in `.mcp.json` (project root):

```json
{
  "mcpServers": {
    "litris": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
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
3. Run `bd sync --flush-only`
4. Verify with `bd status`
<!-- BEADS-RALPH-INTEGRATION-END -->
