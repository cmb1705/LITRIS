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

- Bias for action on clear tasks
- Use TodoWrite for 3+ step operations
- Verify outputs before reporting completion
- Be direct; report exact paths and metrics

## Task Tracking

- Track progress in [STATE.md](STATE.md)
- Use TodoWrite for complex tasks
- Task specs in `proposals/tasks/*.md`

## Key Files

| File | Purpose |
|------|---------|
| STATE.md | Status tracker |
| config.yaml | Configuration |
| proposals/*.md | Specs and plans |

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

## Security

- API keys in environment variables only
- Never commit .env or config.yaml with real paths
- Validate paths; block protected directories

---

**See [CLAUDE_SUPPLEMENTAL.md](CLAUDE_SUPPLEMENTAL.md) for:**
Pipeline Architecture, Extraction Schema, API Usage Details, Common Operations,
Verification Standards, Error Handling, Domain Context, Performance Targets,
Development Workflow (8-step), Implementation Standards
