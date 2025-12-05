# Project Memory: Literature Review Index System

## Project Overview

**Title**: Zotero Literature Review Index System

**Purpose**: Build an AI-assisted literature review system that indexes academic papers from a Zotero library, extracts structured insights using LLM analysis, and enables semantic search for research synthesis and gap identification.

**Core Capabilities**:
- Extract metadata and PDFs from Zotero SQLite database
- Use Claude to extract thesis, methodology, findings, limitations, and future directions
- Generate embeddings for semantic similarity search
- Enable literature review, citation support, and gap analysis

## Development Environment

- **OS**: Windows (PowerShell for shell commands)
- **Python**: 3.10+ with virtual environment
- **Key Dependencies**: anthropic, pymupdf, chromadb, sentence-transformers, pydantic
- **Zotero Location**: D:\Zotero\ (read-only access)
- **Project Location**: D:\Git_Repos\Lit_Review\

## Git Workflow

- Never add attribution lines to git commits
- No "Generated with Claude Code" or "Co-Authored-By: Claude" lines
- Branch naming: `feature/description`, `fix/description`, `docs/description`
- Commit messages: Clear, concise descriptions of changes

## Code Quality Standards

### Professional Presentation

- **Never use emojis** in code, comments, documentation, or commit messages
- Maintain professional, academic tone suitable for research use
- All outputs should be publication-quality

### Python Standards

- PEP 8 compliance with type hints for public functions
- Docstrings required for all public functions, classes, and modules
- Use snake_case for naming conventions
- Linting with ruff (auto-applied via hook)

### Markdown Standards

- Language specifiers on all code blocks
- Blank lines around code blocks and lists
- Single H1 header per document

## Data Protection

### Protected Directories (NEVER Modify)

- `D:\Zotero\` - Zotero database and storage (read-only)
- `data/raw/` - If created, treat as immutable

### Allowed Modifications

- `data/index/` - Index outputs (papers.json, extractions.json, ChromaDB)
- `data/cache/` - Extracted text cache
- `data/logs/` - Operation logs
- `data/query_results/` - Search outputs

### Zotero Database Rules

- **ALWAYS** open in read-only mode
- **NEVER** execute INSERT, UPDATE, or DELETE
- **NEVER** modify files in D:\Zotero\storage\
- Use connection string: `file:{path}?mode=ro`

## Pipeline Architecture

### Processing Stages

1. **Zotero Scan**: Extract metadata and PDF paths from SQLite
2. **PDF Extraction**: Extract text using PyMuPDF (OCR fallback with Tesseract)
3. **LLM Extraction**: Use Claude Opus for structured section extraction
4. **Embedding Generation**: Create vectors with sentence-transformers
5. **Indexing**: Store in ChromaDB (vectors) and JSON (structured data)
6. **Query**: Semantic search with metadata filtering

### Key Data Files

| File | Contents |
|------|----------|
| data/index/papers.json | Paper metadata from Zotero |
| data/index/extractions.json | LLM extraction results |
| data/index/metadata.json | Index state and statistics |
| data/index/summary.json | Corpus overview for quick reference |
| data/index/chroma/ | ChromaDB vector store |

### Extraction Schema

Each paper extraction includes:
- thesis_statement
- research_questions
- theoretical_framework
- methodology (approach, design, data_sources, analysis_methods)
- key_findings (list with evidence type and significance)
- conclusions
- limitations
- future_directions
- key_claims (with support type and page reference)
- contribution_summary
- discipline_tags
- extraction_confidence

## Agent Workflow Preferences

### Execution Philosophy

- Bias for action: If task is clear, execute without hesitation
- Plan carefully for multi-step tasks
- Use TodoWrite for complex operations (3+ steps)
- Verify outputs before reporting completion

### Communication Style

- Be direct and efficient
- Report exact file paths and metrics
- Cite specific sources for claims
- Use tables for structured information

### Quality Gates

- **Code changes**: Run ruff before commit
- **Extractions**: Validate against schema
- **Queries**: Verify results are relevant
- **Index updates**: Check for data consistency

## API Usage

### Extraction Modes

Two extraction modes are supported:

| Mode | Cost | Speed | Authentication |
|------|------|-------|----------------|
| **CLI** | Free (Max subscription) | ~30s/paper sequential | `claude login` |
| **Batch API** | ~$0.14/paper (50% off) | ~1hr for 500 parallel | `ANTHROPIC_API_KEY` |

### CLI Mode (Recommended for Budget)

Uses Claude Code CLI in headless mode with your Max subscription.

```powershell
# Ensure no API key (triggers billing)
$env:ANTHROPIC_API_KEY = $null

# Run extraction
python scripts/build_index.py --mode cli
```

**Rate Limits (Max 20):** 200-800 prompts per 5-hour window

### Batch API Mode (Recommended for Speed)

Uses Anthropic Message Batches API with 50% discount.

```powershell
# Set API key
$env:ANTHROPIC_API_KEY = "your-api-key"

# Run extraction
python scripts/build_index.py --mode batch_api
```

**Installation:**

```bash
pip install anthropic
```

### Embedding Model

- Default: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- Store model name with index for compatibility checks
- Batch embedding generation for efficiency

## Task Tracking

- Use TodoWrite for tasks with 3+ steps
- Track progress in STATE.md
- Update task files in proposals/tasks/ as completed
- Log failures to metadata.json for retry

## Key Project Files

| File | Purpose |
|------|---------|
| STATE.md | Project status and task tracker |
| proposals/technical_specification.md | Full system design |
| proposals/project_plan.md | Implementation TODO lists |
| proposals/tasks/*.md | Detailed task specifications |
| config.yaml | System configuration |
| .claude/settings.json | Hooks configuration |

## Common Operations

### Building the Index

```bash
python scripts/build_index.py --limit 10  # Test build
python scripts/build_index.py              # Full build
python scripts/build_index.py --resume     # Resume from checkpoint
```

### Querying the Index

```bash
python scripts/query_index.py -q "search query"
python scripts/query_index.py -q "query" --year-min 2020
python scripts/query_index.py -q "query" -f markdown
```

### Updating the Index

```bash
python scripts/update_index.py --detect-only  # Check for changes
python scripts/update_index.py                 # Apply updates
```

## Verification Standards

### Extraction Quality

- Confidence score > 0.7 for reliable extractions
- Flag low-confidence extractions for review
- Cross-reference thesis with abstract

### Search Quality

- Relevance score > 0.7 for top results
- Verify chunk types match query intent
- Check for expected papers in results

### Data Integrity

- Validate JSON against Pydantic schemas
- Check embedding dimensions match model
- Verify ChromaDB document counts

## Error Handling

### Graceful Degradation

- Skip failed PDFs, log for retry
- Partial extractions better than none
- Continue processing on single failures

### Retry Strategy

- API failures: 3 retries with exponential backoff
- PDF failures: OCR fallback, then skip
- Embedding failures: Log and continue

### Failure Tracking

Track in metadata.json:
- paper_id
- failure stage
- error message
- retry attempts

## Security Considerations

### API Keys

- Store in environment variables only
- Never commit .env file
- Use .env.example as template

### File Access

- Validate paths before operations
- Block writes to protected directories
- Log all file operations

## Domain Context

### Research Areas in Corpus

Based on Zotero collections:
- Network Analysis (citation networks, graph methods)
- Scientometrics (bibliometrics, research metrics)
- Innovation Policy (S&T policy, technology assessment)
- Complexity Theory (systems thinking, emergence)

### Paper Types

| Type | Count | Notes |
|------|-------|-------|
| Journal Articles | 518 | Primary source |
| Books | 52 | Theoretical frameworks |
| Preprints | 27 | Cutting-edge findings |
| Conference Papers | 18 | Novel contributions |
| Theses | 3 | Comprehensive analyses |

## Performance Targets

### Processing Speed

- PDF extraction: < 5 seconds per paper
- LLM extraction: ~30 seconds per paper (API dependent)
- Embedding: < 1 second per paper (batched)

### Query Performance

- Search: < 1 second for 500-paper corpus
- Filter: Minimal additional latency
- Full retrieval: < 5 seconds

### Cost Targets

| Mode | Test (10 papers) | Full (500 papers) | Notes |
|------|------------------|-------------------|-------|
| CLI | $0 | $0 | Uses Max subscription |
| Batch API | ~$1.35 | ~$67.50 | 50% batch discount |

- CLI mode: Free within Max subscription limits
- Batch API: Pay-per-token with 50% batch discount
- Updates: Incremental cost only (CLI recommended)
