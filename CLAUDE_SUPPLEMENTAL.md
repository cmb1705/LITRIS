# CLAUDE_SUPPLEMENTAL.md - Extended Reference

This file contains detailed reference information for the Literature Review Index System.
For core project guidelines, see [CLAUDE.md](CLAUDE.md).

---

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

---

## API Usage

### Extraction Modes

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

### Embedding Model

- Default: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- Store model name with index for compatibility checks
- Batch embedding generation for efficiency

---

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

---

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

---

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

---

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

---

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

---

## Standard Development Workflow

For each task implementation, follow this progression:

### 1. Define Scope and Objectives

- Clarify inputs, outputs, performance goals, and success criteria
- Create a minimal spec or task brief

### 2. Draft (Prototype) the Code

- Implement a first working version focused on core logic
- Keep it modular; add logging and parameterization early
- Default to optimizing with parallelization and GPU use in mind

### 3. Smoke Test

- Run trivial or synthetic examples to confirm code executes without errors
- Verify basic I/O integrity and data shapes

### 4. Micro-benchmark and Debug

- Measure runtime on a small subset
- Profile CPU/GPU utilization and memory footprint
- Identify the bottleneck function(s)

### 5. Optimize

- Refactor for clarity and performance (vectorization, batching, caching)
- Ensure maximum parallelization by default (multiprocessing, multithreading, GPU)
- Assume 16 CPU cores, 64GB RAM, high-end GPU available
- Re-test after each optimization to ensure correctness is preserved

### 6. Scale Test (Pilot Run)

- Execute on a medium data slice or reduced network
- Monitor throughput, checkpointing, and logging output

### 7. Full Production Run

- Run on full dataset or cluster
- Save metrics, artifacts, and environment details (requirements.txt, config)

### 8. Post-run Validation

- Confirm outputs, run sanity checks, generate summary statistics
- Generate useful visualizations to help parse and review data critically
- Archive code, logs, and performance report

---

## Implementation Standards

### Version Control

- Git branch per feature (feature/embedding_refactor)
- PR before merge to main

### Environment Reproducibility

- requirements.txt or environment.yml committed
- Document Python version and key dependency versions

### Testing

- At least one smoke test per core module (pytest or simple asserts)
- Integration tests for pipeline stages

### Instrumentation

- Timing decorators or torch.profiler for GPU jobs
- Log timestamps for script begin/completion
- Log any warnings or errors generated

### Documentation

- Short README with purpose, usage, and sample output
- Docstrings for public functions

### Output Files

Ensure any output files log:

- Parameters used in creation
- Timestamps for begin and completion
- Any warnings or errors generated
