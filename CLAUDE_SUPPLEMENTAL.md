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
| **Batch API** | ~$0.06/paper (Opus 4.5) | ~1hr for 500 parallel | `ANTHROPIC_API_KEY` |

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

## Phase 5: Analysis and Advanced Search

### Citation Graph Pipeline

**Module:** `src/analysis/citation_graph.py`
**Output:** `data/index/citation_graph.json`

Builds a citation network by matching papers within the indexed corpus. Three matching strategies run in sequence:

1. **DOI matching** -- exact DOI lookups across extraction text fields
2. **Title matching** -- normalized title search with Jaccard similarity (threshold configurable via `GraphConfig.fuzzy_threshold`, default 0.85)
3. **Reference matching** -- scans extraction text (thesis, findings, claims) for mentions of other papers

Additional features:
- **Deduplication** (`_deduplicate_papers`) -- merges duplicate papers and redirects edges
- **Part-of relationships** (`_find_part_of_relationships`) -- detects chapter-in-book, reading-in-syllabus hierarchies
- **Collection/year filtering** via `GraphConfig`

Key entry points:
- `build_citation_graph(papers, extractions, config)` -- core builder, returns graph dict with nodes/edges
- `load_and_build_graph(index_dir, config)` -- loads index artifacts and builds
- `save_citation_graph(graph, output_path)` -- persists to JSON

Data classes: `GraphConfig`, `GraphNode`, `GraphEdge`

### Similarity Graph

**Module:** `src/analysis/similarity_graph.py`
**Output:** `data/index/similarity_graph.json`

Builds symmetric "related_to" edges (distinct from directional citation edges) using embedding similarity.

Algorithm:
1. Retrieve per-paper centroid embeddings from ChromaDB (mean of chunk embeddings)
2. Compute pairwise cosine similarity matrix
3. Filter edges by `SimilarityConfig.min_similarity` (default 0.75)
4. Cap at `max_edges_per_node` (default 10)
5. Optionally validate with Jaccard similarity on document text tokens

Key entry point: `build_similarity_graph(paper_ids, vector_store, config, paper_metadata, compute_jaccard)`

Data classes: `SimilarityConfig`, `SimilarityEdge`

### Gap Detection

**Module:** `src/analysis/gap_detection.py`

Heuristic analysis that identifies research gaps across four dimensions:

1. **Underrepresented topics** -- discipline tags / keywords below a frequency quantile
2. **Underrepresented methodologies** -- methodology approaches that appear rarely
3. **Year gaps** -- time periods with sparse coverage (`sparse_year_max_count`)
4. **Unaddressed future directions** -- future_directions from extractions that no other paper covers (checked via token index)

Configuration via `GapDetectionConfig` (max_items, min_count, quantile, evidence_limit, etc.).

Key entry points:
- `analyze_gap_report(papers, extractions, config, collections)` -- returns gap report dict
- `load_gap_report(index_dir, config, collections)` -- loads data and runs analysis

Output report structure: `{ corpus, parameters, topics_underrepresented, methodologies_underrepresented, year_gaps, future_directions, notes }`

### Research Question Generation

**Module:** `src/analysis/research_questions.py`

Generates structured research questions from gap analysis output using LLM prompts with quality guardrails.

Configuration: `ResearchQuestionConfig` with `count`, `styles` (exploratory/comparative/causal/descriptive/evaluative), `scope` (narrow/moderate/broad), `include_rationale`, `include_methodology_hints`.

Key functions:
- `build_prompts_from_gap_report(gap_report, config)` -- creates LLM prompts for each gap category
- `generate_questions_from_prompts(prompts, provider, model, config)` -- runs LLM and parses responses
- `deduplicate_questions(questions)` -- Jaccard-based dedup across generated questions
- `rank_questions(questions, config)` -- scores by novelty, specificity, feasibility
- `format_questions_markdown(result)` -- renders output

### Research Digest

**Module:** `src/analysis/research_digest.py`

Monitors the index for newly added papers and generates narrative summaries for email or chatbot delivery. Tracks processed papers via a state file (`digest_state.json`).

Data classes: `DigestConfig`, `PaperHighlight`, `ResearchDigest`

Key functions:
- `find_new_papers(index_dir, state_path)` -- detects papers not yet digested
- `build_paper_highlight(paper, extraction, config)` -- creates highlight for one paper
- `generate_digest(index_dir, config)` -- end-to-end digest generation
- `format_digest_markdown(digest)` / `format_digest_json(digest)` -- output formatters

### Topic Clustering

**Module:** `src/analysis/clustering.py`

Clusters paper embeddings to identify topic groups using a UMAP + HDBSCAN pipeline.

Steps:
1. `extract_embeddings(vector_store, chunk_type="full_summary")` -- pulls one embedding per paper from ChromaDB
2. `reduce_dimensions(embeddings)` -- UMAP reduction to 2D (configurable n_components, n_neighbors, min_dist)
3. `cluster_papers(embeddings_2d, min_cluster_size=5, min_samples=3)` -- HDBSCAN clustering (returns -1 for noise)
4. `run_clustering(vector_store, ...)` -- orchestrates the full pipeline, returns `ClusteringResult`
5. `save_clustering(result, output_dir)` -- persists to JSON

Data classes: `TopicCluster` (cluster_id, paper_ids, size, representative_titles, label), `ClusteringResult` (n_papers, n_clusters, n_noise, clusters, paper_clusters, umap_coords)

### LLM Council

**Module:** `src/analysis/llm_council.py`

Multi-provider consensus extraction inspired by karpathy/llm-council. Queries multiple LLM providers in parallel and aggregates responses.

Pipeline:
1. **Fan out** -- query all configured providers simultaneously (ThreadPoolExecutor)
2. **Collect** -- gather successful responses with per-provider timeout/error handling
3. **Aggregate** -- build consensus using field-specific strategies

Consensus strategies per field (`FIELD_STRATEGIES`):
- `LONGEST` -- thesis_statement, conclusions, contribution_summary, etc.
- `UNION` -- research_questions, limitations, future_directions, discipline_tags, key_findings, key_claims
- `MAJORITY_VOTE` -- methodology
- `AVERAGE` -- extraction_confidence

Key functions:
- `aggregate_extractions(extractions, weights)` -- merges multiple PaperExtraction objects
- `calculate_consensus_confidence(provider_responses)` -- scores agreement level

Data classes: `ProviderConfig` (name, weight, timeout, max_cost), `CouncilConfig` (providers, min_responses, fallback_to_single), `ProviderResponse`, `CouncilResult`

### RRF Multi-Query Search

**Module:** `src/query/rrf.py`

Reciprocal Rank Fusion for improved recall via query reformulation.

Pipeline:
1. `generate_query_variants(query, n_variants=4, provider, model)` -- LLM generates alternative phrasings (synonyms, research questions, methodology focus, scope changes)
2. Run each variant through vector search independently
3. `rrf_score(rankings, k=60)` -- fuse results using formula: `score(d) = sum(1 / (k + rank_i(d)))` across all rankings

Returns `list[tuple[str, float]]` sorted by fused score descending.

### Agentic Search

**Module:** `src/query/agentic.py`

Multi-round iterative search with LLM-driven gap analysis.

Loop (up to max_rounds):
1. Run vector search with current query
2. `analyze_gaps(query, results, provider, model)` -- LLM identifies missing sub-topics, methodologies, frameworks
3. LLM generates 1-3 follow-up queries to fill gaps
4. Re-search with follow-up queries and merge results
5. Stop when LLM returns empty gaps or max rounds reached

Data classes: `GapAnalysis` (gaps, follow_up_queries), `AgenticRound` (round_number, queries_used, papers_found, new_papers, gap_analysis), `AgenticSearchResult` (original_query, rounds, total_papers)

### Discord Bot

**Module:** `src/discord_bot/bot.py`

Exposes LITRIS search via Discord slash commands. Uses `LitrisAdapter` from the MCP layer.

Features:
- `/search` command with query, top_k, year_min, year_max parameters
- Embed formatting via `src/discord_bot/formatters.py`
- Button-based pagination (`PaginationView`) for multi-page results
- Graceful degradation when `discord.py` is not installed (`HAS_DISCORD` guard)

Entry point: `create_bot(adapter=None)` -- returns configured `discord.Client`

Requires: `pip install discord.py>=2.3` and `DISCORD_TOKEN` env var.

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

| Type         | Count | Notes                          |
|--------------|-------|--------------------------------|
| Total Papers | 1,624 | Run `litris_summary` for current stats |

**Note:** Run `mcp__litris__litris_summary` or check STATE.md for current corpus statistics.

---

## Performance Targets

### Processing Speed

- PDF extraction: < 5 seconds per paper
- LLM extraction: ~30 seconds per paper (API dependent)
- Embedding: < 1 second per paper (batched)

### Query Performance

- Search: < 1 second for 1,600-paper corpus
- Filter: Minimal additional latency
- Full retrieval: < 5 seconds

### Cost Targets

| Mode | Test (10 papers) | Full (500 papers) | Notes |
|------|------------------|-------------------|-------|
| CLI | $0 | $0 | Uses Max subscription |
| Batch API | ~$0.60 | ~$97 | Opus 4.5: $2.50/$12.50 per MTok |

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
