# LITRIS Semantic Analysis Migration Design

**Date:** 2026-03-13
**Status:** Approved
**Scope:** Full replacement of PaperExtraction with 40-question SemanticAnalysis

## Overview

Replace LITRIS's current extraction schema (18 fields, nested objects) with a
40-question semantic analysis modeled on Minty's approach. Every paper is
analyzed across 40 dimensions, each producing a prose answer and its own
embedding vector. This enables dimension-specific search, richer gap analysis,
and full scholarly characterization of each work.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| AI-specific questions | Dropped | Original Minty q15-q18 (llm_role, models, capability_claims, risk_claims) removed; LITRIS q15-q16 are novelty and stance |
| Domain-specific additions | q39 network_properties, q40 policy_recommendations | Core to user's corpus |
| Extraction approach | 6-pass multi-pass via LLM | Higher quality per answer than single-pass |
| Provider | Claude Opus 4.6 via CLI subscription | Best schema adherence, $0 marginal cost |
| Reasoning mode | Extended thinking / max reasoning | Best results per the user's requirement |
| Embedding architecture | Full per-dimension (replaces field-based) | Clean replacement, enables dimension search |
| RAPTOR | Keep paper_overview + core_contribution, drop section_summary | Dimension answers replace section narrative |
| Migration strategy | Full replacement, no backward compatibility layer | User explicitly requested clean break |
| Contribution vs novelty | Both kept as separate dimensions (q15 novelty, q22 contribution) | Distinct concepts: novelty = what's original; contribution = what's explicitly added to the field |
| reference_list | Kept as structured metadata alongside SemanticAnalysis | Required for citation graph DOI/title matching (confidence 0.95-1.0 edges) |
| quality_rating | Derived integer (1-5) from q21 prose via post-processing | Preserves quality_min search filter in ChromaDB |

## Schema: SemanticAnalysis

All question fields are `str | None`. Null means the dimension does not apply.

```python
class SemanticAnalysis(BaseModel):
    paper_id: str
    prompt_version: str          # e.g., "2.0.0"
    extraction_model: str        # model used, set at runtime from CLI args
    extracted_at: str            # ISO timestamp

    # Pass 1: Research Core
    q01_research_question: str | None
    q02_thesis: str | None
    q03_key_claims: str | None
    q04_evidence: str | None
    q05_limitations: str | None

    # Pass 2: Methodology
    q06_paradigm: str | None
    q07_methods: str | None
    q08_data: str | None
    q09_reproducibility: str | None
    q10_framework: str | None

    # Pass 3: Context & Discourse
    q11_traditions: str | None
    q12_key_citations: str | None
    q13_assumptions: str | None
    q14_counterarguments: str | None
    q15_novelty: str | None
    q16_stance: str | None

    # Pass 4: Meta & Audience
    q17_field: str | None
    q18_audience: str | None
    q19_implications: str | None
    q20_future_work: str | None
    q21_quality: str | None
    q22_contribution: str | None
    q23_source_type: str | None
    q24_other: str | None

    # Pass 5: Scholarly Positioning
    q25_institutional_context: str | None
    q26_historical_timing: str | None
    q27_paradigm_influence: str | None
    q28_disciplines_bridged: str | None
    q29_cross_domain_insights: str | None
    q30_cultural_scope: str | None
    q31_philosophical_assumptions: str | None

    # Pass 6: Impact, Gaps & Domain
    q32_deployment_gap: str | None
    q33_infrastructure_contribution: str | None
    q34_power_dynamics: str | None
    q35_gaps_and_omissions: str | None
    q36_dual_use_concerns: str | None
    q37_emergence_claims: str | None
    q38_remaining_other: str | None
    q39_network_properties: str | None
    q40_policy_recommendations: str | None

    # Coverage metadata (computed post-extraction)
    dimension_coverage: float = 0.0
    coverage_flags: list[str] = []
```

### Coverage Scoring

Each SemanticAnalysis receives a computed coverage score and flags.

| Tier | Coverage | Flag | Action |
|------|----------|------|--------|
| Full | >= 85% (34+/40) | None | Normal |
| Partial | 60-84% (24-33/40) | `PARTIAL_COVERAGE` | Expected for non-research docs |
| Sparse | 30-59% (12-23/40) | `SPARSE_COVERAGE` | Review recommended |
| Critical | < 30% (< 12/40) | `CRITICAL_GAPS` | Likely extraction failure |

Research Core (q01-q05) is weighted separately. Any None in q01-q05 raises
`CORE_GAPS` flag listing which dimensions are missing.

Post-extraction coverage report is generated to
`data/logs/extraction_review.json`.

### Storage

Primary artifact: `data/index/semantic_analyses.json`

Replaces: `data/index/extractions.json`

## 6-Pass Extraction Pipeline

### Flow

```
PDF -> ExtractionCascade (unchanged) -> cleaned text
  -> DocumentClassifier (unchanged) -> document_type
  -> truncate_for_llm (unchanged) -> truncated text
  -> Pass 1: Research Core (q01-q05)
  -> Pass 2: Methodology (q06-q10)
  -> Pass 3: Context & Discourse (q11-q16)
  -> Pass 4: Meta & Audience (q17-q24)
  -> Pass 5: Scholarly Positioning (q25-q31)
  -> Pass 6: Impact, Gaps & Domain (q32-q40)
  -> Merge passes -> SemanticAnalysis
  -> Coverage scoring -> flags
  -> RAPTOR (paper_overview + core_contribution)
  -> Create dimension chunks + RAPTOR + abstract
  -> Embed (Qwen3 4096d via Ollama)
  -> Store (ChromaDB + semantic_analyses.json)
```

### Pass Prompt Structure

Each pass uses a shared system prompt and a pass-specific user prompt.

System prompt instructs the model to:
- Analyze as an academic research analyst
- Provide thorough prose responses (2-5 sentences per dimension)
- Return null for inapplicable dimensions
- Return JSON only
- Extended thinking is enabled for maximum reasoning depth

User prompt includes:
- Document type framing note (research paper, book, report, etc.)
- Paper metadata (title, authors, year)
- Truncated paper text
- Pass-specific question definitions with full question text

### Document Type Framing

| Document Type | Framing Note |
|--------------|-------------|
| research_paper | Analyze as primary empirical or theoretical research |
| book / monograph | Analyze as a book-length scholarly work |
| review / meta-analysis | Analyze as a systematic review |
| report / white_paper | Analyze as a policy or technical report |
| thesis / dissertation | Analyze as a graduate research thesis |
| generic / non-academic | Analyze as a non-traditional document; many dimensions may return null |

### ExtractionResult Wrapper

`ExtractionResult` (schemas.py) wraps the analysis with metadata:

```python
class ExtractionResult(BaseModel):
    paper_id: str
    success: bool
    extraction: SemanticAnalysis | None  # was PaperExtraction | None
    error: str | None
    duration_seconds: float
    timestamp: datetime
```

The `extraction` field type changes from `PaperExtraction` to `SemanticAnalysis`.
All code that accesses `result.extraction.thesis_statement` (etc.) must update
to `result.extraction.q02_thesis` (etc.).

### Parallelism

6 passes per paper run sequentially (consistent reasoning context). Multiple
papers run in parallel via `--parallel N` workers (default 8 for CLI mode).

Schema (Phase 1) MUST complete before extraction pipeline (Phase 2) or consumer
rewiring (Phase 3) can begin. Once Phase 1 is done, Phases 2 and 3 can proceed
in parallel since consumers read the schema definition but do not depend on
extraction logic.

### Caching

Cache key per paper per pass:
`sha256(pdf_mtime + pdf_size + model + prompt_version + pass_number)`

Partial runs resume from the first uncached pass. Cache invalidates when
prompt_version changes.

### Provider Configuration

- Default: Claude Opus 4.6 via CLI subscription
- Overridable via `--provider` and `--model` CLI args
- `extraction_model` field records the actual model used per paper
- Extended thinking / max reasoning enabled for all passes

### Cost Estimate

- 1,746 papers x 6 passes = 10,476 LLM calls
- Via CLI subscription: $0 marginal cost
- Time estimate: ~65 hours at 8 parallel workers
  - CLI subscription has rate limits (~3-5 RPM sustained)
  - Budget ~22s per pass + rate limit delays = ~35-40s effective per call
  - Supports `--resume` for multi-session runs (designed for overnight/weekend batches)
  - Monitor via `data/logs/extraction_progress.json`

## Embedding Architecture

### Chunk Types

40 dimension chunks + 2 RAPTOR + 1 abstract = 43 potential chunks per paper.

| Chunk Type | Source | Count per Paper |
|------------|--------|-----------------|
| `dim_q01` through `dim_q40` | SemanticAnalysis fields | Up to 40 (only non-None) |
| `raptor_overview` | RAPTOR paper_overview (100-150 words) | 1 |
| `raptor_core` | RAPTOR core_contribution (40 words) | 1 |
| `abstract` | Zotero metadata | 1 (if available) |

Estimated actual: ~35 non-None dimensions avg + 2 RAPTOR + 1 abstract = ~38
chunks per paper.

### Retired Chunk Types

All 12 current types are retired: `thesis`, `findings`, `claims`,
`methodology`, `limitations`, `future_work`, `full_summary`, `contribution`,
`section_summary`, `paper_overview`, `core_contribution`, `abstract`.

The new `abstract` chunk type uses the same name but sources from Zotero
metadata (`PaperMetadata.abstract` field, populated from the Zotero database
abstract column), not extraction output. During migration, the old abstract
chunks are deleted with all other old types when ChromaDB is rebuilt from
scratch.

### Mapping for Hardcoded References

| Old Reference | New Equivalent | Used By |
|---------------|---------------|---------|
| `full_summary` | `raptor_overview` | similarity pairs, clustering |
| `thesis` | `dim_q02` | similar-paper fallback |
| `contribution` | `dim_q22` | similar-paper fallback |

### ChromaDB Metadata Per Chunk

```python
{
    "paper_id": str,
    "chunk_type": str,          # "dim_q01", "raptor_overview", "abstract"
    "dimension_group": str,     # "research_core", "methodology", etc.
    "title": str,
    "authors": str,
    "year": int,
    "collections": list[str],
    "item_type": str,
    "quality_rating": int,      # derived 1-5 from q21_quality prose; preserves $gte filter
    "dimension_coverage": float,
}
```

### Scale

- ~1,746 papers x ~38 chunks avg = ~66,300 chunks
- At 4096d Qwen3: ~1.1 GB in ChromaDB
- Embedding time: ~2.5 hours on workstation GPU

### RAPTOR

RAPTOR generates from the full set of non-None dimension answers:

| Layer | Length | Purpose |
|-------|--------|---------|
| `paper_overview` | 100-150 words | Broad search, result display, agent context |
| `core_contribution` | 40 words max | Clustering, quick scanning |

`section_summary` (300-500 words) is retired; dimension answers provide richer
section-level detail.

## Search Architecture

### Search Modes

| Mode | Chunks Searched | Use Case |
|------|----------------|----------|
| Standard | All dim + RAPTOR + abstract | General search (current behavior, wider net) |
| Dimension | Single `dim_q{NN}` | "Find papers with similar methodology" |
| Group | All dims in a thematic group | "Search only Context & Discourse dimensions" |
| RAPTOR | `raptor_overview` only | Fast broad topic matching |
| RRF | Standard with query variants | Multi-query fusion (unchanged) |
| Agentic | Standard with gap analysis | Multi-round iterative (unchanged) |

### Dimension Groups

| Group | Dimensions | Chunk Count per Paper |
|-------|-----------|----------------------|
| `research_core` | q01-q05 | Up to 5 |
| `methodology` | q06-q10 | Up to 5 |
| `context` | q11-q16 | Up to 6 |
| `meta` | q17-q24 | Up to 8 |
| `scholarly` | q25-q31 | Up to 7 |
| `impact` | q32-q40 | Up to 9 |
| `raptor` | raptor_overview, raptor_core | 2 |
| `metadata` | abstract | 1 |

### MCP Tools

| Tool | Type |
|------|------|
| `litris_search` | Updated (uses new chunks) |
| `litris_search_rrf` | Unchanged |
| `litris_search_agentic` | Unchanged |
| `litris_search_dimension` | **New** |
| `litris_search_group` | **New** |

### Web UI

Chunk type multiselect reorganized into grouped sections with "Select group"
shortcuts. Extraction display shows dimensions grouped by pass with
human-readable labels.

## Downstream Consumer Changes

### Full Rewiring Required (~35 files)

| File | Change Summary |
|------|---------------|
| `src/analysis/schemas.py` | Replace PaperExtraction with SemanticAnalysis; ExtractionResult.extraction becomes `SemanticAnalysis | None` |
| `src/analysis/prompts.py` | Retire old templates |
| `src/analysis/semantic_prompts.py` | **New**: 6 pass prompt templates |
| `src/analysis/coverage.py` | **New**: coverage scoring, flagging, reports |
| `src/analysis/section_extractor.py` | Rewire for 6-pass orchestration |
| `src/analysis/cli_section_extractor.py` | Replace PaperExtraction construction with SemanticAnalysis; update parse_response() |
| `src/analysis/raptor.py` | Generate from SemanticAnalysis dimensions |
| `src/analysis/gap_detection.py` | Replace structured field access with prose dimension reads (see Gap Detection Rewiring below) |
| `src/analysis/citation_graph.py` | Concatenate all non-None q fields for title matching |
| `src/analysis/research_digest.py` | Read q02, q07, q04 instead of old fields |
| `src/analysis/llm_council.py` | Simplify aggregation: all q fields are `str | None`, use LONGEST strategy for all (no UNION/MAJORITY_VOTE needed) |
| `src/analysis/clustering.py` | Query `raptor_overview` instead of `full_summary` |
| `src/indexing/embeddings.py` | Create dim_q chunks from SemanticAnalysis |
| `src/indexing/vector_store.py` | New CHUNK_TYPES list, updated stats |
| `src/indexing/structured_store.py` | Update file reference from `extractions.json` to `semantic_analyses.json` |
| `src/query/search.py` | Update similar-papers references |
| `src/query/retrieval.py` | Render dimensions grouped by pass |
| `src/query/deep_review.py` | Read q fields from SemanticAnalysis |
| `src/query/dimension_search.py` | **New**: dimension and group search logic |
| `src/mcp/validators.py` | Validate against new chunk types |
| `src/mcp/adapters.py` | Format SemanticAnalysis for MCP responses |
| `src/mcp/server.py` | Add litris_search_dimension, litris_search_group |
| `src/discord_bot/formatters.py` | Replace `thesis_statement` references with `q02_thesis` |
| `scripts/build_index.py` | Update similarity pairs, integration point |
| `scripts/batch_extract.py` | Update extraction field references and `extractions.json` paths |
| `scripts/web_ui.py` | Grouped chunk filter, dimension display |
| `scripts/query_index.py` | Updated type names, help text |
| `scripts/validate_extraction.py` | Validate dimension coverage |
| `scripts/update_index.py` | Replace PaperExtraction construction with SemanticAnalysis |
| `scripts/compare_providers.py` | Update thesis_statement refs to q02_thesis |
| `scripts/export_results.py` | Update thesis_statement refs to q02_thesis |
| `scripts/smoketest_gap_analysis.py` | Update extraction fixture data to SemanticAnalysis fields |
| `scripts/classify_existing.py` | Update extractions.json ref to semantic_analyses.json |
| `scripts/preflight_analysis.py` | Update extractions.json ref to semantic_analyses.json |
| `scripts/smoketest_web_ui.py` | Update chunk_type="thesis" to new dim type |
| `src/analysis/document_types.py` | Update required_extraction_fields to q-field names |
| `src/config.py` | Update FederatedSearchConfig path description |
| `src/query/agentic.py` | Replace thesis_statement, discipline_tags refs in gap analysis prompts |
| `src/analysis/__init__.py` | Update PaperExtraction export to SemanticAnalysis |

### Gap Detection Rewiring Detail

`gap_detection.py` currently reads structured fields from PaperExtraction:
- `methodology.approach` -> becomes text search within `q07_methods` prose
- `methodology.paradigm` -> becomes `q06_paradigm` prose
- `discipline_tags` -> becomes `q17_field` prose
- `future_work` -> becomes `q20_future_work` prose
- `data_characteristics` -> becomes `q08_data` prose

Since all dimensions are now prose (not structured objects), gap detection must
parse prose text for pattern matching. For example, methodology gap detection
currently counts exact `paradigm` enum values; it must switch to keyword/phrase
extraction from q06 prose. This is the most invasive consumer rewiring.

### No Changes Required (~10 files)

| File | Reason |
|------|--------|
| `src/extraction/cascade.py` | Upstream of extraction |
| `src/extraction/pdf_extractor.py` | Upstream of extraction |
| `src/extraction/text_cleaner.py` | Upstream of extraction |
| `src/extraction/marker_extractor.py` | Upstream of extraction |
| `src/extraction/arxiv_extractor.py` | Upstream of extraction |
| `src/extraction/ocr_handler.py` | Upstream of extraction |
| `src/query/rrf.py` | Operates on search results |
| `src/query/federated.py` | Imports ChunkType from embeddings.py; auto-resolves when ChunkType is updated |
| `src/analysis/similarity_graph.py` | Uses embeddings only |
| `src/zotero/` | Metadata layer, upstream |

### Retired

| File | Reason |
|------|--------|
| `scripts/backfill_discipline_tags.py` | Replaced by q17 (field) and q28 (disciplines_bridged) |
| `src/analysis/openai_client.py` | LLM client files all construct PaperExtraction objects. New pipeline uses CLI subscription exclusively. These files are retained but non-functional until rewired for SemanticAnalysis. Rewiring is out of scope for initial migration (tracked as future multi-provider support task). |
| `src/analysis/anthropic_client.py` | Same as above |
| `src/analysis/gemini_client.py` | Same as above |
| `src/analysis/batch_client.py` | Same as above |
| `src/analysis/ollama_client.py` | Same as above |
| `src/analysis/llamacpp_client.py` | Same as above |
| `scripts/smoketest_openai.py` | Provider-specific smoketest; non-functional after migration |
| `scripts/smoketest_gemini.py` | Provider-specific smoketest; non-functional after migration |

### Test Rewrites

| Test File | Change |
|-----------|--------|
| `tests/test_embeddings.py` | New chunk type assertions |
| `tests/test_gap_analysis.py` | New field name references |
| `tests/test_llm_council.py` | SemanticAnalysis aggregation |
| `tests/test_citation_graph.py` | Updated _extract_text_fields inputs |
| `tests/test_research_digest.py` | Updated field references |
| `tests/test_quality_rating.py` | New prompt version assertions |
| `tests/test_semantic_analysis.py` | **New**: schema validation, coverage scoring |
| `tests/test_coverage.py` | **New**: coverage tiers, flagging |
| `tests/test_dimension_search.py` | **New**: dimension and group search |
| `tests/test_llm_extraction.py` | Heavy PaperExtraction usage; rewire to SemanticAnalysis |
| `tests/test_cli_extraction.py` | PaperExtraction construction; update to SemanticAnalysis |
| `tests/test_raptor.py` | PaperExtraction imports; update inputs |
| `tests/test_discord_bot.py` | thesis_statement, methodology refs in test data |
| `tests/test_deep_review.py` | thesis_statement, key_findings refs in test data |
| `tests/test_query.py` | thesis_statement, discipline_tags refs in test data |
| `tests/test_similarity_pairs.py` | thesis_statement in extraction test data |
| `tests/test_rrf.py` | Update chunk_type="thesis" refs in test data |
| `tests/test_federated_search.py` | Update chunk_type="thesis" refs in test data |
| `tests/test_clustering.py` | Update chunk_type="full_summary" and "thesis" refs |
| `tests/test_agentic.py` | Update thesis_statement, discipline_tags in test fixture data |
| `tests/test_document_types.py` | Update required_extraction_fields assertions |
| `tests/test_mcp/` | Updated validators, adapters |

## Migration Strategy

### Phase Sequence

```
Phase 1: Schema & Prompts         (code only, no data)
Phase 2: Extraction Pipeline      (rewire section_extractor)
Phase 3: Downstream Consumers     (rewire all readers)
Phase 4: Embedding & Search       (new chunks, search modes)
Phase 5: Tests                    (rewrite affected tests)
Phase 6: Full Re-extraction       (1,746 papers, 6 passes each)
Phase 7: Rebuild Index            (embed, store, graphs)
Phase 8: Validation & Cleanup     (coverage report, retire old code)
```

Phases 1-5 are code-only. No data is touched until Phase 6.

### Data Safety

Before Phase 6:
```
data/index/ -> data/index_backup_v1/
```

Old extractions.json, papers.json, ChromaDB, and graph files are preserved.

### Rollback Plan

If migration fails partway through:

1. **Phases 1-5 (code only)**: `git revert` the migration commits. No data changes.
2. **Phase 6 (extraction)**: `data/index_backup_v1/` preserves all original data.
   Restore with `cp -r data/index_backup_v1/* data/index/`.
3. **Phase 7 (index rebuild)**: ChromaDB is rebuilt from scratch; no incremental
   state to corrupt. Re-run with old extraction data if needed.
4. **Partial extraction**: `--resume` flag means extraction can restart from where
   it stopped. No need to re-extract completed papers.

### Cache Invalidation

The cache key includes `prompt_version`. When `prompt_version` changes from
"1.4.0" to "2.0.0", all cached extraction results automatically miss. Old cache
entries (keyed with "1.4.0") remain on disk but are never read. No explicit
cache purge is needed; a cleanup script can optionally remove old entries after
migration completes.

### Beads Epic Structure

```
Epic: LITRIS Semantic Analysis Migration (P1)
  |
  +-- Epic: Schema & Extraction (Phases 1-2)
  |     +-- Task: Define SemanticAnalysis schema in schemas.py
  |     +-- Task: Write 6 pass prompt templates in semantic_prompts.py
  |     +-- Task: Write coverage scoring module (coverage.py)
  |     +-- Task: Rewire section_extractor.py for 6-pass orchestration
  |     +-- Task: Update cache key logic for per-pass caching
  |
  +-- Epic: Consumer Rewiring (Phases 3-4)
  |     +-- Task: Rewire embeddings.py (dim chunk generation)
  |     +-- Task: Rewire vector_store.py (chunk types, stats)
  |     +-- Task: Rewire search.py (similar papers, new references)
  |     +-- Task: Rewire gap_detection.py (prose parsing replaces structured fields)
  |     +-- Task: Rewire citation_graph.py
  |     +-- Task: Rewire research_digest.py
  |     +-- Task: Rewire raptor.py
  |     +-- Task: Rewire MCP (adapters, validators, server)
  |     +-- Task: Rewire retrieval.py and deep_review.py
  |     +-- Task: Rewire web_ui.py
  |     +-- Task: Rewire build_index.py and clustering.py
  |     +-- Task: Rewire cli_section_extractor.py (PaperExtraction -> SemanticAnalysis)
  |     +-- Task: Rewire structured_store.py (extractions.json -> semantic_analyses.json)
  |     +-- Task: Rewire batch_extract.py (field refs, file paths)
  |     +-- Task: Rewire discord_bot/formatters.py (thesis_statement -> q02_thesis)
  |     +-- Task: Create dimension_search.py
  |     +-- Task: Rewire agentic.py (thesis_statement, discipline_tags in gap prompts)
  |     +-- Task: Rewire scripts (update_index, compare_providers, export_results, smoketest_gap_analysis, smoketest_web_ui, classify, preflight)
  |     +-- Task: Rewire document_types.py (required_extraction_fields)
  |     +-- Task: Update config.py (FederatedSearchConfig path description)
  |     +-- Task: Rewire query_index.py (type names, help text)
  |     +-- Task: Update validate_extraction.py (dimension coverage validation)
  |     +-- Task: Retire old prompts.py templates
  |     +-- Task: Update llm_council.py aggregation (simplify: all str fields, LONGEST only)
  |     +-- Task: Mark LLM client files as non-functional pending future multi-provider rewire
  |
  +-- Epic: Test Suite (Phase 5)
  |     +-- Task: Rewrite test_embeddings.py
  |     +-- Task: Rewrite test_gap_analysis.py
  |     +-- Task: Rewrite test_llm_council.py
  |     +-- Task: Rewrite test_citation_graph.py
  |     +-- Task: Rewrite test_research_digest.py
  |     +-- Task: Rewrite test_quality_rating.py
  |     +-- Task: Write test_semantic_analysis.py
  |     +-- Task: Write test_coverage.py
  |     +-- Task: Write test_dimension_search.py
  |     +-- Task: Rewrite test_llm_extraction.py and test_cli_extraction.py
  |     +-- Task: Rewrite test_raptor.py
  |     +-- Task: Rewrite test_discord_bot.py and test_deep_review.py
  |     +-- Task: Rewrite test_query.py, test_similarity_pairs.py, test_document_types.py
  |     +-- Task: Update test_rrf.py, test_federated_search.py, test_clustering.py, test_agentic.py
  |     +-- Task: Update test_mcp/ tests
  |
  +-- Epic: Index Rebuild (Phases 6-8)
        +-- Task: Backup current index
        +-- Task: Run full 6-pass extraction (1,746 papers)
        +-- Task: Rebuild embeddings and ChromaDB
        +-- Task: Rebuild citation and similarity graphs
        +-- Task: Generate and review coverage report
        +-- Task: Validation spot-checks (10 papers across doc types)
        +-- Task: Retire old schema code and extractions.json
```

Tasks are sized for single polecat sessions. Schema & Extraction epic MUST
complete first (schema definition is a dependency for all consumers). Once
schema is defined, Consumer Rewiring tasks can run in parallel. Test Suite
depends on both Schema and Consumer Rewiring epics. Index Rebuild depends on
all prior epics.

## 40-Question Reference

### Pass 1: Research Core

| # | Name | Question |
|---|------|----------|
| q01 | research_question | What research questions or objectives does this work address? |
| q02 | thesis | What is the central thesis or main argument? |
| q03 | key_claims | What are the key claims or propositions made? |
| q04 | evidence | What evidence is presented and how strong is it? |
| q05 | limitations | What limitations are acknowledged or apparent? |

### Pass 2: Methodology

| # | Name | Question |
|---|------|----------|
| q06 | paradigm | What research paradigm underlies this work? (positivist, interpretivist, critical, pragmatist, etc.) |
| q07 | methods | What methods and analytical techniques are used? |
| q08 | data | What data sources, sample sizes, and time periods are involved? |
| q09 | reproducibility | How reproducible is this work? Are methods, data, and code available? |
| q10 | framework | What theoretical or conceptual framework is used? |

### Pass 3: Context & Discourse

| # | Name | Question |
|---|------|----------|
| q11 | traditions | What intellectual traditions or schools of thought does this draw from? |
| q12 | key_citations | What are the most influential works cited, and how do they shape this paper? |
| q13 | assumptions | What assumptions (stated or unstated) underlie the analysis? |
| q14 | counterarguments | What counterarguments or alternative interpretations are addressed? |
| q15 | novelty | What is novel or original about this work? |
| q16 | stance | What is the author's stance or perspective on the topic? |

### Pass 4: Meta & Audience

| # | Name | Question |
|---|------|----------|
| q17 | field | What academic field(s) does this work belong to? |
| q18 | audience | Who is the intended audience? |
| q19 | implications | What are the broader theoretical or practical implications? |
| q20 | future_work | What future research directions are suggested? |
| q21 | quality | How would you rate the overall quality? (methodology rigor, evidence strength, contribution significance) |
| q22 | contribution | What is the explicit contribution of this work to its field? |
| q23 | source_type | What type of document is this? (empirical study, review, theoretical, report, etc.) |
| q24 | other | What else is noteworthy that the above questions don't capture? |

### Pass 5: Scholarly Positioning

| # | Name | Question |
|---|------|----------|
| q25 | institutional_context | What institutional or organizational context shaped this work? |
| q26 | historical_timing | Why does this work appear now? What historical/temporal factors are relevant? |
| q27 | paradigm_influence | How does this work relate to dominant paradigms in its field? |
| q28 | disciplines_bridged | What disciplines does this work bridge or draw from? |
| q29 | cross_domain_insights | What insights transfer to or from other domains? |
| q30 | cultural_scope | What cultural, geographic, or demographic scope does this cover? |
| q31 | philosophical_assumptions | What philosophical assumptions underlie the methodology or claims? |

### Pass 6: Impact, Gaps & Domain

| # | Name | Question |
|---|------|----------|
| q32 | deployment_gap | What gap exists between this research and real-world application? |
| q33 | infrastructure_contribution | Does this work contribute tools, datasets, frameworks, or infrastructure? |
| q34 | power_dynamics | What power dynamics, inequities, or stakeholder tensions are relevant? |
| q35 | gaps_and_omissions | What important aspects does this work fail to address? |
| q36 | dual_use_concerns | Are there dual-use or ethical concerns with the findings or methods? |
| q37 | emergence_claims | Does this work describe emergent phenomena or system-level behaviors? |
| q38 | remaining_other | What else is significant that no prior question has captured? |
| q39 | network_properties | What network structures, metrics, or graph algorithms are central? |
| q40 | policy_recommendations | What specific policy recommendations or actionable guidance is proposed? |
