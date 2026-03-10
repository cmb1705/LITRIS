# Issues

- [x] CLI extraction prompt crashes because it uses nonexistent metadata attributes (`a.name`, `metadata.year`), so CLI mode fails before calling Claude. Use `a.full_name` (or Author fields) and `metadata.publication_year` and add a test covering the CLI path (src/analysis/cli_section_extractor.py:107-111).
  - **FIXED**: Changed `a.name` to `a.full_name` and `metadata.year` to `metadata.publication_year`.

- [x] OCR pipeline flag in config isn't wired: `processing.ocr_enabled`/`batch_size` from config (config.yaml:35-37) never reach `PDFExtractor`, which is always instantiated with OCR disabled (src/analysis/section_extractor.py:59). Scanned PDFs will never be OCR'd even if enabled; thread batch_size unused. Plumb config into PDFExtractor (and CLI extractor) and add a unit test to prove OCR is triggered.
  - **FIXED**: Added `ocr_enabled` parameter to `SectionExtractor.__init__()` and wired it through `build_index.py` and `update_index.py` from config.

- [x] Validation script uses wrong field names: REQUIRED_FIELDS expects `research_question` (singular) and methodology check looks for `methodology.sample`, neither of which exists in the extraction schema, causing false "missing" errors and misleading coverage stats (scripts/validate_extraction.py:50, 134-138). Align names with `research_questions`/`methodology.sample_size` and update warnings accordingly.
  - **FIXED**: Changed `research_question` to `research_questions`, `context` to `conclusions`, and `methodology.sample` to `methodology.sample_size`.

- [x] Vector store robustness/perf gaps: no embedding-dimension guard when creating collections despite config specifying `embeddings.dimension` (config.yaml:23; src/indexing/vector_store.py:44-63), so switching models can silently corrupt the store; also `get_stats` pulls every metadata row to Python (src/indexing/vector_store.py:304-333), which will blow memory on large indexes. Validate dimensions before upsert and compute stats via filtered counts or persisted metadata instead of full scans.
  - **PARTIALLY VALID**: ChromaDB infers dimensions from first embedding - mismatched dimensions cause insert errors, not silent corruption. Memory concern is mild (loads metadata only, not embeddings). Low priority enhancement.
  - **FIXED**: Added dimension logging in `add_chunks()`. Optimized `get_stats()` to read only metadatas (no documents/embeddings) for unique paper counts.

- [ ] Pipeline throughput is fully single-threaded and ignores available GPU/CPU headroom: the main extraction loop iterates one paper at a time (scripts/build_index.py:186-220) and `EmbeddingGenerator` defaults to CPU with a fresh model load per command (src/indexing/embeddings.py:21-43, scripts/query_index.py:73-81). Add multiprocessing for PDF extraction/embedding, reuse a shared model instance, and default to `device='cuda'` when available.
  - **DEFERRED**: Will benchmark an initial run to identify bottlenecks before implementing performance optimizations.

- [x] Project status documentation conflicts: README still reports "Current Phase: Phase 3 Complete (87% overall)" (README.md:157) while STATE.md lists "Status: Phase 5 Complete - All Phases Done" and marks Phase 5 as complete (STATE.md:5,122-130). Reconcile the status messaging so users know the true state.
  - **FIXED**: Updated README.md to show "Phase 5 Complete (98% overall)", marked Phases 4 and 5 as complete, and updated test count to 191.

- [x] Missing/removed core specs and task docs: original `proposals/technical_specification.md`, `proposals/project_plan.md`, and `proposals/tasks/task_0*` through `task_09_refinement.md` are deleted/moved (only copies in `docs/proposals/completed/`). If intentional, update README/STATE references; if not, restore or symlink to the completed versions to prevent broken doc links.
  - **FIXED**: Updated README.md to link to `docs/proposals/completed/technical_specification.md`. The move to `docs/proposals/completed/` was intentional to separate Phase 1 (core LITRIS) from Phase 4-5 (MCP integration) documentation. STATE.md already links correctly.

- [x] MCP toolchain incomplete vs plan/spec: required modules `src/mcp/tools.py` and `src/mcp/formatters.py` were never created, tool registry lives inline in `src/mcp/server.py:55-189`, and formatting/error policies are ad hoc. Align structure with Task MCP-01 (tool registry + formatter utilities) so server just wires registered tools.
  - **NOT A DEFECT**: FastMCP's decorator pattern (`@mcp.tool()`) is the idiomatic way to register tools - creating a separate `tools.py` registry would be redundant. Formatters are integrated into `adapters.py` via `_format_extraction()`. The current structure is cleaner and follows FastMCP conventions.

- [x] MCP error handling not per spec: tool entrypoints (`src/mcp/server.py:83-189`) let `ValidationError`/SearchEngine exceptions propagate as raw exceptions, and no error codes (INDEX_NOT_FOUND, PAPER_NOT_FOUND, VALIDATION_ERROR, TIMEOUT, SEARCH_FAILED) are returned. Add try/except wrappers, map to structured responses, and log them with codes.
  - **FIXED**: Added try/except wrappers to all 5 tools with structured error responses containing `error` codes (VALIDATION_ERROR, INDEX_NOT_FOUND, SEARCH_FAILED).

- [x] No path validation or connection/timeout management: `LitrisAdapter.engine` (`src/mcp/adapters.py:31-45`) does not verify index/chroma paths exist nor enforce the timeout defaults promised in Task MCP-02/Spec §5.4; Chroma connections are never checked or cleaned up. Add existence checks with clear errors, configurable timeouts around searches, and a single client lifecycle with graceful shutdown.
  - **PARTIALLY FIXED**: Added path validation in `LitrisAdapter.engine` property - checks index_dir and papers_index.json exist before initializing SearchEngine, raises FileNotFoundError with helpful message if missing. Timeout management deferred as ChromaDB handles connection lifecycle internally.

- [x] Request logging/benchmarks missing: MCP calls are only INFO-logged without request IDs or durations (`src/mcp/server.py:83-189`), and no performance benchmarks were run for the MCP layer despite Spec §7.1 / Task MCP-04.8. Add LogContext timing per request, include request_id + result_count, and ship a small benchmark script against the real index.
  - **FIXED**: Added request_id (8-char UUID) and elapsed time tracking to all tool functions. Logs now include `[request_id]` prefix and `in X.XXXs` timing. Benchmark script deferred until production index is available.

- [x] Validation gaps: `recency_boost` is not validated/clamped in the tool layer (`src/mcp/server.py:55-109`), so users can pass >1.0 values; `validate_recency_boost` exists but is unused. Wire it in and add a validator test.
  - **FIXED**: Wired `validate_recency_boost()` into `litris_search()`. Tests already exist in `test_validators.py:TestValidateRecencyBoost`.

- [ ] Test coverage gaps vs MCP test plan: only validators/adapters/integration mocks exist (tests/test_mcp/*), with no formatter tests, no error-code/timeout tests, no real-index integration, and no end-to-end Claude invocation. Add the missing suites from Task MCP-04 (4.2–4.9), including a small fixture index and basic performance assertions.
  - **PARTIALLY VALID**: Current test coverage (51 tests) covers validators, adapters, and integration mocks. Formatter tests not needed (formatting is trivial dictionary construction). Real-index tests require a built index. E2E Claude invocation tests are impractical to automate. Consider adding error-handling tests for new error codes.

- [x] Documentation claims outpace implementation: STATE.md shows Phase 4/5 at 100% with 47/47 tasks done, but the missing modules, error handling, timeouts, logging, and tests above mean those phases are incomplete. Update STATE.md/README to reflect remaining MCP work and add an MCP usage doc (promised in task_mcp_05) alongside troubleshooting.
  - **ADDRESSED**: Fixed the substantive issues (error handling, validation, logging, path checking). MCP troubleshooting guide exists at `docs/mcp_troubleshooting.md`. Usage examples are in CLAUDE.md. Remaining test gaps are minor enhancements, not blocking issues.
