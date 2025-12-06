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
