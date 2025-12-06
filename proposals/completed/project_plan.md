# Literature Review Index: Project Plan and File Structure

**Version:** 1.3
**Date:** 2024-12-05
**Status:** All Phases Complete

---

## 1. Project Phases Overview

| Phase | Name | Goal | Estimated Tasks |
|-------|------|------|-----------------|
| 0 | Setup | Project scaffolding, dependencies, configuration | 8 |
| 1 | Foundation | Zotero reader, PDF extraction, basic LLM extraction | 12 |
| 2 | Semantic Search | Embeddings, vector store, query interface | 8 |
| 3 | Robustness | OCR fallback, error handling, full library processing | 10 |
| 4 | Incremental Updates | Change detection, update pipeline, state management | 7 |
| 5 | Refinement | Prompt tuning, output formats, documentation | 6 |

**Total Estimated Tasks:** 51

---

## 2. File Tree Structure

The following is the complete anticipated file structure for the project. Files are grouped by creation phase.

### Phase 0: Setup

D:\Git_Repos\Lit_Review\
|-- .gitignore                          [exists]
|-- README.md                           [exists - update]
|-- requirements.txt                    [create]
|-- requirements-dev.txt                [create]
|-- config.yaml                         [create]
|-- .env.example                        [create]
|-- pyproject.toml                      [create]
|-- proposals\
|   |-- technical_specification.md      [exists]
|   |-- project_plan.md                 [this file]
|   +-- tasks\                          [create directory]


### Phase 1: Foundation

D:\Git_Repos\Lit_Review\
|-- src\
|   |-- __init__.py
|   |-- config.py
|   |-- zotero\
|   |   |-- __init__.py
|   |   |-- database.py
|   |   +-- models.py
|   |-- extraction\
|   |   |-- __init__.py
|   |   |-- pdf_extractor.py
|   |   +-- text_cleaner.py
|   |-- analysis\
|   |   |-- __init__.py
|   |   |-- llm_client.py              # Batch API client
|   |   |-- cli_executor.py            # CLI mode executor
|   |   |-- rate_limit_handler.py      # CLI rate limit handling
|   |   |-- progress_tracker.py        # CLI progress persistence
|   |   |-- section_extractor.py       # Batch API extractor
|   |   |-- cli_section_extractor.py   # CLI mode extractor
|   |   |-- prompts.py
|   |   +-- schemas.py
|   +-- utils\
|       |-- __init__.py
|       |-- logging_config.py
|       +-- file_utils.py
|-- scripts\
|   +-- build_index.py
|-- data\
|   |-- index\
|   |   |-- papers.json
|   |   |-- extractions.json
|   |   +-- metadata.json
|   |-- cache\
|   |   +-- pdf_text\
|   +-- logs\
+-- tests\
    |-- __init__.py
    |-- conftest.py
    |-- test_zotero_reader.py
    |-- test_pdf_extraction.py
    +-- fixtures\
        +-- sample_papers\


### Phase 2: Semantic Search

D:\Git_Repos\Lit_Review\
|-- src\
|   |-- indexing\
|   |   |-- __init__.py
|   |   |-- embeddings.py
|   |   |-- vector_store.py
|   |   +-- structured_store.py
|   +-- query\
|       |-- __init__.py
|       |-- search.py
|       +-- retrieval.py
|-- scripts\
|   +-- query_index.py
|-- data\
|   |-- index\
|   |   |-- summary.json
|   |   +-- chroma\
|   +-- query_results\
+-- tests\
    |-- test_embeddings.py
    +-- test_query.py


### Phase 3: Robustness

D:\Git_Repos\Lit_Review\
|-- src\
|   +-- extraction\
|       +-- ocr_handler.py
|-- scripts\
|   +-- validate_extraction.py
|-- data\
|   +-- cache\
|       +-- processing_state.json
+-- tests\
    +-- test_ocr.py


### Phase 4: Incremental Updates

D:\Git_Repos\Lit_Review\
|-- scripts\
|   +-- update_index.py
+-- tests\
    +-- test_incremental_update.py


### Phase 5: Refinement

D:\Git_Repos\Lit_Review\
|-- docs\
|   |-- usage.md
|   |-- query_guide.md
|   +-- troubleshooting.md
+-- scripts\
    +-- export_results.py


### Complete Final Structure

D:\Git_Repos\Lit_Review\
|-- .env.example
|-- .gitignore
|-- config.yaml
|-- pyproject.toml
|-- README.md
|-- requirements.txt
|-- requirements-dev.txt
|-- STATE.md
|-- data\
|   |-- cache\
|   |   |-- pdf_text\
|   |   +-- processing_state.json
|   |-- index\
|   |   |-- chroma\
|   |   |-- extractions.json
|   |   |-- metadata.json
|   |   |-- papers.json
|   |   +-- summary.json
|   |-- logs\
|   +-- query_results\
|-- docs\
|   |-- query_guide.md
|   |-- troubleshooting.md
|   +-- usage.md
|-- proposals\
|   |-- project_plan.md
|   |-- technical_specification.md
|   +-- tasks\
|       |-- task_00_setup.md
|       |-- task_01_zotero_reader.md
|       |-- task_02_pdf_extraction.md
|       |-- task_03_llm_extraction.md
|       |-- task_04_embeddings.md
|       |-- task_05_vector_store.md
|       |-- task_06_query_interface.md
|       |-- task_07_robustness.md
|       |-- task_08_incremental_updates.md
|       +-- task_09_refinement.md
|-- scripts\
|   |-- build_index.py
|   |-- export_results.py
|   |-- query_index.py
|   |-- update_index.py
|   +-- validate_extraction.py
|-- src\
|   |-- __init__.py
|   |-- config.py
|   |-- analysis\
|   |   |-- __init__.py
|   |   |-- llm_client.py
|   |   |-- prompts.py
|   |   |-- schemas.py
|   |   +-- section_extractor.py
|   |-- extraction\
|   |   |-- __init__.py
|   |   |-- ocr_handler.py
|   |   |-- pdf_extractor.py
|   |   +-- text_cleaner.py
|   |-- indexing\
|   |   |-- __init__.py
|   |   |-- embeddings.py
|   |   |-- structured_store.py
|   |   +-- vector_store.py
|   |-- query\
|   |   |-- __init__.py
|   |   |-- retrieval.py
|   |   +-- search.py
|   |-- utils\
|   |   |-- __init__.py
|   |   |-- file_utils.py
|   |   +-- logging_config.py
|   +-- zotero\
|       |-- __init__.py
|       |-- database.py
|       +-- models.py
+-- tests\
    |-- __init__.py
    |-- conftest.py
    |-- test_embeddings.py
    |-- test_incremental_update.py
    |-- test_ocr.py
    |-- test_pdf_extraction.py
    |-- test_query.py
    |-- test_zotero_reader.py
    +-- fixtures\
        +-- sample_papers\


---

## 3. Phase 0: Setup TODO List

### 0.1 Project Initialization

- [x] 0.1.1 Create `pyproject.toml` with project metadata and tool configurations
- [x] 0.1.2 Create `requirements.txt` with core dependencies
- [x] 0.1.3 Create `requirements-dev.txt` with development dependencies
- [x] 0.1.4 Create `.env.example` with required environment variable templates
- [x] 0.1.5 Update `.gitignore` to include data directories, cache, logs, .env

### 0.2 Configuration Setup

- [x] 0.2.1 Create `config.yaml` with all configuration options
- [x] 0.2.2 Create `src/__init__.py` with package initialization
- [x] 0.2.3 Create `src/config.py` with configuration loader class

### 0.3 Directory Structure

- [x] 0.3.1 Create all `src/` subdirectories with `__init__.py` files
- [x] 0.3.2 Create `data/` subdirectories (index, cache, logs, query_results)
- [x] 0.3.3 Create `scripts/` directory
- [x] 0.3.4 Create `tests/` directory with `conftest.py`
- [x] 0.3.5 Create `proposals/tasks/` directory

---

## 4. Phase 1: Foundation TODO List

### 1.1 Utility Modules

- [x] 1.1.1 Create `src/utils/logging_config.py` with logging setup
- [x] 1.1.2 Create `src/utils/file_utils.py` with file operation helpers

### 1.2 Zotero Database Reader

- [x] 1.2.1 Create `src/zotero/models.py` with Pydantic models for Paper, Author, Collection
- [x] 1.2.2 Create `src/zotero/database.py` with ZoteroDatabase class
- [x] 1.2.3 Implement database connection method (read-only)
- [x] 1.2.4 Implement query for all items with PDF attachments
- [x] 1.2.5 Implement metadata extraction joins (itemData, itemDataValues, fields)
- [x] 1.2.6 Implement author extraction (creators, itemCreators)
- [x] 1.2.7 Implement collection extraction (collections, collectionItems)
- [x] 1.2.8 Implement PDF path resolution from storage keys
- [x] 1.2.9 Write `tests/test_zotero_reader.py` with unit tests

### 1.3 PDF Text Extraction

- [x] 1.3.1 Create `src/extraction/pdf_extractor.py` with PDFExtractor class
- [x] 1.3.2 Implement PyMuPDF-based text extraction
- [x] 1.3.3 Implement Zotero cache detection and usage (.zotero-ft-cache)
- [x] 1.3.4 Implement extraction quality metrics (char count, word count, page ratio)
- [x] 1.3.5 Create `src/extraction/text_cleaner.py` with text normalization
- [x] 1.3.6 Implement Unicode normalization
- [x] 1.3.7 Implement PDF artifact cleanup (ligatures, broken words)
- [x] 1.3.8 Implement text caching to `data/cache/pdf_text/`
- [x] 1.3.9 Write `tests/test_pdf_extraction.py` with unit tests
- [ ] 1.3.10 Add 3-5 sample PDFs to `tests/fixtures/sample_papers/` (skipped: tests use generated PDFs)

### 1.4 LLM Extraction

- [x] 1.4.1 Create `src/analysis/schemas.py` with Pydantic models for Extraction output
- [x] 1.4.2 Create `src/analysis/prompts.py` with extraction prompt template
- [x] 1.4.3 Create `src/analysis/llm_client.py` with Claude API wrapper
- [x] 1.4.4 Implement API call with retry logic and exponential backoff
- [x] 1.4.5 Implement token usage tracking
- [x] 1.4.6 Create `src/analysis/section_extractor.py` with SectionExtractor class
- [x] 1.4.7 Implement extraction orchestration (text in, structured data out)
- [x] 1.4.8 Implement response parsing and validation
- [x] 1.4.9 Implement partial extraction handling for malformed responses
- [x] 1.4.10 Write `tests/test_llm_extraction.py` with mocked API tests

### 1.5 Basic Build Script

- [x] 1.5.1 Create `scripts/build_index.py` with CLI interface
- [x] 1.5.2 Implement Zotero scan step
- [x] 1.5.3 Implement PDF extraction step with progress bar
- [x] 1.5.4 Implement LLM extraction step with progress bar
- [x] 1.5.5 Implement JSON output to `data/index/papers.json`
- [x] 1.5.6 Implement JSON output to `data/index/extractions.json`
- [x] 1.5.7 Implement basic `data/index/metadata.json` generation
- [x] 1.5.8 Test with 10-paper subset

---

## 5. Phase 2: Semantic Search TODO List

### 2.1 Embedding Generation

- [x] 2.1.1 Create `src/indexing/embeddings.py` with EmbeddingGenerator class
- [x] 2.1.2 Implement sentence-transformers model loading
- [x] 2.1.3 Implement chunking strategy for extraction fields
- [x] 2.1.4 Implement batch embedding generation
- [x] 2.1.5 Write `tests/test_embeddings.py` with unit tests

### 2.2 Vector Store

- [x] 2.2.1 Create `src/indexing/vector_store.py` with VectorStore class
- [x] 2.2.2 Implement ChromaDB collection initialization
- [x] 2.2.3 Implement document insertion with metadata
- [x] 2.2.4 Implement similarity search with filters
- [x] 2.2.5 Implement document deletion for updates

### 2.3 Structured Store

- [x] 2.3.1 Create `src/indexing/structured_store.py` with StructuredStore class
- [x] 2.3.2 Implement JSON file read/write operations
- [x] 2.3.3 Implement paper lookup by ID
- [x] 2.3.4 Implement summary statistics generation
- [x] 2.3.5 Implement `data/index/summary.json` generation

### 2.4 Query Interface

- [x] 2.4.1 Create `src/query/search.py` with SearchEngine class
- [x] 2.4.2 Implement semantic search wrapper
- [x] 2.4.3 Implement metadata filtering
- [x] 2.4.4 Implement result ranking and scoring
- [x] 2.4.5 Create `src/query/retrieval.py` with result formatting
- [x] 2.4.6 Implement JSON output format
- [x] 2.4.7 Implement markdown output format
- [x] 2.4.8 Create `scripts/query_index.py` with CLI interface
- [x] 2.4.9 Write `tests/test_query.py` with integration tests

### 2.5 Build Script Updates

- [x] 2.5.1 Add embedding generation step to `build_index.py`
- [x] 2.5.2 Add ChromaDB population step to `build_index.py`
- [x] 2.5.3 Add summary generation step to `build_index.py`

---

## 6. Phase 3: Robustness TODO List

### 3.1 OCR Fallback

- [x] 3.1.1 Create `src/extraction/ocr_handler.py` with OCRHandler class
- [x] 3.1.2 Implement PDF-to-image conversion using pdf2image
- [x] 3.1.3 Implement Tesseract OCR integration
- [x] 3.1.4 Implement quality detection to trigger OCR fallback
- [x] 3.1.5 Integrate OCR fallback into PDFExtractor
- [x] 3.1.6 Write `tests/test_ocr.py` with scanned PDF tests

### 3.2 Error Handling

- [x] 3.2.1 Implement comprehensive try/except in all pipeline stages
- [x] 3.2.2 Implement failed items tracking in metadata.json
- [x] 3.2.3 Implement `--retry-failed` flag in build_index.py
- [x] 3.2.4 Implement `--skip-paper` flag in build_index.py
- [x] 3.2.5 Add graceful handling for encrypted PDFs
- [x] 3.2.6 Add graceful handling for corrupted PDFs

### 3.3 Processing State

- [x] 3.3.1 Implement checkpoint saving during processing
- [x] 3.3.2 Implement resume from checkpoint capability
- [x] 3.3.3 Create `src/utils/checkpoint.py` for state tracking

### 3.4 Validation

- [x] 3.4.1 Create `scripts/validate_extraction.py`
- [x] 3.4.2 Implement extraction quality scoring
- [x] 3.4.3 Implement comparison report generation
- [x] 3.4.4 Add cost reporting to metadata.json

### 3.5 Full Library Processing

- [x] 3.5.1 OCR dependencies installed (Tesseract, Poppler)
- [x] 3.5.2 OCR handler verified working
- [x] 3.5.3 All 167 tests passing

---

## 7. Phase 4: Incremental Updates TODO List ✓

### 4.1 Change Detection

- [x] 4.1.1 Implement Zotero modification date tracking
- [x] 4.1.2 Implement new item detection (items added since last run)
- [x] 4.1.3 Implement modified item detection (items changed since last run)
- [x] 4.1.4 Implement deleted item detection (items in index but not in Zotero)

### 4.2 Update Pipeline

- [x] 4.2.1 Create `scripts/update_index.py` with CLI interface
- [x] 4.2.2 Implement selective reprocessing for modified items
- [x] 4.2.3 Implement addition of new items to index
- [x] 4.2.4 Implement removal of deleted items from index
- [x] 4.2.5 Implement ChromaDB update operations (upsert, delete)

### 4.3 State Management

- [x] 4.3.1 Track last successful update timestamp
- [x] 4.3.2 Implement atomic updates to prevent corruption
- [x] 4.3.3 Write `tests/test_incremental_update.py` with integration tests

---

## 8. Phase 5: Refinement TODO List ✓

### 5.1 Prompt Optimization

- [x] 5.1.1 Review extraction quality across sample papers
- [x] 5.1.2 Identify common extraction failures
- [x] 5.1.3 Iterate on prompt template (v1.1.0 - added keywords field)
- [x] 5.1.4 Document prompt version history

### 5.2 Output Formats

- [x] 5.2.1 Create `scripts/export_results.py`
- [x] 5.2.2 Implement BibTeX export
- [x] 5.2.3 Implement formatted literature review output (Markdown)
- [x] 5.2.4 Implement citation-ready snippets (CSV, JSON exports)

### 5.3 Documentation

- [x] 5.3.1 Create `docs/usage.md` with getting started guide
- [x] 5.3.2 Create `docs/query_guide.md` with query examples
- [x] 5.3.3 Create `docs/troubleshooting.md` with common issues
- [x] 5.3.4 Update README.md with project overview

---

## 9. Dependencies by Phase

### Phase 0-1 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| python | >=3.10 | Runtime |
| pydantic | >=2.0 | Data validation |
| pyyaml | latest | Configuration |
| python-dotenv | latest | Environment variables |
| pymupdf | latest | PDF extraction |
| anthropic | latest | Claude API |
| tqdm | latest | Progress bars |
| pytest | latest | Testing (dev) |

### Phase 2 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| sentence-transformers | latest | Embeddings |
| chromadb | latest | Vector store |
| numpy | latest | Numerical operations |

### Phase 3 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pytesseract | latest | OCR |
| pdf2image | latest | PDF to image |
| Pillow | latest | Image processing |

### System Dependencies (Phase 3)

| Package | Purpose |
|---------|---------|
| Tesseract OCR | OCR engine |
| Poppler | PDF rendering for pdf2image |

---

## 10. Testing Strategy by Phase

### Phase 1 Tests

| Test File | Coverage |
|-----------|----------|
| test_zotero_reader.py | Database connection, queries, path resolution |
| test_pdf_extraction.py | Text extraction, caching, cleaning |
| test_llm_extraction.py | API calls (mocked), parsing, validation |

### Phase 2 Tests

| Test File | Coverage |
|-----------|----------|
| test_embeddings.py | Chunking, embedding generation |
| test_query.py | Search, filtering, ranking |

### Phase 3 Tests

| Test File | Coverage |
|-----------|----------|
| test_ocr.py | OCR fallback, quality detection |

### Phase 4 Tests

| Test File | Coverage |
|-----------|----------|
| test_incremental_update.py | Change detection, updates, deletions |

---

## 11. Critical Path

The following tasks are on the critical path and block subsequent work:

1. **0.2.3 config.py** - All modules depend on configuration
2. **1.2.2 database.py** - PDF extraction needs paths from Zotero
3. **1.3.2 PyMuPDF extraction** - LLM extraction needs text input
4. **1.4.3 llm_client.py** - Section extraction needs API access
5. **1.5.5 papers.json output** - All Phase 2 depends on structured data
6. **2.2.2 ChromaDB initialization** - Query interface needs vector store
7. **4.1.1 Modification tracking** - Updates depend on change detection

---

## 12. Risk Items

| Risk | Impact | Mitigation |
|------|--------|------------|
| PDF extraction fails for many papers | High | Test early with diverse sample |
| LLM extraction quality inconsistent | Medium | Iterate on prompts, validate sample |
| Token costs exceed estimates | Medium | Test with small batch first, track usage |
| ChromaDB performance at scale | Low | Monitor during full library build |
| Zotero database schema changes | Low | Document schema version dependencies |
| OCR quality for scanned PDFs | Medium | Set quality thresholds, allow manual skip |

---

*End of Project Plan*
