# Extraction Cascade, Classification Index, and Build Gating

**Date**: 2026-03-14
**Status**: Design
**Epic**: LITRIS-32u
**Related**: LITRIS-6r5 (dead code audit)

## Problem

LITRIS spends LLM tokens extracting every document in the corpus, including presentations, syllabi, course slides, and other non-academic content unlikely to yield useful semantic analysis. The extraction pipeline also uses only PyMuPDF plain-text extraction, despite having fully implemented (but unwired) support for arXiv HTML, ar5iv, Marker PDF-to-Markdown, and a multi-tier cascade orchestrator. This results in:

1. Wasted tokens on non-academic documents (~13% of the corpus are duplicates or non-publications)
2. Loss of structural information (tables, figures, equations) that higher-quality extraction tiers would preserve
3. No way to preview what the pipeline will extract before committing to expensive LLM calls

## Architecture: Classification-First Pipeline

Classification runs as a fast, standalone pre-pass using only metadata and PyMuPDF text heuristics (no LLM, no network calls). The extraction cascade activates only during actual LLM extraction, providing the best possible text to the 6-pass pipeline.

```
classify-only:  Reference source → Tier 1 classify (metadata) → PyMuPDF text → Tier 2 classify (heuristics) → classification_index.json
build_index:    Load classification_index → filter by academic flag → cascade extract (best text) → LLM 6-pass → index
```

Classification is source-agnostic. All reference adapters (Zotero, BibTeX, Paperpile, EndNote, Mendeley, PDF folder) produce `PaperMetadata` objects through the `BaseReferenceDB` interface.

---

## Feature 1: Classification Index

### Data Model

File: `data/index/classification_index.json`

```json
{
  "schema_version": "1.0.0",
  "classified_at": "2026-03-14T10:00:00",
  "stats": {
    "total": 1561,
    "by_type": {
      "RESEARCH_PAPER": 1102,
      "REVIEW_PAPER": 87,
      "BOOK_MONOGRAPH": 198,
      "THESIS": 34,
      "REPORT": 62,
      "REFERENCE_MATERIAL": 26,
      "NON_ACADEMIC": 52
    },
    "extractable_count": 1340,
    "non_extractable_count": 221
  },
  "papers": {
    "<paper_id>": {
      "title": "GraphCast: Learning skillful medium-range global weather...",
      "item_type": "journalArticle",
      "document_type": "RESEARCH_PAPER",
      "confidence": 0.90,
      "tier": 1,
      "reasons": ["item_type=journalArticle", "doi_present", "journal_present"],
      "extractable": true,
      "word_count": 15234,
      "page_count": 28,
      "section_markers": 7,
      "classified_at": "2026-03-14T10:00:01"
    }
  }
}
```

### Extractable Logic

A paper is `extractable: true` when ALL of:
- `document_type` is not `NON_ACADEMIC`
- `word_count` >= `min_publication_words` (default: 500)
- `page_count` >= `min_publication_pages` (default: 2)

Thresholds are sourced from existing `ProcessingConfig`.

### New Module: `src/analysis/classification_store.py`

Small, focused class (~100 lines). Responsibilities:

- `load() -> ClassificationIndex` -- read from disk
- `save(index: ClassificationIndex)` -- write to disk with stats recomputed
- `classify_paper(paper: PaperMetadata, text: str | None) -> ClassificationRecord` -- run Tier 1, optionally Tier 2
- `get_paper(paper_id: str) -> ClassificationRecord | None`
- `get_extractable_ids() -> set[str]` -- all paper IDs where `extractable == true`
- `get_stats() -> dict` -- summary statistics

Uses existing `classify_metadata()` and `classify_text()` from `document_classifier.py`. No new classification logic needed.

### CLI: `build_index.py --classify-only`

- Loads all papers from the configured reference source
- For each paper not already in the index:
  - Tier 1: `classify_metadata(paper)` (instant, no PDF)
  - If confidence < `high_confidence_threshold` (default 0.8, from existing `classify()` parameter): extract PDF text with PyMuPDF, run `classify_text()`
- Writes/updates `classification_index.json`
- Prints summary table to terminal
- `--reclassify` flag forces re-evaluation of all papers (replaces all records)

**Incremental limitations (v1)**: Incremental mode does not detect metadata changes since last classification or prune papers removed from the reference source. Both are addressable in a future version by comparing `date_modified` timestamps and performing set-difference pruning. For now, `--reclassify` is the escape hatch.

Expected performance: ~5-15 minutes for full 1,561-paper corpus (most papers get high-confidence Tier 1 classification from item_type alone; Tier 2 PyMuPDF fallback triggers for a minority). Zero LLM tokens, zero network calls.

---

## Feature 2: Build Index Gating

### New CLI Flags

| Flag | Behavior |
|------|----------|
| `--index-academic-only` | Default. Skip papers where `extractable == false` in classification index. |
| `--index-all` | Override. Extract all papers regardless of classification. |

No flag needed for the default -- `--index-academic-only` is implied.

### Pipeline Integration

When `build_index.py` runs extraction (without `--classify-only`):

1. Load `classification_index.json` if it exists
2. For papers NOT in the index: classify on-the-fly, append to index
3. Apply filter:
   - Default (academic-only): keep only `extractable == true`
   - `--index-all`: keep everything
4. Log skip summary: "Skipping N papers (M non-academic, K insufficient text). Use --index-all to include."
5. Proceed with cascade extraction on remaining papers
6. Save updated classification index (includes any on-the-fly classifications)

When classification_index.json does NOT exist:
- Log: "No classification index found. Running inline classification. Use --classify-only for a preview."
- Classify each paper before extraction, building the index as a side effect

### End-of-Run Report

```
Classification summary:
  RESEARCH_PAPER:     1102 (71%)
  BOOK_MONOGRAPH:      198 (13%)
  REVIEW_PAPER:         87  (6%)
  REPORT:               62  (4%)
  THESIS:               34  (2%)
  REFERENCE_MATERIAL:   26  (2%)
  NON_ACADEMIC:         52  (3%)

Skipped 221 papers:
  NON_ACADEMIC (confidence >= 0.8):  143
  Insufficient text (<500 words):     52
  Too short (<2 pages):               26

Extracting: 1340 papers
```

### Source Agnostic

The gating logic operates on `PaperMetadata` objects from `BaseReferenceDB`. The current hardcoded `ZoteroDatabase` call in `build_index.py` should be replaced with the adapter abstraction via `create_reference_db()` from `src/references/factory.py`. The six adapters (Zotero, BibTeX, Paperpile, EndNote, Mendeley, PDF folder) already produce `PaperMetadata`.

**Note on `--collection` filtering**: The existing `--collection` substring filter relies on `PaperMetadata.collections`, which is populated by Zotero. Non-Zotero adapters may return empty collections. The `--collection` flag should be silently ignored (with a warning log) when using adapters that do not support collections. This preserves backward compatibility while enabling source-agnostic operation.

---

## Feature 3: Extraction Cascade

### Current State

`src/extraction/cascade.py` contains a fully implemented `ExtractionCascade` class. It is never instantiated. `arxiv_extractor.py` and `marker_extractor.py` are fully implemented. They are never called. `SectionExtractor` calls `PDFExtractor` directly at line 338.

### Tier Ordering

| Priority | Tier | Source | Output Format | Speed | Condition |
|----------|------|--------|---------------|-------|-----------|
| 1 | Companion | `.md` file alongside PDF | Markdown | Instant | File exists |
| 2 | arXiv HTML | `arxiv.org/html/{id}` | HTML to text | ~2s | arXiv ID in metadata |
| 3 | ar5iv | `ar5iv.labs.arxiv.org/html/{id}` | HTML to text | ~2s | arXiv tier fails |
| 4 | Marker | ML-based PDF to Markdown | Markdown | 5-15s | `marker-pdf` installed |
| 5 | PyMuPDF | Direct PDF text extraction | Plain text | <1s | Always available |
| 6 | Tesseract OCR | Image-based text recognition | Plain text | 10-30s | PyMuPDF insufficient text |

### Companion Tier (New)

Checks for a markdown file with the same stem as the PDF:
- `paper.pdf` checks for `paper.md`
- Optionally checks a separate `companion_dir` if configured
- If found: read file, validate word count >= threshold, return as markdown
- Simple file existence check, no external dependencies

**Code changes required**:
- Add `"companion"` to the `CascadeMethod` Literal type in `cascade.py` (currently only has `arxiv_html`, `ar5iv`, `marker`, `pymupdf`, `ocr`, `hybrid`)
- Add `is_markdown: bool` field to existing `CascadeResult` dataclass (default `False`; set `True` for `companion` and `marker` methods)
- Add companion tier logic to `ExtractionCascade.extract_text()` as Priority 1 (before arXiv)

### Wiring Change

In `SectionExtractor.__init__()`:
- If `cascade_enabled` (default `true`): instantiate `ExtractionCascade`
- If `cascade_enabled == false`: use `PDFExtractor` directly (current behavior, backward compat)

In `SectionExtractor.extract_paper()`:
- Replace `self.pdf_extractor.extract_text_with_method(pdf_path)` with `self.cascade.extract_text(pdf_path, metadata)`
- Returns `CascadeResult` dataclass (fields: `text`, `method`, `word_count`, `tiers_attempted`, `is_markdown`)
- Pass `config.processing.min_extraction_words` to `ExtractionCascade(min_words=...)` constructor (replaces the hardcoded `MIN_EXTRACTION_WORDS = 100` constant)

### Markdown Preservation

`TextCleaner.clean()` gains a new `preserve_markdown: bool = False` parameter. This is a **new parameter** (does not currently exist).

When `preserve_markdown=True` (Companion or Marker tiers):
- Skip short-line filtering (`clean()` lines 79-84) that destroys markdown list items and table rows
- Skip `HEADER_FOOTER` regex removal (line 72) that could remove legitimate markdown headers
- Keep `HYPHEN_LINEBREAK` and `PAGE_NUMBERS` removal (safe for markdown)
- Preserve table formatting, heading hierarchy, LaTeX equations, code blocks

When `preserve_markdown=False` (default, arXiv/PyMuPDF/OCR tiers):
- `TextCleaner` operates as today
- No behavior change

`SectionExtractor` passes `preserve_markdown=result.is_markdown` to `TextCleaner.clean()`.

### Extraction Method Tracking

The cascade reports which tier succeeded via `CascadeResult.method`. This is stored in:
- `classification_index.json`: `extraction_method` field per paper (written after extraction completes)
- `ExtractionResult`: Add new **optional** field `extraction_method: str | None = None` to the Pydantic model in `schemas.py`. This is a schema addition, not a modification of existing fields.
- Enables queries like "how many papers used Marker vs PyMuPDF?"

### Configuration

New fields in `config.yaml` under `processing` (alongside existing `min_publication_words`, `min_publication_pages`, `ocr_enabled`):

```yaml
processing:
  cascade_enabled: true          # Use cascade (false = PyMuPDF only)
  companion_dir: null             # Optional directory for pre-extracted .md files
  arxiv_enabled: true             # Enable arXiv HTML tiers (requires network)
  marker_enabled: true            # Enable Marker tier (requires marker-pdf)
```

The existing `min_extraction_words` field in `ProcessingConfig` is used as the cascade quality gate (passed to `ExtractionCascade(min_words=config.processing.min_extraction_words)`). No new config field needed for this.

### Dependencies

- `marker-pdf`: optional, silently skipped if not installed. Add to `requirements.txt` with `# optional` comment.
- `pytesseract`, `pdf2image`, `Pillow`: already in requirements.
- `PyMuPDF`: already in requirements.
- `beautifulsoup4`: **deferred**. The existing arXiv extractor uses regex and works. Add bs4 in a future iteration if regex parsing proves insufficient.

Install missing dependencies:
```bash
pip install marker-pdf
```

---

## Error Handling

### Classification Errors

If PDF text extraction fails during classification (corrupt PDF, encrypted file):
- Classify with Tier 1 only (metadata)
- Record: `tier: 1`, `word_count: null`, `classification_error: "<message>"`
- Still marked `extractable` based on metadata alone
- The cascade handles the PDF problem during actual extraction

### Cascade Errors

Each tier catches its own exceptions:
- arXiv: `requests.Timeout`, `requests.ConnectionError` -- fall through
- Marker: `ImportError` (not installed), processing errors -- fall through
- PyMuPDF: corrupt PDF -- fall through to OCR
- OCR: Tesseract not found, conversion failure -- return error

If all tiers fail, the paper gets an `ExtractionResult` with `success=False` and `error="All extraction tiers failed"`. Same error path as today.

### Backward Compatibility

- `cascade_enabled: false` preserves exact current behavior
- `--index-all` preserves extract-everything behavior
- No existing config keys change meaning
- No migration needed for existing indexes

---

## Testing Strategy

| Area | Tests | Count |
|------|-------|-------|
| ClassificationStore load/save/update/filter | Unit | ~8 |
| ClassificationStore error paths (corrupt index, missing file) | Unit | ~3 |
| Cascade tier fallthrough and quality gate | Unit (mocked tiers) | ~6 |
| Companion tier file discovery | Unit | ~2 |
| TextCleaner markdown preservation | Unit | ~3 |
| Build index gating (academic-only vs all) | Integration | ~4 |
| Build index inline classification fallback (no index file) | Integration | ~2 |
| `--reclassify` flag behavior | Integration | ~2 |
| ExtractionResult `extraction_method` field | Unit | ~1 |
| **Total** | | **~31** |

---

## Implementation Order

1. **Install dependencies** -- `marker-pdf`
2. **Wire extraction cascade** -- replace `PDFExtractor` with `ExtractionCascade` in `SectionExtractor`, add Companion tier, add config fields, add `preserve_markdown` to `TextCleaner`
3. **Build classification store** -- new `classification_store.py` module, `--classify-only` flag
4. **Build index gating** -- `--index-academic-only` / `--index-all` flags, inline classification fallback, skip summary report
5. **Wire BaseReferenceDB** -- replace hardcoded `ZoteroDatabase` in `build_index.py` with `create_reference_db()` from adapter factory; handle `--collection` filter for non-Zotero adapters (warn + skip)
6. **Tests** -- unit and integration tests for all three features, including error paths and edge cases

Steps 2-4 are the core features. Step 5 is a moderate refactor (adapter wiring + collection filter compatibility). Step 1 is a prerequisite.

**Concurrency note**: `classification_index.json` is a single-writer file. Concurrent `build_index.py` runs writing to the same index are not supported. This is consistent with the existing single-writer assumption for `papers.json`.

---

## Files Modified

| File | Change |
|------|--------|
| `src/analysis/classification_store.py` | **NEW** -- classification index read/write/query |
| `src/extraction/cascade.py` | Add Companion tier, add `is_markdown` to `CascadeResult`, add `"companion"` to `CascadeMethod` |
| `src/analysis/schemas.py` | Add `extraction_method: str | None = None` field to `ExtractionResult` |
| `src/extraction/text_cleaner.py` | Add `preserve_markdown` parameter |
| `src/analysis/section_extractor.py` | Wire cascade, pass markdown flag to cleaner |
| `scripts/build_index.py` | Add `--classify-only`, `--index-all`, `--reclassify` flags; load classification index; wire `BaseReferenceDB` |
| `src/config.py` | Add cascade config fields to `ProcessingConfig` |
| `config.yaml` | Add cascade fields under `processing` section |
| `requirements.txt` | Add `marker-pdf` (optional) |
| `tests/test_classification_store.py` | **NEW** |
| `tests/test_cascade_wiring.py` | **NEW** |
| `tests/test_build_gating.py` | **NEW** |
