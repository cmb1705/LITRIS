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
    "by_extractable": {"true": 1340, "false": 221}
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
  - If confidence < 0.8: extract PDF text with PyMuPDF, run `classify_text()`
- Writes/updates `classification_index.json`
- Prints summary table to terminal
- `--reclassify` flag forces re-evaluation of all papers

Expected performance: ~30-60 minutes for full 1,561-paper corpus (dominated by PyMuPDF text extraction for Tier 2 candidates). Zero LLM tokens, zero network calls.

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

The gating logic operates on `PaperMetadata` objects from `BaseReferenceDB`. The current hardcoded `ZoteroDatabase` call in `build_index.py` (line 702) should be replaced with the adapter abstraction. The six adapters (Zotero, BibTeX, Paperpile, EndNote, Mendeley, PDF folder) already produce `PaperMetadata`. This is a mechanical wiring change, not a design change.

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

### Wiring Change

In `SectionExtractor.__init__()`:
- If `cascade_enabled` (default `true`): instantiate `ExtractionCascade`
- If `cascade_enabled == false`: use `PDFExtractor` directly (current behavior, backward compat)

In `SectionExtractor.extract_paper()`:
- Replace `self.pdf_extractor.extract_text_with_method(pdf_path)` with `self.cascade.extract(pdf_path, metadata)`
- Cascade returns `(text, method_used, is_markdown)` tuple

### Markdown Preservation

When the cascade produces markdown (Companion or Marker tiers):
- `TextCleaner` receives `preserve_markdown=True`
- Skip line-length filtering and header stripping that destroys markdown structure
- Preserve table formatting, heading hierarchy, LaTeX equations, code blocks
- `SectionExtractor` passes `preserve_markdown=True` when `is_markdown` is reported by cascade

When the cascade produces plain text (arXiv, PyMuPDF, OCR tiers):
- `TextCleaner` operates as today (default `preserve_markdown=False`)
- No behavior change

### Extraction Method Tracking

The cascade reports which tier succeeded. This is stored in:
- `classification_index.json`: `extraction_method` field per paper
- `ExtractionResult` metadata: `extraction_method` field
- Enables queries like "how many papers used Marker vs PyMuPDF?"

### Configuration

New fields in `config.yaml` under `extraction`:

```yaml
extraction:
  cascade_enabled: true          # Use cascade (false = PyMuPDF only)
  companion_dir: null             # Optional directory for pre-extracted .md files
  arxiv_enabled: true             # Enable arXiv HTML tiers (requires network)
  marker_enabled: true            # Enable Marker tier (requires marker-pdf)
  min_extraction_words: 100       # Quality gate between tiers
```

### Dependencies

- `marker-pdf`: optional, silently skipped if not installed. Add to `requirements.txt` with `# optional` comment.
- `beautifulsoup4`: for better arXiv HTML parsing (current implementation uses regex). Add to `requirements.txt`.
- `pytesseract`, `pdf2image`, `Pillow`: already in requirements.
- `PyMuPDF`: already in requirements.

Install missing dependencies:
```bash
pip install marker-pdf beautifulsoup4
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
| Cascade tier fallthrough and quality gate | Unit (mocked tiers) | ~6 |
| Build index gating (academic-only vs all) | Integration | ~4 |
| Companion tier file discovery | Unit | ~2 |
| TextCleaner markdown preservation | Unit | ~3 |
| **Total** | | **~23** |

---

## Implementation Order

1. **Install dependencies** -- `marker-pdf`, `beautifulsoup4`
2. **Wire extraction cascade** -- replace `PDFExtractor` with `ExtractionCascade` in `SectionExtractor`, add Companion tier, add config fields, add `preserve_markdown` to `TextCleaner`
3. **Build classification store** -- new `classification_store.py` module, `--classify-only` flag
4. **Build index gating** -- `--index-academic-only` / `--index-all` flags, inline classification fallback, skip summary report
5. **Wire BaseReferenceDB** -- replace hardcoded `ZoteroDatabase` in `build_index.py` with adapter abstraction
6. **Tests** -- unit and integration tests for all three features

Steps 2-4 are the core features. Step 5 is a mechanical wiring change. Step 1 is a prerequisite.

---

## Files Modified

| File | Change |
|------|--------|
| `src/analysis/classification_store.py` | **NEW** -- classification index read/write/query |
| `src/extraction/cascade.py` | Add Companion tier, ensure proper return tuple |
| `src/extraction/text_cleaner.py` | Add `preserve_markdown` parameter |
| `src/analysis/section_extractor.py` | Wire cascade, pass markdown flag to cleaner |
| `scripts/build_index.py` | Add `--classify-only`, `--index-all`, `--reclassify` flags; load classification index; wire `BaseReferenceDB` |
| `src/config.py` | Add cascade config fields |
| `config.yaml` | Add cascade section with defaults |
| `requirements.txt` | Add `marker-pdf`, `beautifulsoup4` |
| `tests/test_classification_store.py` | **NEW** |
| `tests/test_cascade_wiring.py` | **NEW** |
| `tests/test_build_gating.py` | **NEW** |
