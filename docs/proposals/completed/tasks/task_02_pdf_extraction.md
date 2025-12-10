# Task 02: PDF Text Extraction

**Phase:** 1 (Foundation)
**Priority:** Critical (Blocking)
**Estimated Effort:** 3-4 hours
**Dependencies:** Task 00 (Setup), Task 01 (Zotero Reader)

---

## Objective

Extract readable text from PDF files using PyMuPDF, with support for Zotero's existing full-text cache, text cleaning and normalization, and a caching layer to avoid re-processing.

---

## Prerequisites

- Task 00 completed (configuration system)
- Task 01 completed (PDF paths available)
- PyMuPDF installed (`pip install pymupdf`)

---

## Implementation Details

### 02.1 Create PDFExtractor Class

**File:** `src/extraction/pdf_extractor.py`

**Purpose:** Extract text content from PDF files with quality metrics.

**Class: PDFExtractor**

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| cache_dir | Path | From config | Directory for text cache |
| use_zotero_cache | bool | True | Check for .zotero-ft-cache first |

---

### 02.2 Extraction Result Model

**Create dataclass or Pydantic model:**

**ExtractionResult**

| Field | Type | Description |
|-------|------|-------------|
| text | str | Extracted text content |
| page_count | int | Total pages in PDF |
| char_count | int | Total characters extracted |
| word_count | int | Total words extracted |
| extraction_method | str | 'zotero_cache', 'pymupdf', 'ocr' |
| quality_score | float | 0.0-1.0 quality estimate |
| warnings | list[str] | Any issues encountered |
| source_path | Path | Original PDF path |
| cached | bool | Whether result was from cache |

---

### 02.3 Implement Zotero Cache Check

**Method:** `check_zotero_cache(pdf_path: Path)`

**Returns:** `str | None`

**Logic:**
1. Derive storage folder from PDF path
2. Check for `.zotero-ft-cache` file
3. If exists, read and return contents
4. If not exists or empty, return None

**Cache Location:**
- PDF at: `{storage}/{key}/filename.pdf`
- Cache at: `{storage}/{key}/.zotero-ft-cache`

**Validation:**
- Check cache file is non-empty
- Compare page counts if `.zotero-ft-info` exists
- Return None if cache appears incomplete

---

### 02.4 Implement PyMuPDF Extraction

**Method:** `extract_with_pymupdf(pdf_path: Path)`

**Returns:** `ExtractionResult`

**Logic:**
1. Open PDF with `fitz.open(pdf_path)`
2. Iterate through pages
3. Extract text from each page using `page.get_text()`
4. Join pages with delimiter (e.g., `\n\n--- Page X ---\n\n`)
5. Calculate quality metrics
6. Close document

**Text Extraction Options:**
- Use `"text"` format for plain text
- Consider `"blocks"` for structured extraction
- Handle rotated pages

**Error Handling:**
- Catch `fitz.FileDataError` for corrupted PDFs
- Catch `fitz.PasswordError` for encrypted PDFs
- Log and return empty result with appropriate warning

---

### 02.5 Implement Quality Metrics

**Method:** `calculate_quality_score(text: str, page_count: int)`

**Returns:** `float` (0.0 to 1.0)

**Metrics to Consider:**

| Metric | Weight | Good Value | Bad Value |
|--------|--------|------------|-----------|
| Chars per page | 0.3 | >1000 | <100 |
| Word length avg | 0.2 | 4-8 chars | <2 or >15 |
| Alphanumeric ratio | 0.3 | >0.7 | <0.3 |
| Whitespace ratio | 0.2 | 0.1-0.3 | >0.5 or <0.05 |

**Thresholds:**
- Score < 0.3: Likely needs OCR
- Score 0.3-0.6: Partial extraction, may have issues
- Score > 0.6: Good extraction

**Detection of Extraction Failure:**
- Almost no text but multiple pages
- Mostly non-ASCII characters
- Extremely short words (garbled text)

---

### 02.6 Implement Page Boundary Markers

**Purpose:** Preserve page structure for later reference.

**Format Options:**

Option A: Simple markers
```
[Page 1]
Text from page 1...

[Page 2]
Text from page 2...
```

Option B: Minimal markers
```
<<1>>
Text from page 1...

<<2>>
Text from page 2...
```

**Recommendation:** Use Option B for smaller footprint, document format in extraction result.

---

### 02.7 Create Text Cleaner

**File:** `src/extraction/text_cleaner.py`

**Purpose:** Normalize and clean extracted text.

**Class: TextCleaner**

**Methods:**

#### `clean(text: str) -> str`

Main cleaning pipeline, applies all cleaners in order.

#### `normalize_unicode(text: str) -> str`

- Apply NFKC normalization
- Replace common Unicode variants with ASCII
- Handle ligatures (fi, fl, ff, ffi, ffl)

#### `fix_hyphenation(text: str) -> str`

- Detect line-ending hyphens
- Join hyphenated words across lines
- Preserve intentional hyphens

#### `normalize_whitespace(text: str) -> str`

- Replace multiple spaces with single space
- Normalize line endings to `\n`
- Remove trailing whitespace
- Preserve paragraph breaks (double newline)

#### `remove_headers_footers(text: str) -> str`

- Detect repeated lines at page boundaries
- Remove likely headers/footers
- Preserve content that happens to repeat

#### `fix_common_artifacts(text: str) -> str`

- Fix broken words from column layouts
- Handle common OCR substitutions (rn -> m, etc.)
- Remove null characters and control characters

---

### 02.8 Implement Caching Layer

**Method:** `get_cached_extraction(paper_id: str)`

**Returns:** `ExtractionResult | None`

**Cache Structure:**
```
data/cache/pdf_text/
  {paper_id}.txt       # Cleaned text
  {paper_id}.json      # Extraction metadata
```

**Metadata JSON Contents:**
- extraction_method
- quality_score
- page_count
- char_count
- word_count
- extracted_at (timestamp)
- source_pdf_hash (for invalidation)
- warnings

**Method:** `cache_extraction(paper_id: str, result: ExtractionResult)`

**Logic:**
1. Write text to `{paper_id}.txt`
2. Write metadata to `{paper_id}.json`
3. Use atomic write (write to temp, then rename)

**Method:** `invalidate_cache(paper_id: str)`

**Logic:**
1. Delete text file if exists
2. Delete metadata file if exists

---

### 02.9 Implement Main Extraction Method

**Method:** `extract(pdf_path: Path, paper_id: str, force: bool = False)`

**Returns:** `ExtractionResult`

**Logic:**
1. If not force, check local cache
   - If cached and valid, return cached result
2. If use_zotero_cache, check Zotero cache
   - If found, clean text and return
3. Extract with PyMuPDF
4. Clean extracted text
5. Calculate quality metrics
6. Cache result
7. Return ExtractionResult

**Force Parameter:**
- When True, skip cache and re-extract
- Useful for re-processing after improvements

---

### 02.10 Implement Batch Extraction

**Method:** `extract_batch(papers: list[PaperMetadata], progress_callback: Callable = None)`

**Returns:** `dict[str, ExtractionResult]` (paper_id -> result)

**Logic:**
1. For each paper with pdf_path:
   - Extract text
   - Track success/failure
   - Call progress callback if provided
2. Return results dictionary

**Progress Callback Signature:**
```python
def callback(current: int, total: int, paper_title: str, status: str)
```

---

## Test Scenarios

### T02.1 PyMuPDF Extraction

**Test:** Extract text from valid PDF
**Input:** Sample academic PDF with selectable text
**Expected:** Text extracted with reasonable quality score
**Verify:** Key phrases from paper present in output

### T02.2 Zotero Cache Usage

**Test:** Use existing Zotero cache when available
**Input:** PDF with .zotero-ft-cache file present
**Expected:** extraction_method is 'zotero_cache'
**Verify:** Content matches cache file

### T02.3 Cache Bypass

**Test:** Extract fresh when Zotero cache missing
**Input:** PDF without .zotero-ft-cache
**Expected:** extraction_method is 'pymupdf'
**Verify:** Text extracted successfully

### T02.4 Quality Score Calculation

**Test:** Quality score reflects extraction quality
**Input:** PDF with good selectable text
**Expected:** quality_score > 0.6
**Verify:** Score decreases for known-bad PDFs

### T02.5 Low Quality Detection

**Test:** Detect PDFs that need OCR
**Input:** Scanned PDF with no selectable text
**Expected:** quality_score < 0.3
**Verify:** Warning indicates OCR may be needed

### T02.6 Text Cleaning - Ligatures

**Test:** Ligatures converted to standard characters
**Input:** Text containing "ﬁnd" and "ﬂow"
**Expected:** Converted to "find" and "flow"
**Verify:** No ligature characters in output

### T02.7 Text Cleaning - Hyphenation

**Test:** Hyphenated line breaks rejoined
**Input:** "infor-\nmation" in extracted text
**Expected:** Converted to "information"
**Verify:** Word is whole, no orphan hyphen

### T02.8 Text Cleaning - Whitespace

**Test:** Whitespace normalized
**Input:** Text with multiple spaces and mixed line endings
**Expected:** Single spaces, consistent newlines
**Verify:** No double spaces except paragraph breaks

### T02.9 Local Caching

**Test:** Extraction cached for reuse
**Input:** Extract same PDF twice
**Expected:** Second call returns cached result
**Verify:** cached=True in second result

### T02.10 Cache Invalidation

**Test:** Cache can be invalidated
**Input:** Invalidate then re-extract
**Expected:** Fresh extraction performed
**Verify:** cached=False after invalidation

### T02.11 Encrypted PDF Handling

**Test:** Encrypted PDFs handled gracefully
**Input:** Password-protected PDF
**Expected:** Empty text with warning
**Verify:** No exception raised

### T02.12 Corrupted PDF Handling

**Test:** Corrupted PDFs handled gracefully
**Input:** Truncated or invalid PDF file
**Expected:** Empty text with warning
**Verify:** No exception raised

### T02.13 Missing PDF Handling

**Test:** Missing PDF returns appropriate result
**Input:** Non-existent file path
**Expected:** None or error result
**Verify:** FileNotFoundError logged

### T02.14 Page Markers

**Test:** Page boundaries preserved
**Input:** Multi-page PDF
**Expected:** Page markers in output
**Verify:** Can identify text from specific pages

---

## Caveats and Edge Cases

### Multi-Column Layouts

- Academic papers often have two columns
- PyMuPDF extracts left-to-right, then top-to-bottom
- May result in jumbled text across columns
- Consider using `"blocks"` extraction and sorting by position

### Tables and Figures

- Tables extract poorly as text
- Figures have no text content
- Table captions may be useful
- Consider detecting and marking table regions

### Mathematical Notation

- Equations may not extract correctly
- Unicode math symbols might be garbled
- Document this limitation
- Consider marking mathematical sections

### Non-English PDFs

- PyMuPDF handles Unicode well
- Some fonts may not have proper mappings
- Log character encoding issues

### Very Large PDFs

- Some papers are 100+ pages
- Memory usage scales with page count
- Consider page-by-page extraction for large files
- Implement timeout for extraction

### PDF/A vs Standard PDF

- PDF/A (archival) format is well-supported
- Older PDF versions may have issues
- Log PDF version for debugging

### Scanned PDF Detection

Beyond low character count, detect:
- All text is in images
- Text is not selectable in reader
- Low text-to-page ratio

### Header/Footer False Positives

- Don't remove content that legitimately repeats
- Only remove if appears at page boundaries
- Be conservative with removal

### Cache Corruption

- Validate cache files before using
- Handle JSON parse errors
- Re-extract if cache invalid

### Concurrent Access

- If running multiple extractions in parallel
- Use file locking for cache writes
- Or use separate cache per process

---

## Acceptance Criteria

- [x] Extracts text from standard academic PDFs
- [x] Uses Zotero cache when available
- [x] Calculates meaningful quality score
- [x] Cleans text (ligatures, hyphenation, whitespace)
- [x] Caches extractions for reuse
- [x] Handles encrypted PDFs without crashing
- [x] Handles corrupted PDFs without crashing
- [x] Preserves page boundary information
- [x] All unit tests pass

---

## Files Created

| File | Status |
|------|--------|
| src/extraction/__init__.py | Complete |
| src/extraction/pdf_extractor.py | Complete |
| src/extraction/text_cleaner.py | Complete |
| tests/test_pdf_extraction.py | Complete |
| tests/fixtures/sample_papers/*.pdf | Skipped (tests use generated PDFs) |

---

*End of Task 02*
