# Task 07: Robustness and Error Handling

**Phase:** 3 (Robustness)
**Priority:** Medium-High
**Estimated Effort:** 4-5 hours
**Dependencies:** Tasks 01-06 (All core functionality)

---

## Objective

Add OCR fallback for scanned PDFs, comprehensive error handling, failed item tracking, checkpoint/resume capability, and validation tools for production-quality processing.

---

## Prerequisites

- Tasks 01-06 completed (core pipeline working)
- Tesseract OCR installed (system dependency)
- Poppler installed (for pdf2image)

---

## Implementation Details

### 07.1 Create OCR Handler

**File:** `src/extraction/ocr_handler.py`

**Purpose:** OCR fallback for PDFs where text extraction fails.

**Class: OCRHandler**

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tesseract_path | str | None | Path to Tesseract (if not in PATH) |
| dpi | int | 300 | Resolution for PDF rendering |
| language | str | 'eng' | Tesseract language |

**System Dependencies:**

| Dependency | Installation | Purpose |
|------------|--------------|---------|
| Tesseract | `choco install tesseract` (Windows) | OCR engine |
| Poppler | `choco install poppler` (Windows) | PDF to image |

---

### 07.2 Implement PDF to Image Conversion

**Method:** `pdf_to_images(pdf_path: Path)`

**Returns:** `list[Image]`

**Logic:**
1. Use pdf2image to convert PDF
2. Convert each page to PIL Image
3. Apply preprocessing (if needed)
4. Return list of images

**Preprocessing Options:**
- Grayscale conversion
- Contrast enhancement
- Deskewing (optional)

---

### 07.3 Implement OCR Extraction

**Method:** `ocr_images(images: list[Image])`

**Returns:** `str`

**Logic:**
1. For each image:
   - Run Tesseract OCR
   - Extract text
   - Add page markers
2. Combine all pages
3. Return full text

**Tesseract Configuration:**
- Use `--oem 3` (LSTM + Legacy)
- Use `--psm 3` (Automatic page segmentation)
- Output as text, not hOCR

---

### 07.4 Implement Quality Detection

**Method:** `needs_ocr(extraction_result: ExtractionResult)`

**Returns:** `bool`

**Criteria for OCR Trigger:**

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| quality_score | < 0.3 | Very poor extraction |
| chars_per_page | < 100 | Almost no text extracted |
| word_count | < 50 | Minimal content |

**Additional Checks:**
- High ratio of non-printable characters
- Very short average word length (garbled text)
- Extracted text is mostly whitespace

---

### 07.5 Integrate OCR into PDFExtractor

**Update:** `src/extraction/pdf_extractor.py`

**New Method:** `extract_with_ocr(pdf_path: Path)`

**Updated Logic for `extract()`:**
1. Attempt PyMuPDF extraction
2. Calculate quality score
3. If quality < threshold AND ocr_enabled:
   - Attempt OCR extraction
   - Compare qualities
   - Use better result
4. Cache final result
5. Return with method indicator

**Configuration:**
- `options.ocr_enabled`: Toggle OCR fallback
- `options.ocr_threshold`: Quality threshold for trigger

---

### 07.6 Implement Failed Item Tracking

**File:** `src/utils/error_tracking.py`

**Purpose:** Track and manage processing failures.

**Model: FailedItem**

| Field | Type | Description |
|-------|------|-------------|
| paper_id | str | Paper identifier |
| zotero_key | str | Zotero key for reference |
| title | str | Paper title |
| stage | str | Where failure occurred |
| error_type | str | Exception type |
| error_message | str | Error details |
| attempts | int | Number of retry attempts |
| first_failure | datetime | When first failed |
| last_attempt | datetime | Most recent attempt |
| skip | bool | Permanently skip this item |

**Stage Values:**
- `pdf_not_found`: PDF file doesn't exist
- `pdf_extraction`: PyMuPDF failed
- `ocr_extraction`: OCR failed
- `llm_extraction`: Claude API failed
- `embedding`: Embedding generation failed
- `indexing`: Vector store insertion failed

**Class: ErrorTracker**

**Methods:**

| Method | Purpose |
|--------|---------|
| `record_failure(paper_id, stage, error)` | Log a failure |
| `record_success(paper_id)` | Remove from failures |
| `get_failures(stage=None)` | Get failed items |
| `get_retry_candidates()` | Items to retry |
| `mark_skip(paper_id)` | Permanently skip |
| `save()` | Persist to metadata.json |
| `load()` | Load from metadata.json |

---

### 07.7 Implement Checkpoint System

**Purpose:** Save progress during long runs, enable resume.

**File:** `src/utils/checkpoint.py`

**Model: ProcessingCheckpoint**

| Field | Type | Description |
|-------|------|-------------|
| run_id | str | Unique run identifier |
| started_at | datetime | Run start time |
| stage | str | Current stage |
| total_items | int | Total to process |
| processed_items | list[str] | Completed paper_ids |
| current_item | str | In-progress paper_id |
| last_checkpoint | datetime | When last saved |

**Class: CheckpointManager**

**Methods:**

| Method | Purpose |
|--------|---------|
| `create_checkpoint(run_id)` | Start new run |
| `load_checkpoint(run_id)` | Resume existing |
| `mark_item_complete(paper_id)` | Record progress |
| `mark_item_started(paper_id)` | Track current |
| `save()` | Persist checkpoint |
| `get_remaining_items(all_items)` | Items left to process |

**Checkpoint File:** `data/cache/processing_state.json`

**Auto-Save Interval:** Every N items (default: 10)

---

### 07.8 Update Build Script with Robustness

**Updates to:** `scripts/build_index.py`

**New Arguments:**

| Argument | Description |
|----------|-------------|
| `--resume` | Resume from last checkpoint |
| `--retry-failed` | Retry previously failed items |
| `--skip-paper ID` | Skip specific paper |
| `--max-failures N` | Stop after N failures |
| `--checkpoint-interval N` | Save every N items |
| `--no-ocr` | Disable OCR fallback |

**Enhanced Logic:**

1. **Initialization:**
   - Check for existing checkpoint if --resume
   - Load error tracker
   - Determine items to process

2. **Processing Loop:**
   - Wrap each paper in try/except
   - Record failures to error tracker
   - Save checkpoint periodically
   - Check failure threshold

3. **Completion:**
   - Save final checkpoint
   - Report failures summary
   - Clean up checkpoint if complete

---

### 07.9 Implement Validation Script

**File:** `scripts/validate_extraction.py`

**Purpose:** Quality check extractions and generate reports.

**Features:**

1. **Extraction Coverage:**
   - Papers with extractions vs without
   - Fields populated per extraction
   - Average confidence scores

2. **Quality Metrics:**
   - Extractions with low confidence
   - Papers with very few chunks
   - Missing critical fields

3. **Comparison Tools:**
   - Compare extraction to original abstract
   - Flag potential mismatches
   - Identify unusual extractions

**Output:** Validation report in markdown format

---

### 07.10 Implement Cost Tracking

**Purpose:** Track API costs for budget management.

**Additions to ProcessingStats:**

| Field | Type | Description |
|-------|------|-------------|
| total_input_tokens | int | Sum of input tokens |
| total_output_tokens | int | Sum of output tokens |
| estimated_cost_usd | float | Calculated cost |
| cost_by_stage | dict | Breakdown by stage |
| tokens_per_paper | float | Average per paper |

**Cost Calculation:**

| Model | Input $/M | Output $/M |
|-------|-----------|------------|
| claude-3-opus | $15 | $75 |
| claude-3-sonnet | $3 | $15 |
| claude-3-haiku | $0.25 | $1.25 |

**Reporting:**
- Log running cost during processing
- Include in metadata.json
- Display in validation report

---

### 07.11 Implement Graceful Degradation

**Purpose:** Handle partial failures without stopping entire run.

**Strategies:**

1. **PDF Extraction Failure:**
   - Try OCR fallback
   - If still fails, log and continue
   - Create paper entry without extraction

2. **LLM Extraction Failure:**
   - Retry with exponential backoff
   - After max retries, save partial extraction
   - Log for manual review

3. **Embedding Failure:**
   - Log specific paper
   - Continue with other papers
   - Index will have gaps

4. **Vector Store Failure:**
   - Retry insertion
   - If persistent, save to separate file
   - Allow manual insertion later

---

## Test Scenarios

### T07.1 OCR Handler Loading

**Test:** OCR handler initializes correctly
**Input:** System with Tesseract installed
**Expected:** Handler ready
**Verify:** Can access Tesseract

### T07.2 OCR Text Extraction

**Test:** Extract text from scanned PDF
**Input:** Known scanned PDF
**Expected:** Readable text extracted
**Verify:** Contains expected content

### T07.3 OCR Quality Detection

**Test:** Detect when OCR is needed
**Input:** Scanned PDF with PyMuPDF failure
**Expected:** needs_ocr() returns True
**Verify:** OCR fallback triggered

### T07.4 Failed Item Recording

**Test:** Record processing failure
**Input:** Paper that fails LLM extraction
**Expected:** Added to failed items
**Verify:** Appears in get_failures()

### T07.5 Failed Item Retry

**Test:** Retry failed items
**Input:** Previously failed paper now working
**Expected:** Removed from failures on success
**Verify:** Not in failures after retry

### T07.6 Checkpoint Creation

**Test:** Create checkpoint during processing
**Input:** Process 10 papers
**Expected:** Checkpoint saved
**Verify:** File exists with correct content

### T07.7 Resume from Checkpoint

**Test:** Resume interrupted run
**Input:** Checkpoint with 5/10 complete
**Expected:** Only process remaining 5
**Verify:** Skips completed papers

### T07.8 Skip Specific Paper

**Test:** Skip paper by ID
**Input:** --skip-paper with known ID
**Expected:** Paper excluded from processing
**Verify:** Not attempted

### T07.9 Max Failures Threshold

**Test:** Stop after N failures
**Input:** --max-failures 3 with failing papers
**Expected:** Stops after 3 failures
**Verify:** Processing halted, checkpoint saved

### T07.10 Validation Report

**Test:** Generate validation report
**Input:** Completed index
**Expected:** Report with metrics
**Verify:** Contains coverage, quality stats

### T07.11 Cost Tracking

**Test:** Track API costs
**Input:** Process papers with LLM
**Expected:** Cost calculated
**Verify:** estimated_cost_usd populated

### T07.12 Graceful Degradation

**Test:** Continue after single failure
**Input:** Mix of good and bad PDFs
**Expected:** Good PDFs processed
**Verify:** Failures logged, successes indexed

---

## Caveats and Edge Cases

### Tesseract Installation

- Not installed by default on any OS
- Windows: Use chocolatey or download installer
- Add to PATH or specify path in config
- Test OCR availability at startup

### Poppler Installation

- Required for pdf2image
- Windows: Add bin/ to PATH
- Common source of "unable to find" errors

### OCR Quality Variability

- Depends heavily on scan quality
- Low DPI scans produce poor results
- Consider increasing DPI for better results
- Some PDFs may remain unreadable

### Memory Usage for OCR

- Each page becomes an image in memory
- 300 DPI page ~= 25MB uncompressed
- Process page by page for large PDFs
- Implement memory monitoring

### Checkpoint Corruption

- Interrupted writes can corrupt JSON
- Use atomic write (temp file then rename)
- Validate checkpoint on load
- Provide checkpoint repair option

### Concurrent Processing

- Current design is single-threaded
- Checkpoint system assumes single process
- For parallel processing, need different approach

### Error Message Sensitivity

- Error messages may contain file paths
- Avoid logging sensitive information
- Sanitize paths in error output

### Retry Storm

- If many papers fail, retrying all is slow
- Consider batch retry with limits
- Priority queue for retry order

### Cost Overruns

- Large libraries can be expensive
- Implement cost ceiling option
- Pause and prompt user if exceeded
- Consider cheaper models for retry

### Validation False Positives

- Some papers legitimately have sparse content
- Short papers, letters, notes
- Don't flag all low-content as failures

---

## Acceptance Criteria

- [ ] OCR extracts text from scanned PDFs
- [ ] OCR triggers automatically when needed
- [ ] Failed items tracked across runs
- [ ] Can retry failed items selectively
- [ ] Checkpoint saves during processing
- [ ] Can resume from checkpoint
- [ ] Validation report generated
- [ ] Cost tracking accurate
- [ ] Processing continues after failures
- [ ] All unit tests pass

---

## Files Created

| File | Status |
|------|--------|
| src/extraction/ocr_handler.py | Pending |
| src/utils/error_tracking.py | Pending |
| src/utils/checkpoint.py | Pending |
| scripts/validate_extraction.py | Pending |
| tests/test_ocr.py | Pending |

---

## System Dependencies

| Dependency | Install Command (Windows) |
|------------|---------------------------|
| Tesseract | `choco install tesseract` |
| Poppler | `choco install poppler` |

**Add to PATH:**
- Tesseract: `C:\Program Files\Tesseract-OCR`
- Poppler: `C:\path\to\poppler\bin`

---

*End of Task 07*
