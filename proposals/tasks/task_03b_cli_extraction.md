# Task 03b: CLI-Based LLM Extraction (Subscription Mode)

**Phase:** 1 (Foundation)
**Priority:** Critical (Alternative to Task 03)
**Estimated Effort:** 3-4 hours
**Dependencies:** Task 00 (Setup), Task 02 (PDF Extraction)

---

## Objective

Use Claude Code CLI in headless mode to extract structured academic content from paper text, leveraging your Max subscription at no additional cost beyond the monthly fee.

---

## Prerequisites

- Task 00 completed (configuration)
- Task 02 completed (text extraction working)
- Claude Code CLI installed and authenticated via `claude login`
- **No `ANTHROPIC_API_KEY` environment variable set** (this triggers API billing)
- Max subscription (5x or 20x plan)
- Understanding of extraction schema (see technical specification)

---

## Key Differences from Batch API (Task 03)

| Aspect | CLI Mode (This Task) | Batch API (Task 03) |
|--------|---------------------|---------------------|
| Cost | Free (subscription) | ~$67.50 for 500 papers |
| Speed | Sequential (~30s/paper) | Parallel (~1hr total) |
| Rate Limits | 200-800/5hr (Max 20) | No practical limit |
| Authentication | `claude login` | `ANTHROPIC_API_KEY` |
| Best For | Budget, incremental | Speed, bulk builds |

---

## Implementation Details

### 03b.1 Create CLI Executor

**File:** `src/analysis/cli_executor.py`

**Purpose:** Execute Claude Code CLI commands and parse responses.

**Class: ClaudeCliExecutor**

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| timeout | int | 120 | Seconds before command timeout |
| output_format | str | "json" | CLI output format |

**Methods:**

#### `verify_authentication() -> bool`

**Purpose:** Ensure CLI is using subscription, not API billing.

**Logic:**

1. Check if `ANTHROPIC_API_KEY` environment variable is set
2. If set, raise error with instructions to unset it
3. Run `claude --version` to verify CLI is installed
4. Return True if ready

#### `extract(prompt: str, input_text: str) -> dict`

**Returns:** Parsed JSON response from Claude

**Logic:**

1. Write `input_text` to temporary file
2. Execute: `cat temp_file | claude -p "{prompt}" --output-format json`
3. Parse JSON from stdout
4. Clean up temporary file
5. Return parsed response

**Error Handling:**

- Timeout: Raise `ExtractionTimeoutError`
- Rate limit: Raise `RateLimitError` with reset time if available
- Invalid JSON: Raise `ParseError` with raw output
- CLI errors: Raise `CliExecutionError`

---

### 03b.2 Create Rate Limit Handler

**File:** `src/analysis/rate_limit_handler.py`

**Purpose:** Monitor and handle Max subscription rate limits.

**Class: RateLimitHandler**

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| pause_on_limit | bool | True | Pause vs fail on limit |
| auto_resume | bool | False | Auto-resume after reset |
| check_interval | int | 60 | Seconds between limit checks |

**Methods:**

#### `check_response_for_limit(response: str) -> bool`

**Purpose:** Detect rate limit indicators in CLI output.

**Indicators:**

- "rate limit" in response (case insensitive)
- "usage limit" in response
- "try again" with time indication
- Non-zero exit code with limit message

#### `handle_limit_hit()`

**Logic:**

1. Log rate limit event with timestamp
2. If `auto_resume`:
   - Calculate next reset time (5 hours from first request)
   - Sleep until reset
   - Return True to continue
3. If `pause_on_limit`:
   - Save current progress
   - Print user message with resume instructions
   - Exit gracefully
4. Else:
   - Raise `RateLimitExceededError`

#### `get_estimated_reset_time() -> datetime`

**Logic:**

- Rate limits reset every 5 hours
- Track first request time in session
- Calculate next reset window

---

### 03b.3 Create Progress Tracker

**File:** `src/analysis/progress_tracker.py`

**Purpose:** Track extraction progress for pause/resume capability.

**Class: ProgressTracker**

**Progress File:** `data/cache/cli_progress.json`

**Schema:**

```json
{
  "started_at": "ISO8601",
  "last_updated": "ISO8601",
  "total_papers": 500,
  "completed": ["paper_id_1", "paper_id_2"],
  "failed": [
    {
      "paper_id": "paper_id_3",
      "error": "Error message",
      "timestamp": "ISO8601"
    }
  ],
  "current_session": {
    "started_at": "ISO8601",
    "requests_this_session": 45
  }
}
```

**Methods:**

#### `load() -> ProgressState`

Load existing progress or create new state.

#### `mark_completed(paper_id: str)`

Add paper to completed list, update timestamp.

#### `mark_failed(paper_id: str, error: str)`

Add paper to failed list with error details.

#### `get_pending_papers(all_papers: list[str]) -> list[str]`

Return papers not in completed or failed lists.

#### `save()`

Write current state to progress file.

#### `get_session_request_count() -> int`

Count requests in current session for rate limit awareness.

---

### 03b.4 Implement CLI Section Extractor

**File:** `src/analysis/cli_section_extractor.py`

**Class: CliSectionExtractor**

**Purpose:** Orchestrate CLI-based extraction with rate limiting and progress tracking.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| executor | ClaudeCliExecutor | Required | CLI execution handler |
| rate_handler | RateLimitHandler | Required | Rate limit handler |
| progress | ProgressTracker | Required | Progress tracker |
| prompt_template | str | From prompts.py | Extraction prompt |

**Methods:**

#### `extract_single(paper_id: str, paper_text: str, metadata: PaperMetadata) -> Extraction`

**Logic:**

1. Construct prompt with paper text
2. Call `executor.extract()`
3. Parse response into Extraction object
4. Validate against schema
5. Return extraction

#### `extract_all(papers: list[tuple[str, str, PaperMetadata]], resume: bool = True) -> list[Extraction]`

**Logic:**

1. If resume, load progress and filter to pending papers
2. For each paper:
   - Check session request count
   - If approaching limit, handle proactively
   - Call `extract_single()`
   - On success: `progress.mark_completed()`
   - On failure: `progress.mark_failed()`
   - Save progress after each paper
3. Log summary statistics
4. Return all extractions

---

### 03b.5 Update Build Script

**File:** `scripts/build_index.py`

**Modifications:**

Add CLI mode support:

```python
# Determine extraction mode from config
mode = config.extraction.mode  # "cli" or "batch_api"

if mode == "cli":
    from src.analysis.cli_section_extractor import CliSectionExtractor
    extractor = CliSectionExtractor(...)
    extractions = extractor.extract_all(papers, resume=args.resume)
else:
    from src.analysis.section_extractor import SectionExtractor
    extractor = SectionExtractor(...)
    extractions = extractor.extract_batch(papers)
```

**New CLI Arguments:**

| Argument | Description |
|----------|-------------|
| `--mode` | Override config extraction mode |
| `--resume` | Resume from last checkpoint (CLI mode) |
| `--reset-progress` | Clear progress and start fresh |

---

## Test Scenarios

### T03b.1 CLI Authentication Check

**Test:** Verify subscription authentication
**Setup:** Unset `ANTHROPIC_API_KEY`
**Expected:** `verify_authentication()` returns True
**Verify:** No API key warnings

### T03b.2 CLI Authentication Failure

**Test:** Detect API key presence
**Setup:** Set `ANTHROPIC_API_KEY`
**Expected:** `verify_authentication()` raises error
**Verify:** Error message explains issue

### T03b.3 Single Extraction

**Test:** Extract from one paper via CLI
**Input:** Sample paper text
**Expected:** Valid Extraction object
**Verify:** All expected fields populated

### T03b.4 JSON Parsing

**Test:** Parse CLI JSON output
**Input:** Valid JSON response
**Expected:** Correct parsing
**Verify:** Fields match response

### T03b.5 Progress Save/Load

**Test:** Progress persistence
**Setup:** Complete 5 extractions
**Expected:** Progress file updated
**Verify:** 5 papers in completed list

### T03b.6 Resume from Checkpoint

**Test:** Resume interrupted extraction
**Setup:** Progress file with 5 completed
**Input:** 10 total papers
**Expected:** Only 5 papers processed
**Verify:** Skipped completed papers

### T03b.7 Rate Limit Detection

**Test:** Detect rate limit response
**Input:** Response with "rate limit" message
**Expected:** `check_response_for_limit()` returns True
**Verify:** Rate limit handler triggered

### T03b.8 Graceful Pause on Limit

**Test:** Pause when limit hit
**Setup:** `pause_on_limit: true`
**Input:** Rate limit response
**Expected:** Progress saved, clean exit
**Verify:** Can resume later

### T03b.9 Timeout Handling

**Test:** Handle slow responses
**Setup:** 120s timeout
**Input:** Simulated slow response
**Expected:** Timeout error raised
**Verify:** Paper marked as failed

### T03b.10 Full Extraction Run

**Test:** Process multiple papers
**Input:** 10 papers
**Expected:** All extracted or progress saved
**Verify:** Extractions match schema

---

## Caveats and Edge Cases

### Environment Variable Conflicts

- **Critical:** If `ANTHROPIC_API_KEY` is set, CLI will use API billing
- Always check and warn at startup
- Provide clear instructions to unset

### Rate Limit Variability

- Limits vary by plan and usage patterns
- 50-200 prompts/5hr for Max 5x
- 200-800 prompts/5hr for Max 20x
- Track requests conservatively

### Session Management

- CLI invocations are stateless
- Each extraction is independent
- No conversation context between papers

### Output Format Reliability

- CLI `--output-format json` should produce valid JSON
- Handle cases where extra text appears
- Look for JSON delimiters if parsing fails

### Long Papers

- CLI has same context limits as API
- Truncate papers before sending
- Same truncation logic as Task 03

### Concurrent Execution

- Do not run multiple CLI extractions in parallel
- Sequential processing only
- Parallel would hit rate limits faster

### Progress File Corruption

- Validate progress file on load
- Backup before modifications
- Handle missing or invalid gracefully

---

## Acceptance Criteria

- [ ] CLI authentication verification works
- [ ] Single paper extraction via CLI succeeds
- [ ] JSON responses parse correctly
- [ ] Progress tracking saves/loads correctly
- [ ] Resume from checkpoint works
- [ ] Rate limit detection triggers appropriately
- [ ] Graceful pause preserves progress
- [ ] Build script supports `--mode cli`
- [ ] All unit tests pass

---

## Files Created

| File | Status |
|------|--------|
| src/analysis/cli_executor.py | Pending |
| src/analysis/rate_limit_handler.py | Pending |
| src/analysis/progress_tracker.py | Pending |
| src/analysis/cli_section_extractor.py | Pending |
| tests/test_cli_extraction.py | Pending |

---

## Usage Examples

### First Run (Full Build)

```bash
# Ensure no API key set
$env:ANTHROPIC_API_KEY = $null

# Run CLI-based extraction
python scripts/build_index.py --mode cli
```

### Resume After Rate Limit

```bash
# Continue from where we left off
python scripts/build_index.py --mode cli --resume
```

### Reset and Start Fresh

```bash
# Clear progress and start over
python scripts/build_index.py --mode cli --reset-progress
```

### Check Progress

```bash
# View current progress
cat data/cache/cli_progress.json | python -m json.tool
```

---

*End of Task 03b*
