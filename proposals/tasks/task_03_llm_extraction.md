# Task 03: LLM-Based Section Extraction

**Phase:** 1 (Foundation)
**Priority:** Critical (Blocking)
**Estimated Effort:** 4-5 hours
**Dependencies:** Task 00 (Setup), Task 02 (PDF Extraction)

---

## Objective

Use the Claude Agent SDK with the Message Batches API to extract structured academic content from paper text, including thesis statements, methodology, key findings, claims, limitations, and future directions.

---

## Prerequisites

- Task 00 completed (configuration with API key)
- Task 02 completed (text extraction working)
- Anthropic API key configured (`ANTHROPIC_API_KEY` environment variable)
- Claude Agent SDK installed (`pip install claude-agent-sdk`)
- Understanding of extraction schema (see technical specification)

---

## Implementation Details

### 03.1 Create Extraction Output Schema

**File:** `src/analysis/schemas.py`

**Purpose:** Pydantic models for LLM extraction output with validation.

**Models Required:**

#### MethodologySection

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| approach | str | Yes | Qualitative, quantitative, mixed, theoretical |
| design | str | No | Case study, experiment, survey, meta-analysis |
| data_sources | list[str] | No | What data was used |
| sample_description | str | No | Sample/population description |
| analysis_methods | list[str] | No | Analytical techniques |
| tools | list[str] | No | Software/instruments mentioned |

#### Finding

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| finding | str | Yes | The finding statement |
| evidence_type | str | No | Statistical, qualitative, theoretical |
| significance | str | No | Why this matters |

#### Claim

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| claim | str | Yes | The specific claim |
| support_type | str | No | How it's supported |
| page_reference | str | No | Approximate location |

#### Extraction

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| paper_id | str | Yes | Links to Paper Metadata |
| extraction_version | str | Yes | Version of prompts used |
| extracted_at | datetime | Yes | Timestamp |
| extraction_model | str | Yes | Model used |
| token_usage | TokenUsage | Yes | Input/output counts |
| thesis_statement | str | No | Core argument |
| research_questions | list[str] | No | Explicit RQs |
| theoretical_framework | str | No | Theoretical grounding |
| methodology | MethodologySection | No | Methods details |
| key_findings | list[Finding] | No | Primary results |
| conclusions | str | No | Author conclusions |
| limitations | list[str] | No | Stated limitations |
| future_directions | list[str] | No | Suggested future work |
| key_claims | list[Claim] | No | Specific claims |
| contribution_summary | str | No | 2-3 sentence summary |
| discipline_tags | list[str] | No | Inferred disciplines |
| extraction_confidence | float | No | 0-1 confidence |
| extraction_notes | str | No | Issues/warnings |

#### TokenUsage

| Field | Type | Description |
|-------|------|-------------|
| input_tokens | int | Tokens in prompt |
| output_tokens | int | Tokens in response |

**Validation:**
- All lists default to empty list
- Confidence score clamped to 0.0-1.0
- extraction_version follows semver format

---

### 03.2 Create Prompt Template

**File:** `src/analysis/prompts.py`

**Purpose:** Store and version extraction prompts.

**Constants:**

```python
EXTRACTION_PROMPT_VERSION = "1.0.0"
```

**Template Structure:**

#### System Prompt

Define the assistant's role as an academic research analyst with expertise in extracting structured information from scholarly papers.

#### User Prompt Template

Structure:
1. Task overview
2. Paper text (inserted)
3. Extraction instructions for each field
4. Output format specification (JSON)
5. Quality guidelines

**Field Extraction Instructions:**

For each field, provide:
- What to look for
- Where in the paper to find it
- How to format the output
- What to do if not found

**Example Instructions (Thesis Statement):**

```
THESIS STATEMENT:
- Look for the main argument or hypothesis
- Usually found in the introduction or abstract
- Should be a clear, falsifiable statement
- If not explicitly stated, infer from research questions
- Output as a single clear sentence
```

**JSON Output Specification:**

Specify exact JSON structure expected, matching Extraction schema.

**Quality Guidelines:**

- Prefer direct quotes over paraphrasing for claims
- Note uncertainty with confidence scores
- Mark missing sections as null, not empty strings
- Include page references where possible

---

### 03.3 Create Claude Batch Client

**File:** `src/analysis/llm_client.py`

**Purpose:** Wrapper for Anthropic Message Batches API for cost-efficient bulk extraction.

**Class: ClaudeBatchClient**

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| api_key | str | From env | Anthropic API key |
| model | str | From config | Model identifier (claude-opus-4-5-20251101) |
| poll_interval | int | 30 | Seconds between batch status checks |
| max_tokens | int | 4096 | Max tokens per response |

**Methods:**

#### `create_batch(requests: list[BatchRequest]) -> str`

**Returns:** `batch_id: str`

**Logic:**

1. Format each request with `custom_id` (paper_id) and `params` (model, messages)
2. Call `client.messages.batches.create(requests=requests)`
3. Return `batch.id` for tracking

#### `wait_for_completion(batch_id: str, timeout_minutes: int = 60) -> BatchStatus`

**Returns:** `BatchStatus` with counts for succeeded, errored, expired

**Logic:**

1. Poll `client.messages.batches.retrieve(batch_id)` every `poll_interval` seconds
2. Check `batch.processing_status` for "ended"
3. Return batch status with request counts
4. Raise timeout error if exceeds `timeout_minutes`

#### `get_results(batch_id: str) -> Iterator[BatchResult]`

**Returns:** Iterator of results (streaming to avoid memory issues)

**Logic:**

1. Stream results via `client.messages.batches.results(batch_id)`
2. Yield each result with `custom_id`, status, and response content
3. Handle succeeded vs errored results

**Batch API Benefits:**

- 50% cost reduction on all tokens
- Automatic handling of transient failures
- Up to 100,000 requests per batch
- 24-hour processing window
- Results retained for 29 days

**Error Handling:**

- Batch-level errors logged and raised
- Individual request failures captured in results
- Timeout handling with graceful degradation

---

### 03.4 Implement Response Parser

**File:** `src/analysis/section_extractor.py`

**Class: ResponseParser**

**Purpose:** Parse and validate LLM JSON responses.

**Methods:**

#### `parse_extraction_response(response_text: str, paper_id: str)`

**Returns:** `Extraction`

**Logic:**
1. Find JSON block in response (may have surrounding text)
2. Parse JSON
3. Validate against schema
4. Create Extraction object
5. Return validated extraction

**JSON Extraction:**
- Look for content between ```json and ```
- Or parse entire response as JSON
- Handle responses that include explanatory text

**Partial Extraction:**
- If some fields fail validation
- Keep valid fields
- Log invalid fields in extraction_notes
- Set extraction_confidence lower

---

### 03.5 Create Section Extractor

**File:** `src/analysis/section_extractor.py`

**Class: SectionExtractor**

**Purpose:** Orchestrate the extraction process.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| client | ClaudeClient | Required | API client |
| prompt_template | str | From prompts.py | Extraction prompt |

**Methods:**

#### `extract(paper_id: str, paper_text: str, metadata: PaperMetadata)`

**Returns:** `Extraction`

**Logic:**
1. Validate text length
2. Truncate if exceeds context limit
3. Construct full prompt with paper text
4. Call Claude API
5. Parse response
6. Attach metadata (paper_id, model, timestamp)
7. Return Extraction object

**Context Management:**

| Model | Context Limit | Safe Limit |
|-------|---------------|------------|
| Claude Opus | 200K tokens | 150K input |
| Claude Sonnet | 200K tokens | 150K input |

**Truncation Strategy:**
- Estimate token count (chars / 4 rough estimate)
- If over limit, prioritize:
  1. Abstract
  2. Introduction (first N pages)
  3. Methods section
  4. Results section
  5. Discussion/Conclusion
- Log that truncation occurred

---

### 03.6 Implement Section Detection

**Purpose:** Pre-process text to identify sections for smart truncation.

**Method:** `detect_sections(text: str)`

**Returns:** `dict[str, tuple[int, int]]` (section_name -> (start, end))

**Common Section Patterns:**
- Abstract: "Abstract", "ABSTRACT", "Summary"
- Introduction: "Introduction", "INTRODUCTION", "1. Introduction"
- Methods: "Method", "Methodology", "Materials and Methods"
- Results: "Results", "Findings"
- Discussion: "Discussion", "Analysis"
- Conclusion: "Conclusion", "Conclusions"
- References: "References", "Bibliography"

**Logic:**
1. Search for section headers using regex
2. Track positions of each section
3. Handle numbered sections (1., 2., etc.)
4. Handle variations in formatting

---

### 03.7 Implement Metadata Enhancement

**Method:** `enhance_with_metadata(extraction: Extraction, metadata: PaperMetadata)`

**Purpose:** Add paper metadata context to extraction.

**Logic:**
1. If abstract extraction empty, use Zotero abstract
2. Validate year consistency
3. Add any relevant metadata fields
4. Return enhanced extraction

---

### 03.8 Implement Batch Extraction

**Method:** `extract_batch(papers: list[tuple[str, str, PaperMetadata]], progress_callback: Callable = None)`

**Returns:** `list[Extraction]`

**Logic:**

1. Prepare batch requests:
   - For each (paper_id, text, metadata), create extraction request
   - Format with system prompt, user prompt containing paper text
   - Set `custom_id` to paper_id for result matching

2. Submit batch:
   - Call `batch_client.create_batch(requests)`
   - Log batch_id for tracking

3. Wait for completion:
   - Poll batch status via `batch_client.wait_for_completion(batch_id)`
   - Report progress via callback

4. Process results:
   - Stream results via `batch_client.get_results(batch_id)`
   - Match results to papers using `custom_id`
   - Parse and validate each response
   - Track success/failure counts

5. Finalize:
   - Log cost summary (with 50% batch discount)
   - Return list of extractions

**Cost Tracking:**

- Sum input_tokens across all extractions
- Sum output_tokens across all extractions
- Calculate estimated cost using model pricing with 50% batch discount
- Log running total

---

## Test Scenarios

### T03.1 API Connection

**Test:** Successfully call Claude API
**Input:** Simple test prompt
**Expected:** Response received
**Verify:** Token usage tracked

### T03.2 Full Extraction

**Test:** Extract all fields from complete paper
**Input:** Well-structured academic paper text
**Expected:** All fields populated
**Verify:** thesis_statement, methodology, findings present

### T03.3 Partial Paper

**Test:** Handle paper missing sections
**Input:** Paper without explicit methods section
**Expected:** methodology.approach set, other fields null
**Verify:** extraction_notes mentions missing sections

### T03.4 JSON Parsing

**Test:** Parse JSON from response
**Input:** Response with JSON block
**Expected:** Extraction object created
**Verify:** All parsed fields match response

### T03.5 Malformed JSON

**Test:** Handle malformed JSON gracefully
**Input:** Response with syntax errors in JSON
**Expected:** Partial extraction with error notes
**Verify:** extraction_confidence lowered

### T03.6 Batch API Submission

**Test:** Successfully submit batch and retrieve results
**Input:** List of 5 paper requests
**Expected:** Batch created, completed, results retrieved
**Verify:** All 5 results returned with custom_ids matching paper_ids

### T03.7 Token Usage Tracking

**Test:** Track tokens accurately
**Input:** Extract from paper
**Expected:** TokenUsage populated
**Verify:** input_tokens and output_tokens > 0

### T03.8 Long Paper Truncation

**Test:** Truncate papers exceeding context
**Input:** Paper with 200K+ characters
**Expected:** Extraction completes
**Verify:** extraction_notes mentions truncation

### T03.9 Section Detection

**Test:** Detect standard sections
**Input:** Paper with clear section headers
**Expected:** All sections identified
**Verify:** Introduction, Methods, Results, Conclusion found

### T03.10 Prompt Template

**Test:** Prompt template produces valid output
**Input:** Known test paper
**Expected:** Consistent extraction format
**Verify:** Output matches schema

### T03.11 Confidence Scoring

**Test:** Confidence reflects extraction quality
**Input:** Clear paper vs ambiguous paper
**Expected:** Clear paper has higher confidence
**Verify:** Scores differ appropriately

### T03.12 Batch Processing

**Test:** Process multiple papers
**Input:** List of 5 papers
**Expected:** All extracted or failures logged
**Verify:** Results list has 5 entries

---

## Caveats and Edge Cases

### Token Limits

- Track both input and output limits
- Leave room for response (4K-8K tokens)
- Long papers may need aggressive truncation

### Model Variability

- Different Claude versions may produce slightly different outputs
- Lock to specific model version in config
- Document expected behavior per model

### Non-Academic Documents

- Some PDFs may not be academic papers
- Reports, presentations, books have different structures
- Adjust extraction based on item_type from Zotero

### Non-English Papers

- Prompt is in English
- May struggle with non-English papers
- Consider language detection and appropriate handling

### Very Short Papers

- Letters, notes may have minimal structure
- Expect many fields to be null
- Don't fail on sparse extraction

### Papers with Multiple Studies

- Some papers report multiple experiments
- methodology and findings may need to accommodate this
- Consider list structure for multi-study papers

### API Cost Control

- Track cumulative cost during batch processing (with 50% batch discount applied)
- Batch API provides predictable costs (submit all at once)
- Monitor batch status for early warning of issues
- Log final costs in metadata.json after batch completes

### Response Format Changes

- Claude may occasionally format JSON differently
- Be flexible in JSON extraction
- Handle with/without markdown code blocks

### Prompt Injection

- Paper text is inserted into prompt
- Malicious papers could attempt prompt injection
- Consider text sanitization if concerned

### Timeout Handling

- Long extractions may timeout
- Implement timeout at request level
- Retry on timeout with smaller chunk

### Caching Extractions

- Don't re-extract papers already processed
- Check existing extractions before API call
- Allow force re-extraction flag

---

## Acceptance Criteria

- [ ] Successfully calls Claude API with retry logic
- [ ] Extracts all schema fields from well-structured papers
- [ ] Handles missing sections gracefully
- [ ] Parses JSON from LLM responses
- [ ] Tracks token usage per extraction
- [ ] Truncates long papers appropriately
- [ ] Detects paper sections for smart truncation
- [ ] Reports extraction confidence
- [ ] Logs errors without crashing
- [ ] All unit tests pass

---

## Files Created

| File | Status |
|------|--------|
| src/analysis/__init__.py | Task 00 |
| src/analysis/schemas.py | Pending |
| src/analysis/prompts.py | Pending |
| src/analysis/llm_client.py | Pending |
| src/analysis/section_extractor.py | Pending |
| tests/test_llm_extraction.py | Pending |

---

*End of Task 03*
