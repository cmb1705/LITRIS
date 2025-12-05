# Task 04: Embedding Generation

**Phase:** 2 (Semantic Search)
**Priority:** High
**Estimated Effort:** 2-3 hours
**Dependencies:** Task 00 (Setup), Task 03 (LLM Extraction)

---

## Objective

Generate vector embeddings from extracted paper content to enable semantic similarity search. Create multiple embedding chunks per paper for granular retrieval.

---

## Prerequisites

- Task 00 completed (configuration)
- Task 03 completed (extractions available)
- sentence-transformers installed

---

## Implementation Details

### 04.1 Create Embedding Generator Class

**File:** `src/indexing/embeddings.py`

**Purpose:** Generate embeddings from text using configurable models.

**Class: EmbeddingGenerator**

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_name | str | From config | Embedding model identifier |
| max_chunk_tokens | int | 512 | Maximum tokens per chunk |
| device | str | 'cpu' | Device for inference |

**Model Loading:**
- Use sentence-transformers for local models
- Support for: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`
- Optional: OpenAI embeddings via API

---

### 04.2 Create Embedding Chunk Model

**Model: EmbeddingChunk**

| Field | Type | Description |
|-------|------|-------------|
| chunk_id | str | Unique identifier |
| paper_id | str | Parent paper ID |
| chunk_type | str | Type of content |
| text | str | Text content |
| embedding | list[float] | Vector representation |

**Chunk Types:**

| Type | Content Source | Purpose |
|------|----------------|---------|
| abstract | Original or extracted abstract | General matching |
| thesis | thesis_statement | Core argument |
| contribution | contribution_summary | What paper adds |
| methodology | Formatted methodology section | Methods matching |
| finding | Individual finding | Result matching |
| claim | Individual claim | Claim retrieval |
| limitation | Individual limitation | Gap analysis |
| future_work | Individual future direction | Opportunity ID |
| full_summary | All content combined | Broad matching |

---

### 04.3 Implement Chunking Strategy

**Method:** `create_chunks(paper_id: str, extraction: Extraction, metadata: PaperMetadata)`

**Returns:** `list[EmbeddingChunk]`

**Chunking Rules:**

1. **Single-field chunks** (one chunk per field):
   - abstract
   - thesis_statement
   - contribution_summary
   - conclusions

2. **Formatted section chunks**:
   - methodology: Combine approach, design, methods into prose
   - theoretical_framework: As-is if present

3. **List-item chunks** (one chunk per item):
   - key_findings: One chunk per Finding
   - key_claims: One chunk per Claim
   - limitations: One chunk per limitation
   - future_directions: One chunk per direction

4. **Aggregate chunks**:
   - full_summary: Combine all extracted content

**Chunk ID Format:**
```
{paper_id}_{chunk_type}_{index}
```
Example: `abc123_finding_2`

---

### 04.4 Implement Text Formatting for Chunks

**Method:** `format_methodology(methodology: MethodologySection)`

**Returns:** `str`

**Format:**
```
This study uses a {approach} approach with a {design} design.
Data sources include: {data_sources}.
The sample consists of {sample_description}.
Analysis methods: {analysis_methods}.
Tools used: {tools}.
```

**Method:** `format_finding(finding: Finding)`

**Returns:** `str`

**Format:**
```
Finding: {finding}
Evidence type: {evidence_type}
Significance: {significance}
```

**Method:** `format_claim(claim: Claim)`

**Returns:** `str`

**Format:**
```
Claim: {claim}
Support: {support_type}
Location: {page_reference}
```

---

### 04.5 Implement Embedding Generation

**Method:** `embed_text(text: str)`

**Returns:** `list[float]`

**Logic:**
1. Tokenize text
2. Truncate if exceeds max tokens
3. Generate embedding via model
4. Return as list of floats

**Method:** `embed_batch(texts: list[str])`

**Returns:** `list[list[float]]`

**Logic:**
1. Batch texts for efficiency
2. Generate embeddings in parallel
3. Return list of embeddings

**Batch Size:**
- Default: 32 texts per batch
- Configurable based on memory

---

### 04.6 Implement Full Paper Embedding

**Method:** `embed_paper(paper_id: str, extraction: Extraction, metadata: PaperMetadata)`

**Returns:** `list[EmbeddingChunk]`

**Logic:**
1. Create all chunks for paper
2. Extract texts from chunks
3. Batch embed all texts
4. Attach embeddings to chunks
5. Return complete chunks

---

### 04.7 Implement Batch Paper Embedding

**Method:** `embed_papers(papers: list[tuple[str, Extraction, PaperMetadata]], progress_callback: Callable = None)`

**Returns:** `list[EmbeddingChunk]`

**Logic:**
1. Create chunks for all papers
2. Collect all texts across papers
3. Batch embed all texts at once
4. Distribute embeddings back to chunks
5. Call progress callback after each paper
6. Return all chunks

**Memory Management:**
- Process in paper batches if memory constrained
- Clear model cache between large batches
- Track total chunks generated

---

### 04.8 Implement Metadata Attachment

**Method:** `attach_metadata(chunk: EmbeddingChunk, metadata: PaperMetadata)`

**Returns:** `dict` (metadata for vector store)

**Attached Metadata:**

| Field | Source | Purpose |
|-------|--------|---------|
| paper_id | chunk.paper_id | Linking |
| chunk_type | chunk.chunk_type | Filtering |
| title | metadata.title | Display |
| authors | Joined author names | Display/filter |
| year | metadata.publication_year | Filtering |
| collections | Joined collection names | Filtering |
| item_type | metadata.item_type | Filtering |
| doi | metadata.doi | Reference |

---

## Test Scenarios

### T04.1 Model Loading

**Test:** Load embedding model successfully
**Input:** Valid model name from config
**Expected:** Model loaded, ready to embed
**Verify:** Can embed sample text

### T04.2 Single Text Embedding

**Test:** Embed single text string
**Input:** "This is a test sentence"
**Expected:** Vector of correct dimension
**Verify:** Dimension matches model spec (384 or 768)

### T04.3 Batch Embedding

**Test:** Embed multiple texts at once
**Input:** List of 10 texts
**Expected:** List of 10 vectors
**Verify:** Each vector has correct dimension

### T04.4 Chunk Creation

**Test:** Create chunks from extraction
**Input:** Complete extraction with all fields
**Expected:** Multiple chunks created
**Verify:** Has abstract, thesis, findings, claims chunks

### T04.5 Empty Field Handling

**Test:** Handle extraction with missing fields
**Input:** Extraction with null thesis_statement
**Expected:** No thesis chunk created
**Verify:** Other chunks still created

### T04.6 Finding Chunks

**Test:** Create one chunk per finding
**Input:** Extraction with 3 findings
**Expected:** 3 finding chunks
**Verify:** Each has unique chunk_id

### T04.7 Methodology Formatting

**Test:** Format methodology section as prose
**Input:** MethodologySection with all fields
**Expected:** Readable paragraph
**Verify:** Contains approach, design, methods

### T04.8 Long Text Truncation

**Test:** Handle text exceeding max tokens
**Input:** Very long abstract (1000+ tokens)
**Expected:** Truncated before embedding
**Verify:** No error, embedding generated

### T04.9 Metadata Attachment

**Test:** Correct metadata attached to chunk
**Input:** Chunk with associated PaperMetadata
**Expected:** Metadata dict populated
**Verify:** Has title, authors, year, collections

### T04.10 Full Paper Embedding

**Test:** Embed all chunks for one paper
**Input:** Complete paper with extraction
**Expected:** List of chunks with embeddings
**Verify:** All expected chunk types present

### T04.11 Batch Paper Processing

**Test:** Embed multiple papers efficiently
**Input:** 5 papers with extractions
**Expected:** All chunks for all papers
**Verify:** Correct paper_id on each chunk

---

## Caveats and Edge Cases

### Model Dimensions

| Model | Dimensions |
|-------|------------|
| all-MiniLM-L6-v2 | 384 |
| all-mpnet-base-v2 | 768 |
| text-embedding-3-small | 1536 |

- Store dimension with index metadata
- Verify consistency across all embeddings

### GPU Availability

- sentence-transformers can use GPU
- Check CUDA availability at startup
- Fall back to CPU gracefully
- Log device being used

### Memory Usage

- Large batch sizes consume more memory
- Monitor memory during batch processing
- Reduce batch size if OOM errors occur
- Consider streaming to disk for very large libraries

### Empty Extractions

- Some papers may have minimal extraction
- Still create abstract chunk from Zotero metadata
- Log papers with very few chunks

### Duplicate Text

- Same text may appear in multiple fields
- Consider deduplication within paper
- Or allow duplicates for different chunk types

### Model Versioning

- Embedding models may update
- Store model version with index
- Invalidate embeddings if model changes
- Document which version was used

### Token vs Character Limits

- Models have token limits, not character
- Use tokenizer to count accurately
- rough estimate: 4 chars per token

### Non-English Text

- Multilingual models available if needed
- Default models work best for English
- Document language limitations

### Null Embeddings

- If embedding fails, don't store None
- Skip chunk entirely
- Log the failure

### Vector Normalization

- Some models return normalized vectors
- Some require normalization for cosine similarity
- Check model documentation
- Normalize if needed for consistent scoring

---

## Acceptance Criteria

- [ ] Loads sentence-transformers model successfully
- [ ] Generates embeddings for text strings
- [ ] Creates multiple chunks per paper
- [ ] Handles all extraction chunk types
- [ ] Formats methodology and findings as prose
- [ ] Truncates long texts appropriately
- [ ] Attaches correct metadata to chunks
- [ ] Batch embeds efficiently
- [ ] Handles missing/empty fields gracefully
- [ ] All unit tests pass

---

## Files Created

| File | Status |
|------|--------|
| src/indexing/__init__.py | Task 00 |
| src/indexing/embeddings.py | Pending |
| tests/test_embeddings.py | Pending |

---

*End of Task 04*
