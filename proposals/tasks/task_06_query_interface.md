# Task 06: Query Interface

**Phase:** 2 (Semantic Search)
**Priority:** High
**Estimated Effort:** 3-4 hours
**Dependencies:** Task 04 (Embeddings), Task 05 (Vector Store)

---

## Objective

Create a query interface that enables semantic search across the paper index, with result formatting suitable for LLM consumption and human readability.

---

## Prerequisites

- Task 04 completed (embedding generation)
- Task 05 completed (vector store and structured store)
- All index files populated (papers.json, extractions.json, ChromaDB)

---

## Implementation Details

### 06.1 Create SearchEngine Class

**File:** `src/query/search.py`

**Purpose:** Unified search interface combining semantic and metadata search.

**Class: SearchEngine**

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| vector_store | VectorStore | Required | ChromaDB interface |
| structured_store | StructuredStore | Required | JSON data interface |
| embedding_generator | EmbeddingGenerator | Required | For query embedding |

---

### 06.2 Create Search Request Model

**Model: SearchRequest**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| query | str | Required | Natural language query |
| top_k | int | 10 | Number of results |
| chunk_types | list[str] | None | Filter by chunk type |
| collections | list[str] | None | Filter by collection |
| year_min | int | None | Minimum publication year |
| year_max | int | None | Maximum publication year |
| item_types | list[str] | None | Filter by item type |
| include_embeddings | bool | False | Include vectors in result |

---

### 06.3 Create Search Result Models

**Model: ChunkResult**

| Field | Type | Description |
|-------|------|-------------|
| chunk_id | str | Unique chunk identifier |
| paper_id | str | Parent paper ID |
| chunk_type | str | Type of chunk |
| text | str | Chunk text content |
| score | float | Similarity score (0-1) |
| rank | int | Result rank (1-based) |

**Model: PaperResult**

| Field | Type | Description |
|-------|------|-------------|
| paper_id | str | Paper identifier |
| title | str | Paper title |
| authors | str | Formatted author list |
| year | int | Publication year |
| journal | str | Journal/publication |
| doi | str | DOI if available |
| collections | list[str] | Collection membership |
| best_score | float | Highest chunk score |
| matching_chunks | list[ChunkResult] | Relevant chunks |

**Model: SearchResponse**

| Field | Type | Description |
|-------|------|-------------|
| query | str | Original query |
| total_results | int | Number of papers found |
| results | list[PaperResult] | Ranked paper results |
| search_time_ms | float | Query execution time |
| filters_applied | dict | Active filters |

---

### 06.4 Implement Semantic Search

**Method:** `search(request: SearchRequest)`

**Returns:** `SearchResponse`

**Logic:**
1. Generate embedding for query text
2. Build ChromaDB where clause from filters
3. Execute similarity search
4. Group chunks by paper_id
5. Enrich with paper metadata
6. Rank papers by best chunk score
7. Format and return response

**Filter Translation:**

| Request Field | ChromaDB Where |
|---------------|----------------|
| chunk_types | {"chunk_type": {"$in": [...]}} |
| collections | {"collections": {"$contains": ...}} |
| year_min | {"year": {"$gte": N}} |
| year_max | {"year": {"$lte": N}} |
| item_types | {"item_type": {"$in": [...]}} |

---

### 06.5 Implement Paper Grouping

**Method:** `group_by_paper(chunks: list[ChunkResult], structured_store: StructuredStore)`

**Returns:** `list[PaperResult]`

**Logic:**
1. Group chunks by paper_id
2. For each paper_id:
   - Fetch metadata from structured store
   - Calculate best_score as max of chunk scores
   - Format author list
   - Attach matching chunks
3. Sort papers by best_score descending
4. Return list of PaperResult

---

### 06.6 Implement Multi-Query Search

**Method:** `search_multi(queries: list[str], request: SearchRequest)`

**Purpose:** Search with multiple query variations for broader recall

**Returns:** `SearchResponse`

**Logic:**
1. Execute search for each query
2. Merge results, keeping highest score per paper
3. Re-rank by merged scores
4. Return combined response

**Use Case:** When user query has synonyms or variations

---

### 06.7 Implement Paper Lookup

**Method:** `get_paper_details(paper_id: str)`

**Returns:** `tuple[PaperMetadata, Extraction]`

**Purpose:** Retrieve full paper data after search

**Method:** `get_similar_papers(paper_id: str, top_k: int = 5)`

**Returns:** `list[PaperResult]`

**Purpose:** Find papers similar to a given paper

**Logic:**
1. Get full_summary chunk for paper
2. Use its embedding as query
3. Exclude the source paper from results

---

### 06.8 Create Result Formatter

**File:** `src/query/retrieval.py`

**Purpose:** Format search results for different outputs.

**Class: ResultFormatter**

---

### 06.9 Implement JSON Output

**Method:** `to_json(response: SearchResponse)`

**Returns:** `str`

**Format:** Pretty-printed JSON matching SearchResponse schema

---

### 06.10 Implement Markdown Output

**Method:** `to_markdown(response: SearchResponse)`

**Returns:** `str`

**Format:**

```markdown
# Search Results

**Query:** {query}
**Results:** {total_results} papers
**Search time:** {search_time_ms}ms

## Top Results

### 1. {title} ({year})
**Authors:** {authors}
**Journal:** {journal}
**Score:** {best_score}
**Collections:** {collections}

**Matching Content:**

> {chunk_type}: {chunk_text}

---

### 2. {title} ({year})
...
```

---

### 06.11 Implement Brief Output

**Method:** `to_brief(response: SearchResponse)`

**Returns:** `str`

**Format:** Compact one-line-per-result

```
1. [0.92] Smith et al. (2023) - Title of the paper
2. [0.87] Jones et al. (2022) - Another paper title
3. [0.85] Brown et al. (2021) - Yet another paper
```

---

### 06.12 Implement Citation Output

**Method:** `to_citations(response: SearchResponse)`

**Returns:** `str`

**Format:** Citation-ready format

```
Smith, J., Doe, A., & Johnson, B. (2023). Title of the paper.
Journal Name, 42(3), 100-120. https://doi.org/10.1000/example

Jones, M. (2022). Another paper title. Conference Proceedings, 50-55.
```

---

### 06.13 Create Query CLI Script

**File:** `scripts/query_index.py`

**Purpose:** Command-line interface for querying the index.

**Arguments:**

| Argument | Short | Type | Description |
|----------|-------|------|-------------|
| --query | -q | str | Search query (required) |
| --top-k | -k | int | Number of results (default: 10) |
| --chunk-types | -c | str | Comma-separated chunk types |
| --collections | | str | Comma-separated collections |
| --year-min | | int | Minimum year |
| --year-max | | int | Maximum year |
| --item-types | | str | Comma-separated item types |
| --format | -f | str | Output format: json, markdown, brief, citations |
| --output | -o | str | Output file path (default: stdout) |
| --save | -s | flag | Save to query_results directory |

**Example Usage:**

```bash
# Basic search
python scripts/query_index.py -q "network analysis methods"

# Filtered search
python scripts/query_index.py -q "citation networks" --year-min 2020 --collections "Scientometrics"

# Save results as markdown
python scripts/query_index.py -q "research fronts" -f markdown -s
```

**Output Files:**
- If --save: Write to `data/query_results/query_{timestamp}.{format}`
- Also update `data/query_results/latest.{format}`

---

### 06.14 Implement Interactive Mode

**Optional Enhancement**

**Method:** `interactive_mode()`

**Purpose:** REPL for running multiple queries

**Features:**
- Persistent filters across queries
- History of recent queries
- Quick re-run of previous query
- Exit with 'quit' or Ctrl+C

---

## Test Scenarios

### T06.1 Basic Search

**Test:** Execute simple query
**Input:** query="machine learning"
**Expected:** Returns ranked results
**Verify:** Results contain relevant papers

### T06.2 Empty Results

**Test:** Handle query with no matches
**Input:** query="xyznonexistent123"
**Expected:** Empty results list
**Verify:** total_results = 0, no error

### T06.3 Year Filtering

**Test:** Filter by publication year
**Input:** query="networks", year_min=2020
**Expected:** Only 2020+ papers
**Verify:** All results have year >= 2020

### T06.4 Collection Filtering

**Test:** Filter by collection
**Input:** query="methods", collections=["Network Analysis"]
**Expected:** Only papers in that collection
**Verify:** All results in Network Analysis

### T06.5 Chunk Type Filtering

**Test:** Filter by chunk type
**Input:** query="findings", chunk_types=["finding", "claim"]
**Expected:** Only those chunk types
**Verify:** matching_chunks have correct types

### T06.6 Combined Filters

**Test:** Multiple filters together
**Input:** Year + collection + chunk type
**Expected:** Intersection of all filters
**Verify:** Results match all criteria

### T06.7 Result Ranking

**Test:** Results ordered by relevance
**Input:** Query with clear best match
**Expected:** Best match is rank 1
**Verify:** Scores decrease with rank

### T06.8 Paper Grouping

**Test:** Chunks grouped by paper
**Input:** Query matching multiple chunks per paper
**Expected:** One PaperResult per paper
**Verify:** matching_chunks contains all matches

### T06.9 JSON Format

**Test:** JSON output valid
**Input:** Search with format=json
**Expected:** Valid JSON string
**Verify:** Parses without error

### T06.10 Markdown Format

**Test:** Markdown output readable
**Input:** Search with format=markdown
**Expected:** Properly formatted markdown
**Verify:** Renders correctly

### T06.11 Brief Format

**Test:** Brief output compact
**Input:** Search with format=brief
**Expected:** One line per result
**Verify:** Contains rank, score, author, title

### T06.12 CLI Execution

**Test:** CLI script runs
**Input:** python query_index.py -q "test"
**Expected:** Output to stdout
**Verify:** Results printed

### T06.13 Similar Papers

**Test:** Find similar papers
**Input:** Known paper_id
**Expected:** Related papers returned
**Verify:** Results thematically similar

---

## Caveats and Edge Cases

### Query Length

- Very short queries (1-2 words) may be ambiguous
- Very long queries may not embed well
- Consider query preprocessing

### Score Interpretation

- Similarity scores depend on embedding model
- Not directly comparable across models
- Consider normalizing for display

### Performance at Scale

- Large result sets (top_k=100+) may be slow
- Consider pagination for UI
- Cache frequent queries

### Filter Combinations

- Some filter combinations may return nothing
- Provide feedback on over-filtering
- Consider suggesting relaxed filters

### Collection Name Matching

- Collection names are case-sensitive
- Provide list of valid collections
- Consider fuzzy matching

### Author Name Variations

- Same author may appear differently
- "Smith, J." vs "John Smith"
- Document limitation

### Missing Metadata

- Some papers lack complete metadata
- Handle gracefully in formatting
- Show "Unknown" for missing fields

### Unicode in Queries

- Support non-ASCII characters
- Some embedding models handle this
- Log any encoding issues

### Rate Limiting

- Multiple rapid queries are fine
- But track for monitoring
- Consider query logging

### Result Stability

- Same query should return same results
- Unless index updated
- Deterministic ranking important

---

## Acceptance Criteria

- [ ] Executes semantic search queries
- [ ] Filters by year, collection, chunk type, item type
- [ ] Groups results by paper
- [ ] Ranks papers by relevance
- [ ] Outputs JSON format
- [ ] Outputs Markdown format
- [ ] Outputs Brief format
- [ ] CLI script works with all options
- [ ] Handles empty results gracefully
- [ ] Finds similar papers
- [ ] All unit tests pass

---

## Files Created

| File | Status |
|------|--------|
| src/query/__init__.py | Task 00 |
| src/query/search.py | Pending |
| src/query/retrieval.py | Pending |
| scripts/query_index.py | Pending |
| tests/test_query.py | Pending |

---

*End of Task 06*
