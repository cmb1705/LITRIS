# Task 05: Vector Store (ChromaDB)

**Phase:** 2 (Semantic Search)
**Priority:** High
**Estimated Effort:** 2-3 hours
**Dependencies:** Task 04 (Embeddings)

---

## Objective

Implement a ChromaDB-based vector store for persisting embeddings and enabling similarity search with metadata filtering.

---

## Prerequisites

- Task 04 completed (embedding generation)
- ChromaDB installed (`pip install chromadb`)

---

## Implementation Details

### 05.1 Create VectorStore Class

**File:** `src/indexing/vector_store.py`

**Purpose:** Manage ChromaDB collection for paper embeddings.

**Class: VectorStore**

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| persist_dir | Path | From config | Directory for database |
| collection_name | str | 'paper_chunks' | ChromaDB collection name |
| embedding_dim | int | 384 | Vector dimension |

**Initialization:**
1. Create ChromaDB client with persistence
2. Get or create collection
3. Verify embedding dimension matches existing

---

### 05.2 Implement Collection Management

**Method:** `initialize()`

**Logic:**
1. Check if collection exists
2. If exists, verify dimension compatibility
3. If new, create with specified settings
4. Store creation metadata

**Method:** `get_collection_stats()`

**Returns:** `dict`

**Contents:**
- total_documents: int
- unique_papers: int
- chunk_type_counts: dict
- embedding_dimension: int
- created_at: datetime

**Method:** `clear_collection()`

**Purpose:** Remove all documents (for rebuild)

---

### 05.3 Implement Document Insertion

**Method:** `add_chunks(chunks: list[EmbeddingChunk], metadata_list: list[dict])`

**Logic:**
1. Extract IDs from chunks (chunk_id)
2. Extract embeddings as lists
3. Extract documents (text content)
4. Extract metadatas
5. Use ChromaDB `collection.add()`

**Batch Handling:**
- ChromaDB handles batching internally
- For very large inserts, chunk into groups of 5000
- Log progress for large inserts

**Duplicate Handling:**
- ChromaDB raises error on duplicate IDs
- Check for existing IDs before insert
- Or use upsert method

**Method:** `upsert_chunks(chunks: list[EmbeddingChunk], metadata_list: list[dict])`

**Purpose:** Add or update chunks (for incremental updates)

---

### 05.4 Implement Similarity Search

**Method:** `search(query_embedding: list[float], n_results: int = 10, where: dict = None, where_document: dict = None)`

**Returns:** `SearchResults`

**SearchResults Model:**

| Field | Type | Description |
|-------|------|-------------|
| ids | list[str] | Chunk IDs |
| documents | list[str] | Text content |
| metadatas | list[dict] | Associated metadata |
| distances | list[float] | Similarity scores |
| embeddings | list[list[float]] | Optional embeddings |

**Where Filter Examples:**
```python
# Filter by year
{"year": {"$gte": 2020}}

# Filter by collection
{"collections": {"$contains": "Network Analysis"}}

# Filter by chunk type
{"chunk_type": "thesis"}

# Combined filters
{"$and": [
    {"year": {"$gte": 2020}},
    {"chunk_type": "finding"}
]}
```

**Method:** `search_by_text(query_text: str, embedding_generator: EmbeddingGenerator, n_results: int = 10, where: dict = None)`

**Purpose:** Convenience method that embeds query first

---

### 05.5 Implement Document Retrieval

**Method:** `get_by_id(chunk_id: str)`

**Returns:** `tuple[str, dict, list[float]]` (document, metadata, embedding)

**Method:** `get_by_paper_id(paper_id: str)`

**Returns:** `list[tuple[str, dict, list[float]]]`

**Purpose:** Retrieve all chunks for a specific paper

**Method:** `get_by_chunk_type(chunk_type: str, limit: int = 100)`

**Returns:** `list[tuple[str, dict]]`

**Purpose:** Retrieve chunks of specific type across all papers

---

### 05.6 Implement Deletion

**Method:** `delete_by_id(chunk_id: str)`

**Purpose:** Remove single chunk

**Method:** `delete_by_paper_id(paper_id: str)`

**Purpose:** Remove all chunks for a paper (for updates or removal)

**Logic:**
1. Query for all chunk IDs with matching paper_id
2. Delete all matching IDs

**Method:** `delete_by_ids(chunk_ids: list[str])`

**Purpose:** Batch delete multiple chunks

---

### 05.7 Implement Index Persistence

**Persistence Details:**
- ChromaDB auto-persists to directory
- Directory: `data/index/chroma/`
- Contains: `chroma.sqlite3`, embeddings files

**Method:** `persist()`

**Purpose:** Force persistence (though ChromaDB auto-persists)

**Method:** `get_persist_directory()`

**Returns:** `Path`

---

### 05.8 Create StructuredStore Class

**File:** `src/indexing/structured_store.py`

**Purpose:** JSON-based storage for full paper data and extractions.

**Class: StructuredStore**

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| index_dir | Path | From config | Directory for JSON files |

**Files Managed:**
- `papers.json`: Paper metadata
- `extractions.json`: LLM extractions
- `metadata.json`: Index state
- `summary.json`: Statistics

---

### 05.9 Implement Paper Storage

**Method:** `save_papers(papers: list[PaperMetadata])`

**Logic:**
1. Serialize papers to JSON
2. Include schema version
3. Include generation timestamp
4. Write atomically (temp file then rename)

**Method:** `load_papers()`

**Returns:** `list[PaperMetadata]`

**Method:** `get_paper(paper_id: str)`

**Returns:** `PaperMetadata | None`

---

### 05.10 Implement Extraction Storage

**Method:** `save_extractions(extractions: list[Extraction])`

**Logic:**
1. Serialize extractions to JSON
2. Key by paper_id for easy lookup
3. Include schema version
4. Write atomically

**Method:** `load_extractions()`

**Returns:** `dict[str, Extraction]` (paper_id -> Extraction)

**Method:** `get_extraction(paper_id: str)`

**Returns:** `Extraction | None`

---

### 05.11 Implement Metadata Storage

**Method:** `save_metadata(metadata: IndexMetadata)`

**IndexMetadata Model:**

| Field | Type | Description |
|-------|------|-------------|
| schema_version | str | Current schema version |
| created_at | datetime | Index creation time |
| last_updated | datetime | Last modification |
| total_papers | int | Paper count |
| total_extractions | int | Extraction count |
| total_chunks | int | Vector chunk count |
| failed_items | list[FailedItem] | Processing failures |
| processing_stats | ProcessingStats | Token usage, costs |

**Method:** `load_metadata()`

**Returns:** `IndexMetadata | None`

---

### 05.12 Implement Summary Generation

**Method:** `generate_summary(papers: list[PaperMetadata], extractions: dict[str, Extraction])`

**Returns:** `IndexSummary`

**IndexSummary Contents:**

| Field | Description |
|-------|-------------|
| total_papers | Total paper count |
| papers_by_year | Count by publication year |
| papers_by_type | Count by item type |
| papers_by_collection | Count by collection |
| top_authors | Most frequent authors |
| recent_additions | Recently indexed papers |
| extraction_coverage | Percent with extractions |
| common_themes | Most common discipline tags |

**Method:** `save_summary(summary: IndexSummary)`

**Writes:** `summary.json`

---

## Test Scenarios

### T05.1 Collection Creation

**Test:** Create new ChromaDB collection
**Input:** Fresh persist directory
**Expected:** Collection created successfully
**Verify:** Collection exists, empty

### T05.2 Chunk Insertion

**Test:** Insert embedding chunks
**Input:** List of 10 EmbeddingChunks
**Expected:** All added to collection
**Verify:** get_collection_stats shows 10 docs

### T05.3 Duplicate Prevention

**Test:** Handle duplicate chunk IDs
**Input:** Insert same chunk twice
**Expected:** Error or upsert behavior
**Verify:** Only one copy exists

### T05.4 Similarity Search

**Test:** Find similar documents
**Input:** Query embedding for "machine learning"
**Expected:** Relevant chunks returned
**Verify:** Results ordered by similarity

### T05.5 Metadata Filtering

**Test:** Filter search by year
**Input:** Search with where={"year": {"$gte": 2020}}
**Expected:** Only 2020+ papers returned
**Verify:** All results have year >= 2020

### T05.6 Chunk Type Filtering

**Test:** Filter search by chunk type
**Input:** Search with where={"chunk_type": "thesis"}
**Expected:** Only thesis chunks returned
**Verify:** All results have chunk_type "thesis"

### T05.7 Combined Filters

**Test:** Apply multiple filters
**Input:** Filter by year AND chunk_type
**Expected:** Results match both criteria
**Verify:** Correct intersection

### T05.8 Document Retrieval by Paper

**Test:** Get all chunks for a paper
**Input:** Known paper_id
**Expected:** All chunks for that paper
**Verify:** Correct count and types

### T05.9 Document Deletion

**Test:** Delete chunks by paper_id
**Input:** paper_id to delete
**Expected:** All chunks removed
**Verify:** get_by_paper_id returns empty

### T05.10 Persistence

**Test:** Data survives restart
**Input:** Insert chunks, close, reopen
**Expected:** Chunks still present
**Verify:** Search returns same results

### T05.11 JSON Paper Storage

**Test:** Save and load papers
**Input:** List of PaperMetadata
**Expected:** Round-trip preserves data
**Verify:** Loaded papers match saved

### T05.12 JSON Extraction Storage

**Test:** Save and load extractions
**Input:** List of Extractions
**Expected:** Round-trip preserves data
**Verify:** Loaded extractions match saved

### T05.13 Summary Generation

**Test:** Generate index summary
**Input:** Papers and extractions
**Expected:** Summary with statistics
**Verify:** Counts match input data

---

## Caveats and Edge Cases

### ChromaDB Version Compatibility

- ChromaDB API may change between versions
- Lock to specific version in requirements
- Test after any version upgrade

### Persist Directory Structure

- ChromaDB creates internal files
- Don't modify files manually
- Include in .gitignore except structure

### Memory Usage

- Large collections load into memory
- Monitor memory with 10K+ chunks
- Consider pagination for very large indexes

### Embedding Dimension Mismatch

- Collection has fixed dimension after creation
- Cannot mix 384 and 768 dim vectors
- Store dimension in metadata
- Check on initialization

### Metadata Value Types

- ChromaDB metadata values: str, int, float, bool
- No nested objects
- Serialize lists as comma-separated strings
- Or use JSON strings for complex data

### Where Clause Limitations

- Limited operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $contains
- No regex support
- No full-text search (that's what semantic search is for)

### Atomic Operations

- ChromaDB operations are atomic
- Batch operations succeed or fail together
- No partial failure handling needed

### Concurrent Access

- ChromaDB supports concurrent reads
- Single writer recommended
- Consider locking for updates

### JSON File Size

- papers.json and extractions.json grow with library
- For 1000+ papers, files may be 10+ MB
- Still readable, but consider chunking for very large libraries

### Unicode in JSON

- Ensure UTF-8 encoding
- Handle special characters
- Use ensure_ascii=False for readability

### Backup Considerations

- ChromaDB persist directory is source of truth
- JSON files are secondary/derived
- Include both in backup strategy

---

## Acceptance Criteria

- [ ] Creates ChromaDB collection successfully
- [ ] Inserts embedding chunks with metadata
- [ ] Performs similarity search
- [ ] Filters by metadata (year, type, collection)
- [ ] Retrieves chunks by paper_id
- [ ] Deletes chunks correctly
- [ ] Persists data across restarts
- [ ] Saves/loads papers.json
- [ ] Saves/loads extractions.json
- [ ] Generates summary statistics
- [ ] All unit tests pass

---

## Files Created

| File | Status |
|------|--------|
| src/indexing/vector_store.py | Pending |
| src/indexing/structured_store.py | Pending |
| tests/test_vector_store.py | Pending |

---

*End of Task 05*
