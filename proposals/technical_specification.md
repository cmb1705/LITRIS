# Literature Review Index System: Technical Specification

**Version:** 1.0
**Date:** 2024-12-05
**Status:** Draft Proposal

---

## 1. System Overview

### 1.1 Purpose

Build an intermediary knowledge layer between a Zotero library (~500+ academic documents) and an LLM assistant, enabling efficient literature review, citation support, research front mapping, and gap analysis without loading full documents into context.

### 1.2 High-Level Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Zotero Library │────▶│  Extraction      │────▶│  Knowledge      │
│  (SQLite + PDFs)│     │  Pipeline        │     │  Index          │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  Query Layer    │
                                                 │  (LLM Access)   │
                                                 └─────────────────┘
```

### 1.3 Core Capabilities

- Extract and structure academic paper content (thesis, methods, results, conclusions, claims, limitations, future directions)
- Maintain citation-ready metadata linked to source documents
- Enable semantic search via vector embeddings
- Support incremental updates when new papers are added to Zotero
- Provide file-based query interface accessible by LLM assistant

---

## 2. Data Sources

### 2.1 Zotero SQLite Database

**Location:** `D:\Zotero\zotero.sqlite`

**Key Tables to Query:**

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `items` | Core item records | `itemID`, `key`, `itemTypeID`, `dateAdded`, `dateModified` |
| `itemAttachments` | PDF file mappings | `itemID`, `parentItemID`, `path`, `contentType` |
| `itemData` | Metadata field values | `itemID`, `fieldID`, `valueID` |
| `itemDataValues` | Actual metadata values | `valueID`, `value` |
| `fields` | Field name definitions | `fieldID`, `fieldName` |
| `itemTypes` | Item type definitions | `itemTypeID`, `typeName` |
| `creators` | Author records | `creatorID`, `firstName`, `lastName` |
| `itemCreators` | Item-author mappings | `itemID`, `creatorID`, `creatorTypeID`, `orderIndex` |
| `collections` | Folder/collection definitions | `collectionID`, `collectionName`, `parentCollectionID` |
| `collectionItems` | Item-collection mappings | `collectionID`, `itemID` |

**Access Mode:** Read-only. Never modify the Zotero database.

### 2.2 PDF Storage

**Location:** `D:\Zotero\storage\{8-CHAR-KEY}\{filename}.pdf`

**Path Resolution:**
1. Query `itemAttachments` for items where `contentType = 'application/pdf'`
2. Get the `path` field (format: `storage:{filename}.pdf`)
3. Get the item's `key` from the `items` table
4. Construct full path: `D:\Zotero\storage\{key}\{filename}`

### 2.3 Existing Full-Text Cache

**Location:** `D:\Zotero\storage\{KEY}\.zotero-ft-cache`

**Usage:** Check for existence before PDF extraction. If present and valid, use cached text to reduce processing. Only ~28 of ~400 items currently have this cache, so PDF extraction is still required for most documents.

---

## 3. Project Structure

```
D:\Git_Repos\Lit_Review\
├── .gitignore
├── README.md
├── requirements.txt
├── config.yaml                    # Configuration file
├── proposals/
│   └── technical_specification.md
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configuration loader
│   ├── zotero/
│   │   ├── __init__.py
│   │   ├── database.py            # Zotero SQLite reader
│   │   └── models.py              # Zotero data models
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py       # PDF text extraction
│   │   ├── ocr_handler.py         # OCR fallback for scanned PDFs
│   │   └── text_cleaner.py        # Text normalization
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── llm_client.py          # Claude API interface
│   │   ├── section_extractor.py   # LLM-based section extraction
│   │   ├── prompts.py             # Extraction prompt templates
│   │   └── schemas.py             # Output validation schemas
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── embeddings.py          # Embedding generation
│   │   ├── vector_store.py        # ChromaDB interface
│   │   └── structured_store.py    # JSON/SQLite structured storage
│   ├── query/
│   │   ├── __init__.py
│   │   ├── search.py              # Search interface
│   │   └── retrieval.py           # Document retrieval
│   └── utils/
│       ├── __init__.py
│       ├── logging.py             # Logging configuration
│       └── file_utils.py          # File operations
├── scripts/
│   ├── build_index.py             # Full index build
│   ├── update_index.py            # Incremental update
│   ├── query_index.py             # CLI query tool
│   └── validate_extraction.py     # Quality validation
├── data/
│   ├── index/
│   │   ├── papers.json            # Structured paper data
│   │   ├── extractions.json       # LLM extraction results
│   │   ├── metadata.json          # Index metadata & state
│   │   └── chroma/                # ChromaDB vector store
│   ├── cache/
│   │   ├── pdf_text/              # Extracted PDF text cache
│   │   └── processing_state.json  # Pipeline state tracking
│   └── logs/
│       └── extraction.log
└── tests/
    ├── __init__.py
    ├── test_zotero_reader.py
    ├── test_pdf_extraction.py
    ├── test_llm_extraction.py
    └── fixtures/
        └── sample_papers/
```

---

## 4. Data Models

### 4.1 Paper Metadata Model

Represents bibliographic information extracted from Zotero.

**Fields:**

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `paper_id` | string | Generated | Unique identifier (UUID) |
| `zotero_key` | string | `items.key` | Zotero's 8-character key |
| `zotero_item_id` | integer | `items.itemID` | Zotero's internal ID |
| `item_type` | string | `itemTypes.typeName` | journalArticle, book, thesis, etc. |
| `title` | string | itemData | Full title |
| `authors` | list[Author] | creators/itemCreators | List of author objects |
| `publication_year` | integer | itemData.date | Extracted year |
| `publication_date` | string | itemData.date | Full date string |
| `journal` | string | itemData.publicationTitle | Journal/publication name |
| `volume` | string | itemData.volume | Volume number |
| `issue` | string | itemData.issue | Issue number |
| `pages` | string | itemData.pages | Page range |
| `doi` | string | itemData.DOI | Digital Object Identifier |
| `isbn` | string | itemData.ISBN | ISBN if applicable |
| `issn` | string | itemData.ISSN | ISSN if applicable |
| `abstract` | string | itemData.abstractNote | Original abstract |
| `url` | string | itemData.url | Source URL |
| `collections` | list[string] | collections | Zotero collection names |
| `tags` | list[string] | tags/itemTags | User-assigned tags |
| `pdf_path` | string | Constructed | Full path to PDF file |
| `pdf_attachment_key` | string | items.key | Key for attachment item |
| `date_added` | datetime | items.dateAdded | When added to Zotero |
| `date_modified` | datetime | items.dateModified | Last modification |
| `indexed_at` | datetime | Generated | When indexed by this system |

**Author Sub-Model:**

| Field | Type | Description |
|-------|------|-------------|
| `first_name` | string | Author first name |
| `last_name` | string | Author last name |
| `full_name` | string | Combined name |
| `order` | integer | Author order (1 = first author) |
| `role` | string | author, editor, contributor, etc. |

### 4.2 Extraction Model

Represents LLM-extracted structured content from a paper.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `paper_id` | string | Links to Paper Metadata |
| `extraction_version` | string | Version of extraction prompts used |
| `extracted_at` | datetime | Timestamp of extraction |
| `extraction_model` | string | Model used (e.g., claude-3-opus) |
| `token_usage` | object | Input/output token counts |
| `thesis_statement` | string | Core argument or hypothesis |
| `research_questions` | list[string] | Explicit research questions |
| `theoretical_framework` | string | Theoretical grounding/lens |
| `methodology` | MethodologySection | Methods details |
| `key_findings` | list[Finding] | Primary results |
| `conclusions` | string | Author conclusions |
| `limitations` | list[string] | Stated limitations |
| `future_directions` | list[string] | Suggested future research |
| `key_claims` | list[Claim] | Specific claims with evidence refs |
| `contribution_summary` | string | 2-3 sentence contribution summary |
| `discipline_tags` | list[string] | Inferred discipline/field tags |
| `extraction_confidence` | float | 0-1 confidence score |
| `extraction_notes` | string | Any extraction issues/warnings |

**Methodology Sub-Model:**

| Field | Type | Description |
|-------|------|-------------|
| `approach` | string | Qualitative, quantitative, mixed, theoretical, etc. |
| `design` | string | Case study, experiment, survey, meta-analysis, etc. |
| `data_sources` | list[string] | What data was used |
| `sample_description` | string | Sample/population description |
| `analysis_methods` | list[string] | Analytical techniques used |
| `tools` | list[string] | Software/instruments mentioned |

**Finding Sub-Model:**

| Field | Type | Description |
|-------|------|-------------|
| `finding` | string | The finding statement |
| `evidence_type` | string | Statistical, qualitative, theoretical |
| `significance` | string | Why this matters |

**Claim Sub-Model:**

| Field | Type | Description |
|-------|------|-------------|
| `claim` | string | The specific claim |
| `support_type` | string | How it's supported |
| `page_reference` | string | Approximate location in paper |

### 4.3 Embedding Record Model

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `paper_id` | string | Links to Paper Metadata |
| `chunk_id` | string | Unique chunk identifier |
| `chunk_type` | string | abstract, thesis, methodology, finding, claim, full_summary |
| `text` | string | The embedded text content |
| `embedding` | vector | Embedding vector (stored in ChromaDB) |
| `metadata` | object | Additional context for retrieval |

### 4.4 Index State Model

Tracks processing state for incremental updates.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `last_full_build` | datetime | Last complete index build |
| `last_update` | datetime | Last incremental update |
| `total_papers` | integer | Papers in index |
| `total_extractions` | integer | Completed extractions |
| `failed_extractions` | list[FailedItem] | Items that failed processing |
| `zotero_snapshot` | object | Hash/timestamp of Zotero state |
| `version` | string | Index schema version |

---

## 5. Pipeline Components

### 5.1 Zotero Database Reader

**Purpose:** Read-only access to Zotero SQLite database to extract metadata and PDF paths.

**Responsibilities:**
- Connect to Zotero SQLite database
- Query items with PDF attachments
- Join across tables to build complete Paper Metadata objects
- Resolve PDF file paths from storage keys
- Handle items in collections (map collection hierarchy)
- Track item modification dates for incremental updates

**Key Queries:**

1. **Get all items with PDFs:**
   - Join `items` → `itemAttachments` (via parentItemID)
   - Filter by `contentType = 'application/pdf'`
   - Filter by `itemTypeID` not in (attachment, note, annotation)

2. **Get item metadata:**
   - Join `itemData` → `itemDataValues` → `fields`
   - Filter by `itemID`

3. **Get authors:**
   - Join `itemCreators` → `creators` → `creatorTypes`
   - Order by `orderIndex`

4. **Get collections:**
   - Join `collectionItems` → `collections`
   - Recursively resolve parent collections for full path

**Error Handling:**
- Handle missing PDF files gracefully
- Log items with incomplete metadata
- Skip items without PDF attachments

### 5.2 PDF Text Extractor

**Purpose:** Extract raw text content from PDF files.

**Primary Tool:** PyMuPDF (fitz) - fast, reliable, good Unicode support

**Fallback Tool:** pytesseract + pdf2image for OCR when text extraction fails

**Extraction Flow:**

1. Check for existing `.zotero-ft-cache` file
   - If exists and non-empty, use cached text
   - Validate cache is complete (compare page counts)

2. Attempt PyMuPDF text extraction
   - Extract text page by page
   - Preserve page boundaries with markers
   - Calculate extraction quality metrics (char count, word count)

3. Quality check
   - If extracted text is too short relative to page count
   - If text contains mostly garbled characters
   - Flag for OCR processing

4. OCR fallback (if needed)
   - Convert PDF pages to images (pdf2image)
   - Run Tesseract OCR
   - Combine results

5. Text cleaning
   - Normalize Unicode
   - Fix common PDF extraction artifacts (broken words, ligatures)
   - Preserve paragraph structure
   - Remove headers/footers if detected

**Output:**
- Cleaned full text
- Page count
- Extraction method used (cache/pymupdf/ocr)
- Quality score (0-1)
- Any warnings or issues

**Caching:**
- Store extracted text in `data/cache/pdf_text/{paper_id}.txt`
- Include extraction metadata in sidecar JSON file

### 5.3 LLM Section Extractor

**Purpose:** Use Claude to extract structured sections and insights from paper text.

**Model:** Claude Opus 4.5 (model ID: `claude-opus-4-5-20251101`)

**Implementation Approach:** Claude Agent SDK with Message Batches API

The extraction pipeline uses the Claude Agent SDK with the Message Batches API for cost-efficient bulk processing. This approach provides:

- **50% cost reduction** compared to standard API pricing
- **Batch processing** of up to 100,000 requests per batch
- **Asynchronous execution** with 24-hour processing window
- **Automatic retry** handling for transient failures

**Input:** Full paper text (or truncated if exceeds context limits)

**Context Limit Handling:**
- Claude Opus context: 200K tokens
- Most academic papers: 5K-30K tokens
- For papers exceeding limit: prioritize abstract, introduction, methods, results, discussion, conclusion sections
- Use heuristic section detection to identify key sections before sending

**Prompt Strategy:**

Single comprehensive extraction prompt that requests all structured fields in one call. This is more cost-effective than multiple targeted prompts.

**Prompt Template Requirements:**

The prompt must:
1. Provide clear field definitions and expected formats
2. Request JSON output matching the Extraction Model schema
3. Include examples of good extractions
4. Handle missing sections gracefully (paper may not have all sections)
5. Request confidence indicators for uncertain extractions
6. Ask for page/location hints where possible

**Prompt Sections:**

1. **System context:** You are an academic research analyst extracting structured information from scholarly papers.

2. **Task description:** Extract the following structured information from the provided paper.

3. **Field specifications:** Detailed description of each field, what to look for, and format requirements.

4. **Output format:** JSON schema with all required and optional fields.

5. **Handling uncertainty:** Instructions for when information is unclear or missing.

6. **Quality guidelines:** What makes a good extraction vs. poor extraction.

**Response Validation:**
- Parse JSON response
- Validate against Extraction Model schema
- Flag missing required fields
- Log extraction quality metrics

**Batch API Workflow:**

1. **Batch Creation:**
   - Collect papers into batches (up to 10,000 papers per batch for manageability)
   - Each request includes `custom_id` matching `paper_id` for result correlation
   - Submit batch via `client.messages.batches.create(requests=requests)`

2. **Batch Monitoring:**
   - Poll batch status every 30 seconds
   - Track `processing_status` (in_progress, ended)
   - Monitor `request_counts.succeeded`, `request_counts.errored`

3. **Result Processing:**
   - Stream results via `client.messages.batches.results(batch_id)`
   - Match results to papers using `custom_id` (order not guaranteed)
   - Write results to `data/index/extractions.json`

**Token Usage Tracking:**
- Record input tokens, output tokens per extraction
- Calculate running cost (with 50% batch discount applied)
- Store in extraction record
- Track batch-level aggregates in `metadata.json`

**Error Handling:**
- Batch API handles transient failures automatically
- Failed individual requests logged in batch results
- Store partial extractions with error notes
- Retry failed papers in subsequent batch if needed

### 5.4 Embedding Generator

**Purpose:** Generate vector embeddings for semantic search.

**Embedding Model Options (in order of preference):**

1. **OpenAI text-embedding-3-small** - Good quality, reasonable cost, 1536 dimensions
2. **Cohere embed-english-v3.0** - High quality, good for academic text
3. **sentence-transformers/all-MiniLM-L6-v2** - Free, local, 384 dimensions, slightly lower quality

**Recommendation:** Start with sentence-transformers for cost control during development, evaluate OpenAI for production quality.

**What to Embed:**

Create multiple embedding chunks per paper for granular retrieval:

| Chunk Type | Content | Purpose |
|------------|---------|---------|
| `abstract` | Original abstract | General paper matching |
| `thesis` | Thesis statement | Core argument matching |
| `contribution` | Contribution summary | What paper adds |
| `methodology` | Methodology description | Methods matching |
| `findings` | Concatenated key findings | Results matching |
| `claims` | Individual claims (one per chunk) | Specific claim retrieval |
| `limitations` | Concatenated limitations | Gap analysis |
| `future_work` | Concatenated future directions | Opportunity identification |
| `full_summary` | All extracted content combined | Broad matching |

**Embedding Metadata:**

Each embedding stored with:
- `paper_id`
- `chunk_type`
- `paper_title` (for display)
- `authors` (for filtering)
- `year` (for filtering)
- `collections` (for filtering)

**Chunking Strategy:**
- Most chunks are single fields (under 512 tokens)
- For long fields, split at sentence boundaries
- Overlap not required for structured extractions
- Maximum chunk size: 512 tokens

### 5.5 Vector Store (ChromaDB)

**Purpose:** Store and query vector embeddings for semantic search.

**Why ChromaDB:**
- File-based (no server required)
- Python-native
- Supports metadata filtering
- Persistent storage
- Easy to inspect and debug

**Storage Location:** `data/index/chroma/`

**Collection Structure:**

Single collection named `paper_chunks` containing all embedding types, differentiated by metadata.

**Schema:**

```
Collection: paper_chunks
- id: {paper_id}_{chunk_type}_{chunk_index}
- embedding: vector
- document: original text
- metadata:
  - paper_id: string
  - chunk_type: string
  - title: string
  - authors: string (comma-separated)
  - year: integer
  - collections: string (comma-separated)
  - item_type: string
```

**Query Capabilities:**
- Semantic similarity search
- Filter by metadata (year range, collection, item type)
- Return top-k results with scores
- Combine with full-text search if needed

### 5.6 Structured Store

**Purpose:** Store full paper metadata and extractions in queryable JSON format.

**Storage Format:** JSON files (human-readable, git-friendly, LLM-accessible)

**Files:**

1. **papers.json** - Array of Paper Metadata objects
2. **extractions.json** - Array of Extraction objects (keyed by paper_id)
3. **metadata.json** - Index state and configuration

**File Format Considerations:**
- Pretty-printed JSON for readability
- One logical record per concept (not one file per paper - reduces file count)
- Include schema version for migration support

**Query Support:**
- LLM can read entire files if needed
- For large libraries, may need to split by collection or year
- Include summary statistics in metadata.json

**Alternative (Phase 2):** SQLite database for more complex queries, but JSON preferred for initial LLM accessibility.

---

## 6. Query Interface

### 6.1 File-Based Query Protocol

Since the LLM will query by reading files, design files for efficient access.

**Index Summary File:** `data/index/summary.json`

Contains:
- Total paper count
- Papers by collection
- Papers by year
- Papers by item type
- Top keywords/themes
- Recently added papers

**Query Workflow:**

1. LLM reads `summary.json` to understand corpus scope
2. LLM reads `papers.json` to scan metadata
3. LLM formulates semantic query
4. User runs query script with LLM's query
5. Script returns ranked results
6. LLM reads relevant extractions from `extractions.json`
7. If needed, LLM requests full PDF load

### 6.2 Query Script Interface

**Script:** `scripts/query_index.py`

**Input Parameters:**
- `--query`: Natural language query string
- `--top-k`: Number of results (default: 10)
- `--chunk-types`: Filter by chunk type (optional)
- `--collections`: Filter by collection (optional)
- `--year-min`: Minimum year (optional)
- `--year-max`: Maximum year (optional)
- `--item-types`: Filter by item type (optional)
- `--output`: Output format (json, markdown, brief)

**Output:**

Ranked list with:
- Rank and similarity score
- Paper ID and title
- Authors and year
- Matching chunk type and text snippet
- Collection membership

**Output File:** Results written to `data/query_results/latest.json` and `data/query_results/latest.md`

### 6.3 Retrieval Functions

**Get Paper Details:**
- Input: paper_id
- Output: Full Paper Metadata + Extraction

**Get Papers by Collection:**
- Input: collection name
- Output: List of Paper Metadata

**Get Similar Papers:**
- Input: paper_id
- Output: Papers with similar embeddings

**Search by Metadata:**
- Input: author, year range, keywords
- Output: Matching papers

---

## 7. Configuration

### 7.1 Configuration File: `config.yaml`

```yaml
# Zotero Configuration
zotero:
  database_path: "D:/Zotero/zotero.sqlite"
  storage_path: "D:/Zotero/storage"

# Processing Configuration
extraction:
  model: "claude-opus-4-5-20251101"
  use_batch_api: true  # Use Message Batches API for 50% cost savings
  batch_size: 500  # Papers per batch submission
  poll_interval_seconds: 30  # Batch status polling interval
  max_tokens_output: 4096  # Max tokens for response

# Embedding Configuration
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"  # or "openai/text-embedding-3-small"
  chunk_max_tokens: 512

# Storage Configuration
storage:
  index_path: "data/index"
  cache_path: "data/cache"
  log_path: "data/logs"

# API Keys (loaded from environment)
api:
  anthropic_key_env: "ANTHROPIC_API_KEY"
  openai_key_env: "OPENAI_API_KEY"  # if using OpenAI embeddings

# Processing Options
options:
  use_zotero_cache: true  # Use .zotero-ft-cache when available
  ocr_enabled: true  # Enable OCR fallback
  skip_existing: true  # Skip already-processed papers on update
  log_level: "INFO"
```

### 7.2 Environment Variables

Required:
- `ANTHROPIC_API_KEY`: Claude API key for extraction

Optional:
- `OPENAI_API_KEY`: If using OpenAI embeddings

---

## 8. Build Pipeline

### 8.1 Full Build Process

**Script:** `scripts/build_index.py`

**Steps:**

1. **Initialize**
   - Load configuration
   - Create output directories
   - Initialize logging

2. **Zotero Scan**
   - Connect to Zotero database
   - Query all items with PDF attachments
   - Build Paper Metadata objects
   - Resolve and verify PDF paths
   - Write `papers.json`

3. **Text Extraction**
   - For each paper:
     - Check for cached text
     - Extract PDF text (PyMuPDF or OCR)
     - Clean and normalize text
     - Cache extracted text
   - Log extraction statistics

4. **LLM Extraction**
   - For each paper:
     - Load extracted text
     - Call Claude API with extraction prompt
     - Parse and validate response
     - Store extraction result
     - Track token usage
   - Write `extractions.json`
   - Log cost summary

5. **Embedding Generation**
   - For each paper:
     - Create chunks from extraction
     - Generate embeddings
     - Store in ChromaDB
   - Log embedding statistics

6. **Finalization**
   - Generate summary statistics
   - Write `metadata.json`
   - Write `summary.json`
   - Log completion

**Progress Tracking:**
- Progress bar with ETA
- Checkpoint saving every N papers
- Resume capability from last checkpoint

### 8.2 Incremental Update Process

**Script:** `scripts/update_index.py`

**Steps:**

1. **Load State**
   - Read `metadata.json` for last update time
   - Load existing papers.json

2. **Detect Changes**
   - Query Zotero for items modified since last update
   - Query Zotero for new items
   - Identify deleted items (in index but not in Zotero)

3. **Process Changes**
   - Remove deleted items from index
   - Process new items through full pipeline
   - Re-process modified items

4. **Update Index**
   - Merge new/updated extractions
   - Update ChromaDB (add new, update existing, remove deleted)
   - Update state metadata

---

## 9. Error Handling

### 9.1 Error Categories

| Category | Examples | Handling |
|----------|----------|----------|
| Zotero Access | Database locked, file not found | Retry with delay, fail gracefully |
| PDF Extraction | Corrupted PDF, encrypted PDF, no text | Log warning, mark as failed, continue |
| OCR Failure | Image quality too low | Log warning, store partial result |
| LLM API | Rate limit, timeout, invalid response | Exponential backoff, retry 3x |
| Validation | Missing required fields, malformed JSON | Store partial extraction with warnings |
| Embedding | Model unavailable, dimension mismatch | Retry, fall back to alternative model |

### 9.2 Failed Items Tracking

Store failed items in `metadata.json`:

```
failed_extractions: [
  {
    paper_id: "...",
    zotero_key: "...",
    title: "...",
    stage: "pdf_extraction|llm_extraction|embedding",
    error: "Error message",
    attempts: 3,
    last_attempt: "2024-12-05T10:00:00Z"
  }
]
```

### 9.3 Recovery Commands

- `--retry-failed`: Retry all failed items
- `--retry-paper {paper_id}`: Retry specific paper
- `--skip-paper {paper_id}`: Permanently skip paper

---

## 10. Dependencies

### 10.1 Python Version

Python 3.10+ (for modern type hints and match statements)

### 10.2 Core Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| `anthropic` | Claude API client (includes Batch API) | latest |
| `claude-agent-sdk` | Claude Agent SDK for programmatic access | latest |
| `pymupdf` (fitz) | PDF text extraction | latest |
| `chromadb` | Vector store | latest |
| `sentence-transformers` | Local embeddings | latest |
| `pydantic` | Data validation | 2.x |
| `pyyaml` | Configuration | latest |
| `tqdm` | Progress bars | latest |
| `python-dotenv` | Environment variables | latest |

### 10.3 Optional Dependencies

| Package | Purpose | When Needed |
|---------|---------|-------------|
| `pytesseract` | OCR | Scanned PDFs |
| `pdf2image` | PDF to image | OCR preprocessing |
| `openai` | OpenAI embeddings | If using OpenAI embeddings |
| `pytest` | Testing | Development |

### 10.4 System Dependencies

- Tesseract OCR (if OCR enabled): Install via system package manager
- Poppler (for pdf2image): Required for OCR path

---

## 11. Testing Strategy

### 11.1 Unit Tests

- Zotero database queries (mock database)
- PDF text extraction (sample PDFs)
- Text cleaning functions
- JSON schema validation
- Embedding chunking logic

### 11.2 Integration Tests

- End-to-end single paper processing
- Query and retrieval workflow
- Incremental update logic

### 11.3 Validation Tests

- Compare LLM extractions to manual review
- Measure semantic search relevance
- Check for extraction consistency

### 11.4 Test Fixtures

Include in `tests/fixtures/sample_papers/`:
- 3-5 representative PDFs (different formats, lengths)
- Expected extraction outputs
- Sample Zotero database subset

---

## 12. Phased Build Plan

### Phase 1: Foundation (MVP)

**Goal:** Extract and index 10 papers, prove the concept works.

**Deliverables:**
1. Zotero database reader
2. PDF text extraction (PyMuPDF only)
3. Basic LLM extraction with single prompt
4. JSON storage for papers and extractions
5. Manual query by reading JSON files

**Success Criteria:**
- Can extract metadata from Zotero
- Can extract text from PDFs
- LLM produces structured extractions
- Data is stored and readable

### Phase 2: Semantic Search

**Goal:** Add vector search capability.

**Deliverables:**
1. Embedding generation pipeline
2. ChromaDB integration
3. Query script with semantic search
4. Summary statistics generation

**Success Criteria:**
- Can query by natural language
- Results are relevantly ranked
- Query script is usable by LLM

### Phase 3: Robustness

**Goal:** Handle edge cases and scale to full library.

**Deliverables:**
1. OCR fallback for scanned PDFs
2. Error handling and retry logic
3. Failed items tracking
4. Full library processing
5. Cost tracking and reporting

**Success Criteria:**
- 95%+ of papers successfully processed
- Clear reporting on failures
- Known cost per paper

### Phase 4: Incremental Updates

**Goal:** Support ongoing library maintenance.

**Deliverables:**
1. Change detection from Zotero
2. Incremental update script
3. State persistence between runs
4. Add/update/delete handling

**Success Criteria:**
- New papers indexed without full rebuild
- Modified papers re-processed
- Deleted papers removed from index

### Phase 5: Refinement

**Goal:** Improve quality and usability.

**Deliverables:**
1. Extraction prompt tuning based on results
2. Query interface improvements
3. Output format options (markdown, structured)
4. Documentation and usage guides

**Success Criteria:**
- Extraction quality validated on sample
- Query workflow smooth for LLM use
- Clear documentation

---

## 13. Cost Estimation

### 13.1 LLM Extraction Costs

**Assumptions:**
- Average paper: 8,000 tokens input, 2,000 tokens output
- Claude Opus 4.5 pricing: $15/M input, $75/M output
- **Message Batches API: 50% discount on all tokens**
- 500 papers

**Calculation (with Batch API 50% discount):**

- Input: 500 × 8,000 = 4M tokens × $15/M × 0.5 = $30
- Output: 500 × 2,000 = 1M tokens × $75/M × 0.5 = $37.50
- **Total: ~$67.50 for full library** (50% savings vs standard API)

**Standard API Comparison (without batch discount):**

- Input: 4M tokens × $15/M = $60
- Output: 1M tokens × $75/M = $75
- Total: ~$135 (2x more expensive)

**Mitigation:**

- Test on 10-paper sample first (~$1.35 with batch API)
- Consider Sonnet for bulk processing (~$8 total with batch discount)
- Cache extractions to avoid re-processing
- Batch API provides automatic retry handling

### 13.2 Embedding Costs

**Using sentence-transformers (local):** $0

**Using OpenAI text-embedding-3-small:**
- ~2,000 tokens per paper across all chunks
- 500 papers = 1M tokens
- $0.02/M tokens = **$0.02 total**

---

## 14. Security Considerations

### 14.1 API Key Management

- Store API keys in environment variables only
- Never commit keys to git
- Use `.env` file (in `.gitignore`)

### 14.2 Zotero Database

- Read-only access enforced at connection level
- No writes to Zotero database ever
- Handle file locking gracefully

### 14.3 Data Privacy

- Index data stored locally only
- No data sent externally except to LLM APIs
- PDFs not uploaded anywhere

---

## 15. Future Considerations (Out of Scope)

The following are explicitly out of scope for initial build but documented for future phases:

1. **MCP Server:** Expose index via Model Context Protocol for direct LLM tool access
2. **Web UI:** Browser-based query interface
3. **Multi-library support:** Handle multiple Zotero libraries
4. **Citation network analysis:** Cross-reference citations between papers
5. **Automatic Zotero sync:** Watch for real-time changes
6. **Collaborative features:** Multi-user access
7. **Export formats:** BibTeX, RIS generation from index

---

## Appendix A: Key File Schemas

### papers.json Schema

```json
{
  "schema_version": "1.0",
  "generated_at": "ISO8601 timestamp",
  "paper_count": 500,
  "papers": [
    {
      "paper_id": "uuid",
      "zotero_key": "8CHARKEY",
      "zotero_item_id": 123,
      "item_type": "journalArticle",
      "title": "Paper Title",
      "authors": [
        {
          "first_name": "John",
          "last_name": "Smith",
          "full_name": "John Smith",
          "order": 1,
          "role": "author"
        }
      ],
      "publication_year": 2023,
      "publication_date": "2023-05-15",
      "journal": "Journal Name",
      "volume": "42",
      "issue": "3",
      "pages": "100-120",
      "doi": "10.1000/example",
      "abstract": "Original abstract text...",
      "url": "https://...",
      "collections": ["Network Analysis", "Scientometrics"],
      "tags": ["tag1", "tag2"],
      "pdf_path": "D:/Zotero/storage/8CHARKEY/filename.pdf",
      "pdf_attachment_key": "ATTCHKEY",
      "date_added": "ISO8601",
      "date_modified": "ISO8601",
      "indexed_at": "ISO8601"
    }
  ]
}
```

### extractions.json Schema

```json
{
  "schema_version": "1.0",
  "generated_at": "ISO8601 timestamp",
  "extraction_count": 500,
  "extractions": [
    {
      "paper_id": "uuid (matches papers.json)",
      "extraction_version": "1.0",
      "extracted_at": "ISO8601",
      "extraction_model": "claude-3-opus-20240229",
      "token_usage": {
        "input_tokens": 8000,
        "output_tokens": 2000
      },
      "thesis_statement": "The core argument...",
      "research_questions": ["RQ1...", "RQ2..."],
      "theoretical_framework": "This paper uses...",
      "methodology": {
        "approach": "quantitative",
        "design": "longitudinal study",
        "data_sources": ["Survey data", "Administrative records"],
        "sample_description": "500 participants from...",
        "analysis_methods": ["regression", "factor analysis"],
        "tools": ["SPSS", "R"]
      },
      "key_findings": [
        {
          "finding": "Finding statement",
          "evidence_type": "statistical",
          "significance": "Why it matters"
        }
      ],
      "conclusions": "The authors conclude...",
      "limitations": ["Limitation 1", "Limitation 2"],
      "future_directions": ["Future work 1", "Future work 2"],
      "key_claims": [
        {
          "claim": "Specific claim",
          "support_type": "empirical",
          "page_reference": "p. 15"
        }
      ],
      "contribution_summary": "2-3 sentence summary...",
      "discipline_tags": ["scientometrics", "network analysis"],
      "extraction_confidence": 0.85,
      "extraction_notes": "Abstract was missing, inferred from introduction"
    }
  ]
}
```

### metadata.json Schema

```json
{
  "schema_version": "1.0",
  "index_name": "Lit_Review_Index",
  "created_at": "ISO8601",
  "last_full_build": "ISO8601",
  "last_update": "ISO8601",
  "statistics": {
    "total_papers": 500,
    "total_extractions": 495,
    "total_embeddings": 4500,
    "papers_by_type": {
      "journalArticle": 400,
      "book": 50,
      "thesis": 20
    },
    "papers_by_year": {
      "2023": 50,
      "2022": 75
    },
    "papers_by_collection": {
      "Network Analysis": 120,
      "Scientometrics": 85
    }
  },
  "processing": {
    "total_input_tokens": 4000000,
    "total_output_tokens": 1000000,
    "estimated_cost_usd": 135.00,
    "extraction_model": "claude-3-opus-20240229",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
  },
  "failed_extractions": [
    {
      "paper_id": "uuid",
      "zotero_key": "8CHARKEY",
      "title": "Failed Paper Title",
      "stage": "pdf_extraction",
      "error": "PDF is encrypted",
      "attempts": 3,
      "last_attempt": "ISO8601"
    }
  ],
  "config_hash": "sha256 of config used"
}
```

---

*End of Technical Specification*
