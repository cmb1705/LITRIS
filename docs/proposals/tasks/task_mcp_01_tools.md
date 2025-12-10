# Task MCP-01: Tool Implementation

## Overview

| Field | Value |
|-------|-------|
| Phase | 1 |
| Status | Complete |
| Dependencies | MCP-00 (Foundation) |
| Reference | mcp_technical_specification.md Section 4 |

---

## Objective

Implement all five MCP tools with complete parameter handling, validation, and response formatting.

---

## Tasks

### 1.1 Create Tool Registry Module

**Description**: Create module for registering tools with MCP server.

**Implementation Details**:

- Create src/mcp/tools.py
- Define tool registration mechanism
- Create decorator or function for tool definition
- Export tool list for server registration

**Acceptance Criteria**:

- Tool registry module exists
- Tools can be registered with metadata
- Server can enumerate registered tools

**Caveats**:

- Follow MCP SDK patterns for tool registration
- Ensure tool names match specification

---

### 1.2 Implement litris_search

**Description**: Implement semantic search tool.

**Implementation Details**:

- Accept query string and optional filters
- Validate all input parameters
- Call SearchEngine.search() method
- Format results according to specification
- Handle empty results gracefully

**Parameters**:

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| query | string | Yes | - |
| top_k | integer | No | 10 |
| chunk_types | array | No | all |
| year_min | integer | No | none |
| year_max | integer | No | none |
| collections | array | No | all |
| item_types | array | No | all |
| include_extraction | boolean | No | true |

**Acceptance Criteria**:

- Returns results matching query semantically
- All filters work correctly
- Extraction data included when requested
- Empty queries rejected with error

**Caveats**:

- Large result sets may impact performance
- Chunk type validation must match spec

---

### 1.3 Implement litris_get_paper

**Description**: Implement single paper retrieval tool.

**Implementation Details**:

- Accept paper_id parameter
- Look up paper in structured store
- Return full paper metadata and extraction
- Include PDF path for deep reading
- Handle missing papers with clear error

**Parameters**:

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| paper_id | string | Yes | - |

**Acceptance Criteria**:

- Returns complete paper data for valid ID
- Returns PAPER_NOT_FOUND for invalid ID
- PDF path included when available

**Caveats**:

- Paper IDs may use old or new format
- Handle both zotero_key and zotero_key_attachment format

---

### 1.4 Implement litris_similar

**Description**: Implement similar paper discovery tool.

**Implementation Details**:

- Accept source paper_id and optional top_k
- Find paper's summary chunk in vector store
- Search for similar papers using that embedding
- Exclude source paper from results
- Return ranked similar papers

**Parameters**:

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| paper_id | string | Yes | - |
| top_k | integer | No | 10 |

**Acceptance Criteria**:

- Returns papers similar to source
- Source paper excluded from results
- Similarity scores included

**Caveats**:

- Papers without embeddings cannot be used as source
- Similarity is based on full_summary chunk

---

### 1.5 Implement litris_summary

**Description**: Implement index statistics tool.

**Implementation Details**:

- No input parameters required
- Call StructuredStore.generate_summary()
- Add vector store statistics
- Format for easy reading

**Parameters**: None

**Acceptance Criteria**:

- Returns paper counts by type, year, collection
- Includes vector store chunk counts
- Includes recent papers list

**Caveats**:

- Summary generation may be slow for large indices
- Consider caching summary results

---

### 1.6 Implement litris_collections

**Description**: Implement collection listing tool.

**Implementation Details**:

- No input parameters required
- Get unique collections from index
- Include paper counts per collection
- Sort by count or alphabetically

**Parameters**: None

**Acceptance Criteria**:

- Returns all collection names
- Includes paper counts
- Sorted consistently

**Caveats**:

- Collection names come from Zotero
- Some papers may have no collections

---

### 1.7 Create Search Adapter Layer

**Description**: Create adapter between MCP tools and SearchEngine.

**Implementation Details**:

- Create src/mcp/adapters.py
- Wrap SearchEngine methods
- Handle parameter translation
- Manage SearchEngine lifecycle

**Acceptance Criteria**:

- Adapter initializes SearchEngine
- All tool methods work through adapter
- Clean separation of concerns

**Caveats**:

- SearchEngine initialization is expensive
- Consider lazy initialization

---

### 1.8 Implement Result Formatting

**Description**: Create formatters for tool responses.

**Implementation Details**:

- Create src/mcp/formatters.py
- Convert EnrichedResult to dict
- Format extraction data consistently
- Handle None values gracefully

**Acceptance Criteria**:

- All response formats match specification
- No None values in required fields
- Consistent date/number formatting

**Caveats**:

- JSON serialization of complex types
- Unicode handling in text fields

---

### 1.9 Implement Parameter Validation

**Description**: Create validators for tool inputs.

**Implementation Details**:

- Create src/mcp/validators.py
- Validate required parameters present
- Validate types and ranges
- Return clear error messages

**Acceptance Criteria**:

- Missing required params rejected
- Invalid types rejected
- Ranges enforced (top_k, years)
- Error messages are helpful

**Caveats**:

- Chunk type names must match exactly
- Year validation should allow reasonable range

---

### 1.10 Implement Error Responses

**Description**: Define and implement error response structure.

**Implementation Details**:

- Define error codes from specification
- Create error response format
- Ensure all error paths return structured errors
- Log errors with context

**Error Codes**:

| Code | Meaning |
|------|---------|
| INDEX_NOT_FOUND | Index directory missing |
| INVALID_QUERY | Empty or malformed query |
| SEARCH_FAILED | Vector store error |
| PAPER_NOT_FOUND | No paper with ID |
| VALIDATION_ERROR | Parameter validation failed |

**Acceptance Criteria**:

- All errors have code and message
- Errors logged with context
- No unstructured exceptions escape

**Caveats**:

- MCP SDK may have its own error handling
- Maintain consistency with SDK patterns

---

## Test Scenarios

### T1.1 Basic Search

**Scenario**: Search with simple query

**Steps**:
1. Invoke litris_search with query "network analysis"
2. Observe results

**Expected Result**: Returns relevant papers with extractions

---

### T1.2 Filtered Search

**Scenario**: Search with year filter

**Steps**:
1. Invoke litris_search with query and year_min=2020
2. Observe results

**Expected Result**: All results have year >= 2020

---

### T1.3 Empty Query

**Scenario**: Search with empty query

**Steps**:
1. Invoke litris_search with query=""
2. Observe error

**Expected Result**: INVALID_QUERY error returned

---

### T1.4 Get Valid Paper

**Scenario**: Retrieve existing paper

**Steps**:
1. Invoke litris_get_paper with known paper_id
2. Observe result

**Expected Result**: Full paper data and extraction returned

---

### T1.5 Get Invalid Paper

**Scenario**: Retrieve non-existent paper

**Steps**:
1. Invoke litris_get_paper with paper_id="INVALID"
2. Observe error

**Expected Result**: PAPER_NOT_FOUND error returned

---

### T1.6 Similar Papers

**Scenario**: Find similar papers

**Steps**:
1. Invoke litris_similar with valid paper_id
2. Observe results

**Expected Result**: Returns similar papers, excludes source

---

### T1.7 Index Summary

**Scenario**: Get index statistics

**Steps**:
1. Invoke litris_summary
2. Observe result

**Expected Result**: Statistics with paper counts and vector store info

---

## Verification Checklist

- [x] Tool registry created
- [x] litris_search implemented
- [x] litris_get_paper implemented
- [x] litris_similar implemented
- [x] litris_summary implemented
- [x] litris_collections implemented
- [x] Search adapter working
- [x] Result formatting correct
- [x] Parameter validation complete
- [x] Error responses structured
