# Task MCP-04: Testing

## Overview

| Field | Value |
|-------|-------|
| Phase | 4 |
| Status | Complete |
| Dependencies | MCP-02 (Integration) |
| Reference | mcp_technical_specification.md Section 9 |

---

## Objective

Implement comprehensive test coverage for MCP functionality including unit tests, integration tests, and end-to-end validation.

---

## Tasks

### 4.1 Unit Tests for Parameter Validation

**Description**: Test parameter validation logic.

**Test Cases**:

| Test | Input | Expected |
|------|-------|----------|
| Valid query | "search term" | Pass |
| Empty query | "" | Reject with INVALID_QUERY |
| Query too long | 2000 chars | Reject with VALIDATION_ERROR |
| Valid top_k | 10 | Pass |
| top_k too large | 100 | Reject or cap at 50 |
| top_k negative | -1 | Reject |
| Valid year_min | 2020 | Pass |
| Invalid year | "abc" | Reject |
| Valid chunk_types | ["thesis"] | Pass |
| Invalid chunk_type | ["invalid"] | Reject |

**Acceptance Criteria**:

- All parameter types validated
- Clear error messages for failures
- Edge cases handled

**Caveats**:

- Validation should be fast
- Consider caching validation results

---

### 4.2 Unit Tests for Result Formatting

**Description**: Test response formatting logic.

**Test Cases**:

| Test | Input | Expected |
|------|-------|----------|
| Format search result | EnrichedResult | Correct JSON structure |
| Handle None values | Result with nulls | Default values used |
| Truncate long text | 1000 char matched_text | Truncated to spec limit |
| Format extraction | Full extraction | All fields present |
| Handle missing extraction | Paper without extraction | Empty extraction object |
| Format dates | datetime objects | ISO 8601 strings |

**Acceptance Criteria**:

- All result types formatted correctly
- JSON serializable output
- No TypeError or KeyError

**Caveats**:

- Unicode handling in text
- Datetime timezone awareness

---

### 4.3 Unit Tests for Error Handling

**Description**: Test error response generation.

**Test Cases**:

| Test | Condition | Expected |
|------|-----------|----------|
| Index not found | Missing index dir | INDEX_NOT_FOUND error |
| Search failure | ChromaDB error | SEARCH_FAILED error |
| Paper not found | Invalid paper_id | PAPER_NOT_FOUND error |
| Validation failure | Bad params | VALIDATION_ERROR |
| Timeout | Slow query | TIMEOUT error |

**Acceptance Criteria**:

- All error codes tested
- Error messages helpful
- No raw exceptions escaped

**Caveats**:

- Simulating failures may require mocking
- Error codes must match spec

---

### 4.4 Integration Tests for Each Tool

**Description**: Test each tool with real components.

**Test Matrix**:

| Tool | Test | Setup |
|------|------|-------|
| litris_search | Basic query | Real index |
| litris_search | With filters | Real index |
| litris_search | Empty results | Obscure query |
| litris_get_paper | Valid paper | Known paper_id |
| litris_get_paper | Invalid paper | Random string |
| litris_similar | Valid source | Known paper_id |
| litris_similar | Invalid source | Random string |
| litris_summary | No params | Real index |
| litris_collections | No params | Real index |

**Acceptance Criteria**:

- All tools return expected structure
- Results match SearchEngine output
- Performance acceptable

**Caveats**:

- Requires test index data
- May need test fixtures

---

### 4.5 Integration Tests for Filtered Searches

**Description**: Test filter combinations.

**Test Cases**:

| Filter Combination | Expected |
|--------------------|----------|
| year_min only | Papers >= year |
| year_max only | Papers <= year |
| year_min + year_max | Papers in range |
| chunk_types single | Only that chunk type |
| chunk_types multiple | Any of those types |
| collections single | Papers in collection |
| collections + year | Combined filter |
| All filters | Most restrictive result |

**Acceptance Criteria**:

- Each filter works independently
- Filters combine correctly (AND logic)
- Empty results handled gracefully

**Caveats**:

- Test data must have variety
- Some filter combinations may yield no results

---

### 4.6 Integration Tests for Error Cases

**Description**: Test error conditions end-to-end.

**Test Cases**:

| Scenario | Setup | Expected |
|----------|-------|----------|
| Missing index | Delete index files | INDEX_NOT_FOUND |
| Corrupt ChromaDB | Corrupt db file | SEARCH_FAILED |
| Invalid config | Bad config.yaml | Configuration error |
| Empty index | Index with no papers | Empty results |

**Acceptance Criteria**:

- Errors detected and reported
- No server crash
- Clear error messages

**Caveats**:

- May need isolated test environments
- Cleanup after destructive tests

---

### 4.7 End-to-End Test with Claude Code

**Description**: Verify full workflow in Claude Code.

**Test Workflow**:

1. Start Claude Code with LITRIS project
2. Verify MCP server starts
3. Invoke each tool via prompts
4. Execute multi-tool workflow
5. Verify synthesis capability

**Test Prompts**:

| Step | Prompt | Expected |
|------|--------|----------|
| 1 | "Search LITRIS for methodology" | Search results returned |
| 2 | "Get paper [first result ID]" | Full paper data |
| 3 | "Find similar papers" | Similar papers list |
| 4 | "Show index summary" | Statistics displayed |
| 5 | "Synthesize findings" | Claude uses retrieved context |

**Acceptance Criteria**:

- All tools callable from Claude
- Results integrated into conversation
- Multi-step workflows succeed

**Caveats**:

- Claude Code behavior may vary
- May need specific prompting

---

### 4.8 Performance Benchmark Tests

**Description**: Measure and validate performance.

**Benchmarks**:

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Server startup | < 1s | Time to ready |
| First search | < 3s | Includes init |
| Subsequent search | < 500ms | Cached components |
| Get paper | < 200ms | Single lookup |
| Similar papers | < 1s | Vector search |
| Summary | < 500ms | Aggregation |

**Acceptance Criteria**:

- Meet specification targets
- Consistent performance
- No degradation over time

**Caveats**:

- Hardware affects results
- First query includes model loading

---

### 4.9 Document Test Procedures

**Description**: Create testing documentation.

**Documentation Sections**:

| Section | Content |
|---------|---------|
| Running unit tests | pytest commands |
| Running integration tests | Setup and execution |
| Test fixtures | How to create/maintain |
| Adding new tests | Guidelines |
| CI/CD integration | Future automation |

**Acceptance Criteria**:

- Clear instructions
- Reproducible procedures
- Coverage reporting

**Caveats**:

- Keep documentation updated
- Include troubleshooting

---

## Test Fixtures Required

### Fixture: Mock Index

**Purpose**: Provide test data without real Zotero library

**Contents**:

| Item | Count |
|------|-------|
| Papers | 10-20 |
| Extractions | 10-20 |
| Collections | 3-5 |
| Years | 2015-2024 |

### Fixture: Mock SearchEngine

**Purpose**: Unit testing without real components

**Mocked Methods**:

- search() - Returns fixed results
- get_paper() - Returns fixture paper
- search_similar_papers() - Returns fixture results

---

## Verification Checklist

- [x] Validation unit tests complete
- [x] Formatting unit tests complete
- [x] Error handling unit tests complete
- [x] Tool integration tests complete
- [x] Filter integration tests complete
- [x] Error case tests complete
- [x] End-to-end tests complete
- [x] Performance benchmarks run
- [x] Test documentation complete
