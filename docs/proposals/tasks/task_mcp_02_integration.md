# Task MCP-02: Integration

## Overview

| Field | Value |
|-------|-------|
| Phase | 2 |
| Status | Complete |
| Dependencies | MCP-01 (Tools) |
| Reference | mcp_technical_specification.md Section 5.2-5.4 |

---

## Objective

Connect the MCP server to existing LITRIS components, ensuring reliable data access and acceptable performance.

---

## Tasks

### 2.1 Create SearchEngine Adapter

**Description**: Create adapter layer between MCP and SearchEngine.

**Implementation Details**:

- Wrap SearchEngine initialization
- Handle path resolution for index and chroma directories
- Provide methods matching tool requirements
- Abstract away SearchEngine internals

**Acceptance Criteria**:

- Adapter initializes successfully with valid paths
- All SearchEngine methods accessible through adapter
- Clean error handling for initialization failures

**Caveats**:

- SearchEngine requires embedding model loading
- First query may be slow due to model initialization

---

### 2.2 Handle Index Path Resolution

**Description**: Reliably resolve index paths across environments.

**Implementation Details**:

- Support absolute paths from config
- Support relative paths from project root
- Validate paths exist before use
- Provide clear errors for missing paths

**Path Requirements**:

| Path | Purpose | Config Key |
|------|---------|------------|
| Index Directory | JSON index files | storage.index_path |
| Chroma Directory | Vector store | storage.chroma_path |
| Cache Directory | Extraction cache | storage.cache_path |

**Acceptance Criteria**:

- Absolute paths work as-is
- Relative paths resolved from project root
- Missing paths raise clear errors

**Caveats**:

- Windows and Unix path handling differences
- Paths may be relative to different bases

---

### 2.3 Implement Lazy Initialization

**Description**: Defer expensive initialization until first use.

**Implementation Details**:

- Do not initialize SearchEngine on server start
- Initialize on first tool call
- Cache initialized instance
- Handle initialization errors gracefully

**Acceptance Criteria**:

- Server starts quickly (under 1 second)
- First tool call initializes components
- Subsequent calls use cached instance
- Initialization errors surface to caller

**Caveats**:

- First query will be slower
- Thread safety considerations

---

### 2.4 Add Connection Management

**Description**: Manage ChromaDB connections for reliability.

**Implementation Details**:

- Ensure ChromaDB client initialized once
- Handle connection failures gracefully
- Implement reconnection logic if needed
- Clean up connections on shutdown

**Acceptance Criteria**:

- Single ChromaDB client instance
- Connection failures logged and reported
- Clean shutdown closes connections

**Caveats**:

- ChromaDB uses SQLite under the hood
- File locking may cause issues on Windows

---

### 2.5 Implement Timeout Handling

**Description**: Add timeout handling for long-running operations.

**Implementation Details**:

- Configure timeout from config or default
- Apply timeout to search operations
- Return timeout error if exceeded
- Log timeout events

**Timeout Defaults**:

| Operation | Default | Maximum |
|-----------|---------|---------|
| Search | 5 seconds | 30 seconds |
| Get Paper | 2 seconds | 10 seconds |
| Similar | 10 seconds | 30 seconds |

**Acceptance Criteria**:

- Operations respect timeout limit
- Timeout returns structured error
- Partial results not returned on timeout

**Caveats**:

- Python threading for timeout may be complex
- Consider async/await patterns

---

### 2.6 Add Request/Response Logging

**Description**: Log all tool invocations for debugging.

**Implementation Details**:

- Log tool name and parameters on request
- Log result summary on response
- Log execution time
- Include request ID for correlation

**Log Format**:

| Field | Description |
|-------|-------------|
| timestamp | ISO 8601 timestamp |
| request_id | Unique request identifier |
| tool | Tool name |
| params | Request parameters (sanitized) |
| duration_ms | Execution time in milliseconds |
| result_count | Number of results (for searches) |
| error | Error code if failed |

**Acceptance Criteria**:

- All requests logged
- All responses logged
- Execution time captured
- Errors logged with context

**Caveats**:

- Do not log full extraction content (too verbose)
- Sanitize any sensitive data

---

### 2.7 Test with Existing Index

**Description**: Verify tools work with real LITRIS index data.

**Implementation Details**:

- Use existing index from Phase 1 build
- Test each tool with real queries
- Verify result accuracy
- Measure performance

**Test Queries**:

| Query | Expected Results |
|-------|-----------------|
| "citation network analysis" | Papers on bibliometrics |
| "research methodology" | Papers discussing methods |
| "machine learning" | ML-related papers if present |

**Acceptance Criteria**:

- All tools return valid results
- Results match CLI query output
- Performance within specification

**Caveats**:

- Index content varies by Zotero library
- May need sample queries adjusted

---

## Test Scenarios

### T2.1 Lazy Initialization

**Scenario**: Verify lazy loading behavior

**Steps**:
1. Start MCP server
2. Note startup time
3. Invoke first tool
4. Note response time

**Expected Result**: Startup under 1 second, first query slower

---

### T2.2 Connection Recovery

**Scenario**: Recover from ChromaDB issues

**Steps**:
1. Start server and run query
2. Corrupt ChromaDB lock file temporarily
3. Attempt another query
4. Restore lock file
5. Attempt another query

**Expected Result**: Clear error on failure, recovery on retry

---

### T2.3 Timeout Behavior

**Scenario**: Verify timeout enforcement

**Steps**:
1. Configure very short timeout (100ms)
2. Run complex search query
3. Observe response

**Expected Result**: Timeout error returned if query exceeds limit

---

### T2.4 Request Logging

**Scenario**: Verify logging completeness

**Steps**:
1. Run several tool invocations
2. Check data/logs/mcp_server.log
3. Verify log entries

**Expected Result**: All requests and responses logged with timing

---

### T2.5 Real Index Query

**Scenario**: Query actual LITRIS index

**Steps**:
1. Ensure index built with real papers
2. Invoke litris_search with domain-relevant query
3. Verify results

**Expected Result**: Relevant papers returned with extractions

---

## Verification Checklist

- [x] SearchEngine adapter created
- [x] Path resolution working
- [x] Lazy initialization implemented
- [x] Connection management working
- [x] Timeout handling implemented
- [x] Request/response logging complete
- [x] Tested with real index data
