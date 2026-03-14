# MCP Integration Project Plan

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Created | 2025-12-06 |
| Status | Complete |
| Reference | mcp_technical_specification.md |

---

## 1. Project Overview

### 1.1 Objective

Implement an MCP server that exposes LITRIS search functionality as tools directly callable by Claude Code, enabling seamless human-AI research collaboration.

### 1.2 Scope

| In Scope | Out of Scope |
|----------|--------------|
| MCP server implementation | Index modification via MCP |
| Five core tools | PDF text extraction via MCP |
| Claude Code integration | Web search integration |
| Unit and integration tests | Research skill automation |
| Documentation | GUI interface |

---

## 2. Phase Breakdown

### Phase 0: Foundation

**Objective**: Set up MCP development environment and project structure

**Tasks**:

- [x] 0.1 Add MCP SDK to project dependencies
- [x] 0.2 Create src/mcp/ module directory
- [x] 0.3 Create MCP server entry point
- [x] 0.4 Implement configuration loading for MCP
- [x] 0.5 Add MCP-specific logging configuration
- [x] 0.6 Create tests/test_mcp/ directory
- [x] 0.7 Document MCP module in CLAUDE.md

**Acceptance Criteria**:

- MCP SDK installed and importable
- Server starts without errors
- Configuration loaded from config.yaml
- Logging writes to data/logs/mcp_server.log

---

### Phase 1: Tool Implementation

**Objective**: Implement all five MCP tools with full functionality

**Tasks**:

- [x] 1.1 Create tool registry module
- [x] 1.2 Implement litris_search tool
- [x] 1.3 Implement litris_get_paper tool
- [x] 1.4 Implement litris_similar tool
- [x] 1.5 Implement litris_summary tool
- [x] 1.6 Implement litris_collections tool
- [x] 1.7 Create search adapter layer
- [x] 1.8 Implement result formatting for each tool
- [x] 1.9 Add parameter validation for all tools
- [x] 1.10 Implement error responses

**Acceptance Criteria**:

- All five tools registered and callable
- Parameter validation returns clear errors
- Results match specification format
- Errors are structured and informative

---

### Phase 2: Integration

**Objective**: Connect MCP server to existing LITRIS components

**Tasks**:

- [x] 2.1 Create SearchEngine adapter for MCP
- [x] 2.2 Handle index path resolution
- [x] 2.3 Implement lazy initialization of components
- [x] 2.4 Add connection pooling for ChromaDB
- [x] 2.5 Implement timeout handling
- [x] 2.6 Add request/response logging
- [x] 2.7 Test with existing index data

**Acceptance Criteria**:

- Tools return data from actual index
- Performance meets specification targets
- Timeouts handled gracefully
- All operations logged

---

### Phase 3: Claude Code Configuration

**Objective**: Enable Claude Code to discover and use LITRIS tools

**Tasks**:

- [x] 3.1 Create Claude Code settings template
- [x] 3.2 Document server registration process
- [x] 3.3 Add server to project .claude/settings.json
- [x] 3.4 Test tool discovery in Claude Code
- [x] 3.5 Verify tool invocation works
- [x] 3.6 Test multi-tool workflows
- [x] 3.7 Document troubleshooting steps

**Acceptance Criteria**:

- Claude Code discovers all five tools
- Tools invocable with mcp__litris__ prefix
- Results render correctly in Claude Code
- Errors surface appropriately

---

### Phase 4: Testing

**Objective**: Comprehensive test coverage for MCP functionality

**Tasks**:

- [x] 4.1 Unit tests for parameter validation
- [x] 4.2 Unit tests for result formatting
- [x] 4.3 Unit tests for error handling
- [x] 4.4 Integration tests for each tool
- [x] 4.5 Integration tests for filtered searches
- [x] 4.6 Integration tests for error cases
- [x] 4.7 End-to-end test with Claude Code
- [x] 4.8 Performance benchmark tests
- [x] 4.9 Document test procedures

**Acceptance Criteria**:

- Unit test coverage above 90%
- All integration tests pass
- End-to-end workflow verified
- Performance within specification

---

### Phase 5: Documentation and Refinement

**Objective**: Complete documentation and optimize implementation

**Tasks**:

- [x] 5.1 Update README with MCP usage
- [x] 5.2 Create MCP-specific documentation
- [x] 5.3 Add usage examples to docs
- [x] 5.4 Optimize query performance
- [x] 5.5 Review and improve error messages
- [x] 5.6 Update CLAUDE.md with MCP context
- [x] 5.7 Create troubleshooting guide

**Acceptance Criteria**:

- Documentation complete and accurate
- All examples tested and working
- Performance optimized
- Troubleshooting covers common issues

---

## 3. File Structure

The following files and directories will be created or modified:

LITRIS/
|
+-- src/
|   +-- mcp/
|       +-- __init__.py
|       +-- server.py
|       +-- tools.py
|       +-- adapters.py
|       +-- formatters.py
|       +-- validators.py
|
+-- tests/
|   +-- test_mcp/
|       +-- __init__.py
|       +-- test_tools.py
|       +-- test_adapters.py
|       +-- test_formatters.py
|       +-- test_validators.py
|       +-- test_integration.py
|
+-- .claude/
|   +-- settings.json
|
+-- docs/
|   +-- mcp_usage.md
|   +-- mcp_troubleshooting.md
|   +-- proposals/
|       +-- mcp_technical_specification.md
|       +-- mcp_project_plan.md
|       +-- tasks/
|           +-- task_mcp_00_foundation.md
|           +-- task_mcp_01_tools.md
|           +-- task_mcp_02_integration.md
|           +-- task_mcp_03_claude_config.md
|           +-- task_mcp_04_testing.md
|           +-- task_mcp_05_documentation.md
|
+-- requirements.txt (updated)
+-- config.yaml (unchanged)
+-- README.md (updated)
+-- CLAUDE.md (updated)
+-- STATE.md (new for MCP phase)

---

## 4. Detailed File Descriptions

### 4.1 Source Files

| File | Purpose |
|------|---------|
| src/mcp/__init__.py | Module exports and version |
| src/mcp/server.py | MCP server runtime and lifecycle |
| src/mcp/tools.py | Tool definitions and handlers |
| src/mcp/adapters.py | Bridge between MCP and SearchEngine |
| src/mcp/formatters.py | Response formatting utilities |
| src/mcp/validators.py | Input parameter validation |

### 4.2 Test Files

| File | Purpose |
|------|---------|
| tests/test_mcp/test_tools.py | Tool handler unit tests |
| tests/test_mcp/test_adapters.py | Adapter layer unit tests |
| tests/test_mcp/test_formatters.py | Formatter unit tests |
| tests/test_mcp/test_validators.py | Validator unit tests |
| tests/test_mcp/test_integration.py | Full workflow integration tests |

### 4.3 Configuration Files

| File | Purpose |
|------|---------|
| .claude/settings.json | Claude Code MCP server registration |

### 4.4 Documentation Files

| File | Purpose |
|------|---------|
| docs/mcp_usage.md | How to use LITRIS tools in Claude Code |
| docs/mcp_troubleshooting.md | Common issues and solutions |

---

## 5. Dependencies

### 5.1 New Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| mcp | latest | MCP SDK for Python |

### 5.2 Existing Dependencies (unchanged)

| Package | Purpose |
|---------|---------|
| chromadb | Vector store access |
| sentence-transformers | Embedding generation |
| pydantic | Data validation |

---

## 6. Task Dependencies

| Task | Depends On |
|------|------------|
| 1.x (Tools) | 0.x (Foundation) |
| 2.x (Integration) | 1.x (Tools) |
| 3.x (Claude Config) | 2.x (Integration) |
| 4.x (Testing) | 2.x (Integration) |
| 5.x (Documentation) | 3.x, 4.x |

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| MCP SDK breaking changes | Low | High | Pin version, monitor releases |
| Performance bottlenecks | Medium | Medium | Benchmark early, optimize iteratively |
| ChromaDB connection issues | Low | High | Connection pooling, graceful retry |
| Claude Code compatibility | Low | High | Test with multiple Claude Code versions |

---

## 8. Milestones

| Milestone | Phase | Deliverable |
|-----------|-------|-------------|
| M1: Server Starts | 0 | Runnable MCP server |
| M2: Tools Work | 1 | All five tools functional |
| M3: Index Connected | 2 | Real data from LITRIS index |
| M4: Claude Integration | 3 | Tools callable from Claude Code |
| M5: Tested | 4 | All tests passing |
| M6: Documented | 5 | Complete documentation |

---

## 9. Estimated Effort

| Phase | Tasks | Estimated Effort |
|-------|-------|------------------|
| Phase 0: Foundation | 7 | Small |
| Phase 1: Tools | 10 | Medium |
| Phase 2: Integration | 7 | Medium |
| Phase 3: Claude Config | 7 | Small |
| Phase 4: Testing | 9 | Medium |
| Phase 5: Documentation | 7 | Small |
| **Total** | **47** | - |

---

## 10. Completion Notes

All phases completed. Key deliverables:

- MCP server implemented with FastMCP
- 5 tools: litris_search, litris_get_paper, litris_similar, litris_summary, litris_collections
- 51 tests passing (validators, adapters, integration)
- Documentation in README.md, CLAUDE.md, and docs/mcp_troubleshooting.md
- Claude Code integration verified and working
