# Task MCP-00: Foundation

## Overview

| Field | Value |
|-------|-------|
| Phase | 0 |
| Status | Complete |
| Dependencies | None |
| Reference | mcp_technical_specification.md Section 5 |

---

## Objective

Set up the MCP development environment, create the module structure, and establish the foundation for tool implementation.

---

## Tasks

### 0.1 Add MCP SDK to Dependencies

**Description**: Add the MCP Python SDK to project dependencies.

**Implementation Details**:

- Add mcp package to requirements.txt
- Verify compatibility with existing dependencies
- Test import in virtual environment

**Acceptance Criteria**:

- mcp package listed in requirements.txt
- pip install succeeds without conflicts
- import mcp works in Python REPL

**Caveats**:

- MCP SDK may have transitive dependencies that conflict
- Version pinning may be required for stability

---

### 0.2 Create Module Directory

**Description**: Create the src/mcp/ module directory structure.

**Implementation Details**:

- Create src/mcp/ directory
- Create __init__.py with module exports
- Add module docstring describing purpose

**Acceptance Criteria**:

- Directory exists at src/mcp/
- __init__.py present with exports
- Module importable as src.mcp

**Caveats**:

- Ensure consistent naming with existing modules

---

### 0.3 Create Server Entry Point

**Description**: Create the main MCP server script.

**Implementation Details**:

- Create src/mcp/server.py
- Implement MCP server initialization
- Add stdio transport setup
- Implement graceful shutdown handling

**Acceptance Criteria**:

- Server starts with python -m src.mcp.server
- Connects via stdio transport
- Shuts down cleanly on SIGTERM/SIGINT

**Caveats**:

- Windows may require different signal handling
- stdio transport requires careful stream management

---

### 0.4 Configuration Loading

**Description**: Implement configuration loading for MCP server.

**Implementation Details**:

- Reuse existing Config class from src/config.py
- Add MCP-specific configuration section if needed
- Validate required paths exist at startup

**Acceptance Criteria**:

- Server loads config.yaml on startup
- Missing config raises clear error
- Invalid paths raise clear error

**Caveats**:

- Config path may differ when run from different directories
- Consider environment variable for config path override

---

### 0.5 Logging Configuration

**Description**: Set up MCP-specific logging.

**Implementation Details**:

- Create dedicated log file data/logs/mcp_server.log
- Configure log level from config or environment
- Include timestamps and request IDs
- Log server lifecycle events

**Acceptance Criteria**:

- Log file created on server start
- All levels (DEBUG, INFO, ERROR) work
- Logs include timestamp and context

**Caveats**:

- Log rotation may be needed for long-running servers
- Avoid logging sensitive data

---

### 0.6 Create Test Directory

**Description**: Create test directory structure for MCP tests.

**Implementation Details**:

- Create tests/test_mcp/ directory
- Create __init__.py
- Create conftest.py with shared fixtures
- Add fixture for mock index data

**Acceptance Criteria**:

- Directory exists at tests/test_mcp/
- pytest discovers tests in directory
- Fixtures available for all test files

**Caveats**:

- Test fixtures should not depend on real index
- Consider using temporary directories for test data

---

### 0.7 Update CLAUDE.md

**Description**: Document MCP module in project memory.

**Implementation Details**:

- Add MCP section to CLAUDE.md
- Document module purpose and structure
- Add server startup instructions
- Document tool naming conventions

**Acceptance Criteria**:

- MCP section present in CLAUDE.md
- Instructions are accurate and tested
- Naming conventions documented

**Caveats**:

- Keep documentation concise
- Update as implementation evolves

---

## Test Scenarios

### T0.1 Server Startup

**Scenario**: Start MCP server with valid configuration

**Steps**:
1. Ensure config.yaml exists with valid paths
2. Run python -m src.mcp.server
3. Observe server output

**Expected Result**: Server starts without errors, logs initialization

---

### T0.2 Missing Configuration

**Scenario**: Start server without config file

**Steps**:
1. Rename or remove config.yaml
2. Run python -m src.mcp.server
3. Observe error output

**Expected Result**: Clear error message indicating missing config

---

### T0.3 Invalid Index Path

**Scenario**: Start server with non-existent index path

**Steps**:
1. Set index path to non-existent directory in config
2. Run python -m src.mcp.server
3. Observe error output

**Expected Result**: Clear error message indicating invalid index path

---

### T0.4 Graceful Shutdown

**Scenario**: Stop running server

**Steps**:
1. Start server
2. Send SIGTERM or Ctrl+C
3. Observe shutdown behavior

**Expected Result**: Server logs shutdown and exits cleanly

---

## Verification Checklist

- [x] MCP SDK installed successfully
- [x] Module directory created
- [x] Server entry point works
- [x] Configuration loading works
- [x] Logging configured
- [x] Test directory created
- [x] CLAUDE.md updated
