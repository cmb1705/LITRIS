# Task MCP-03: Claude Code Configuration

## Overview

| Field | Value |
|-------|-------|
| Phase | 3 |
| Status | Complete |
| Dependencies | MCP-02 (Integration) |
| Reference | mcp_technical_specification.md Section 6 |

---

## Objective

Configure Claude Code to discover and use LITRIS MCP tools, enabling seamless research collaboration.

---

## Tasks

### 3.1 Create Settings Template

**Description**: Create template for Claude Code MCP configuration.

**Implementation Details**:

- Create .claude/settings.json if not exists
- Add mcpServers section with litris entry
- Include command, args, and cwd fields
- Document configuration options

**Configuration Structure**:

| Field | Value |
|-------|-------|
| command | python |
| args | path to server.py |
| cwd | project root directory |
| env | optional environment variables |

**Acceptance Criteria**:

- Settings file created with valid JSON
- Configuration points to correct server
- Paths work on target platform

**Caveats**:

- Paths must be absolute or correctly relative
- Windows vs Unix path separators
- Python executable path may vary

---

### 3.2 Document Registration Process

**Description**: Create documentation for MCP server setup.

**Implementation Details**:

- Write step-by-step setup instructions
- Cover Windows and Unix platforms
- Include verification steps
- Add troubleshooting section

**Documentation Sections**:

| Section | Content |
|---------|---------|
| Prerequisites | Required software and setup |
| Configuration | How to edit settings |
| Verification | How to test setup |
| Troubleshooting | Common issues |

**Acceptance Criteria**:

- Instructions are complete and accurate
- Both platforms covered
- Verification steps work

**Caveats**:

- Claude Code versions may differ
- Settings location may vary by OS

---

### 3.3 Add Project Settings

**Description**: Add MCP configuration to project settings.

**Implementation Details**:

- Create or update .claude/settings.json
- Configure litris MCP server entry
- Use relative paths where possible
- Test configuration loads correctly

**Acceptance Criteria**:

- Settings file present in .claude/
- Claude Code recognizes configuration
- Server launchable from settings

**Caveats**:

- User settings may override project settings
- Virtual environment activation

---

### 3.4 Test Tool Discovery

**Description**: Verify Claude Code discovers LITRIS tools.

**Implementation Details**:

- Start Claude Code in project directory
- Check that MCP server starts
- Verify tools listed in available tools
- Confirm tool descriptions correct

**Expected Tools**:

| Tool Name | Description |
|-----------|-------------|
| mcp__litris__litris_search | Semantic search |
| mcp__litris__litris_get_paper | Paper retrieval |
| mcp__litris__litris_similar | Similar papers |
| mcp__litris__litris_summary | Index statistics |
| mcp__litris__litris_collections | Collection listing |

**Acceptance Criteria**:

- All five tools discovered
- Names match expected pattern
- Descriptions accurate

**Caveats**:

- MCP server must start successfully
- Tool discovery may require restart

---

### 3.5 Verify Tool Invocation

**Description**: Test invoking tools from Claude Code.

**Implementation Details**:

- Request Claude to use each tool
- Verify parameters passed correctly
- Verify results returned correctly
- Check error handling

**Test Prompts**:

| Prompt | Expected Tool |
|--------|---------------|
| "Search LITRIS for papers about X" | litris_search |
| "Get details for paper ID X" | litris_get_paper |
| "Find papers similar to X" | litris_similar |
| "Show LITRIS index summary" | litris_summary |
| "List collections in LITRIS" | litris_collections |

**Acceptance Criteria**:

- Tools invocable via natural prompts
- Results displayed correctly
- Errors surface appropriately

**Caveats**:

- Claude may need prompting to use MCP tools
- Result formatting affects readability

---

### 3.6 Test Multi-Tool Workflows

**Description**: Test complex workflows using multiple tools.

**Implementation Details**:

- Create research question requiring multiple searches
- Verify Claude aggregates results
- Test deep-dive workflow (search then get_paper)
- Test gap analysis workflow

**Workflow Examples**:

| Workflow | Tools Used |
|----------|------------|
| Research synthesis | litris_search (multiple) |
| Paper deep-dive | litris_search, litris_get_paper |
| Similarity exploration | litris_search, litris_similar |
| Coverage analysis | litris_summary, litris_search |

**Acceptance Criteria**:

- Multi-tool workflows execute correctly
- Results can be synthesized
- PDF paths enable deep reading

**Caveats**:

- Context window limits
- Tool result aggregation

---

### 3.7 Document Troubleshooting

**Description**: Create troubleshooting guide for common issues.

**Implementation Details**:

- Identify common failure modes
- Document error messages and solutions
- Include diagnostic commands
- Add FAQ section

**Common Issues**:

| Issue | Cause | Solution |
|-------|-------|----------|
| Server not starting | Path issues | Verify paths in settings |
| Tools not discovered | Server crash | Check server logs |
| Timeout errors | Slow initialization | Increase timeout |
| No results | Index missing | Build index first |

**Acceptance Criteria**:

- Common issues documented
- Solutions tested and verified
- Diagnostic steps clear

**Caveats**:

- Issues may be environment-specific
- Update as new issues found

---

## Test Scenarios

### T3.1 Fresh Setup

**Scenario**: Configure MCP from scratch

**Steps**:
1. Remove existing .claude/settings.json
2. Follow documentation to create settings
3. Start Claude Code
4. Verify tools available

**Expected Result**: Tools discovered and usable

---

### T3.2 Tool Invocation

**Scenario**: Use each tool via prompt

**Steps**:
1. Ask Claude to search LITRIS
2. Ask Claude to get a specific paper
3. Ask Claude to find similar papers
4. Ask Claude for index summary
5. Ask Claude to list collections

**Expected Result**: Each tool invoked correctly, results returned

---

### T3.3 Research Workflow

**Scenario**: Complete research question workflow

**Steps**:
1. Ask complex research question
2. Observe Claude searching index
3. Ask for deep dive on specific paper
4. Request synthesis

**Expected Result**: Coherent research assistance using LITRIS

---

### T3.4 Error Recovery

**Scenario**: Handle tool errors gracefully

**Steps**:
1. Ask for non-existent paper
2. Observe error message
3. Provide correct paper ID
4. Verify successful retrieval

**Expected Result**: Clear error message, successful retry

---

## Verification Checklist

- [x] Settings template created
- [x] Registration documented
- [x] Project settings configured
- [x] Tool discovery verified
- [x] Tool invocation working
- [x] Multi-tool workflows tested
- [x] Troubleshooting documented
