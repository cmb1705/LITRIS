# MCP Integration Technical Specification

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Created | 2025-12-06 |
| Status | Implemented |
| Author | LITRIS Development Team |

---

## 1. Executive Summary

This specification defines the architecture and implementation requirements for integrating LITRIS with Claude Code via the Model Context Protocol (MCP). The integration enables direct tool access for semantic search, paper retrieval, and index analysis, facilitating a collaborative human-AI research workflow.

---

## 2. Objectives

### 2.1 Primary Goals

1. Expose LITRIS search functionality as MCP tools callable by Claude Code
2. Enable seamless research collaboration without shell command overhead
3. Support multi-query aggregation for complex research questions
4. Provide structured extraction data for synthesis and comparison

### 2.2 Success Criteria

| Criterion | Metric |
|-----------|--------|
| Tool Response Time | Less than 2 seconds for search queries |
| Result Quality | Extraction data included with all results |
| Reliability | 99% uptime during active sessions |
| Integration | Zero-friction tool invocation from Claude Code |

---

## 3. System Architecture

### 3.1 Component Overview

The MCP server acts as a bridge between Claude Code and the LITRIS index:

```
Claude Code
    |
    | (MCP Protocol - stdio transport)
    v
LITRIS MCP Server
    |
    +---> SearchEngine (src/query/search.py)
    |         |
    |         +---> VectorStore (ChromaDB)
    |         +---> StructuredStore (JSON index)
    |         +---> EmbeddingGenerator
    |
    +---> Configuration (config.yaml)
```

### 3.2 MCP Server Components

| Component | Responsibility |
|-----------|---------------|
| Server Runtime | Handle MCP protocol, manage connections |
| Tool Registry | Define and register available tools |
| Search Adapter | Translate tool calls to SearchEngine methods |
| Result Formatter | Structure responses for Claude consumption |
| Error Handler | Graceful degradation and informative errors |

### 3.3 Communication Protocol

- **Transport**: stdio (standard input/output)
- **Protocol**: MCP JSON-RPC 2.0
- **Message Format**: JSON with tool definitions and results

---

## 4. Tool Specifications

### 4.1 litris_search

**Purpose**: Perform semantic search across the literature index

**Input Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | string | Yes | Natural language search query |
| top_k | integer | No | Number of results (default: 10, max: 50) |
| chunk_types | array[string] | No | Filter by extraction section |
| year_min | integer | No | Minimum publication year |
| year_max | integer | No | Maximum publication year |
| collections | array[string] | No | Filter by Zotero collections |
| item_types | array[string] | No | Filter by item type |
| include_extraction | boolean | No | Include full extraction data (default: true) |
| recency_boost | float | No | Boost recent papers (0.0-1.0, default: 0.0) |

**Valid chunk_types**:
- abstract
- thesis
- contribution
- methodology
- findings
- claims
- limitations
- future_work
- full_summary

**Output Structure**:

```
{
  "query": string,
  "result_count": integer,
  "results": [
    {
      "rank": integer,
      "score": float,
      "paper_id": string,
      "title": string,
      "authors": string,
      "year": integer | null,
      "collections": array[string],
      "item_type": string,
      "chunk_type": string,
      "matched_text": string,
      "extraction": {
        "thesis_statement": string,
        "research_questions": array[string],
        "methodology": object,
        "key_findings": array[object],
        "conclusions": string,
        "limitations": array[string],
        "future_directions": array[string],
        "contribution_summary": string,
        "discipline_tags": array[string],
        "extraction_confidence": float
      },
      "pdf_path": string | null
    }
  ]
}
```

**Error Responses**:

| Error | Condition |
|-------|-----------|
| INDEX_NOT_FOUND | Index directory does not exist |
| INVALID_QUERY | Empty or malformed query |
| SEARCH_FAILED | Vector store or embedding error |

### 4.2 litris_get_paper

**Purpose**: Retrieve complete extraction and metadata for a specific paper

**Input Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| paper_id | string | Yes | LITRIS paper identifier |

**Output Structure**:

```
{
  "paper_id": string,
  "found": boolean,
  "paper": {
    "title": string,
    "authors": array[object],
    "author_string": string,
    "publication_year": integer | null,
    "publication_date": string | null,
    "journal": string | null,
    "doi": string | null,
    "abstract": string | null,
    "collections": array[string],
    "tags": array[string],
    "item_type": string,
    "pdf_path": string | null,
    "zotero_key": string
  },
  "extraction": {
    "thesis_statement": string,
    "research_questions": array[string],
    "methodology": {
      "approach": string,
      "design": string,
      "data_sources": array[string],
      "analysis_methods": array[string],
      "sample_size": string | null
    },
    "key_findings": array[{
      "finding": string,
      "evidence_type": string,
      "significance": string
    }],
    "conclusions": string,
    "limitations": array[string],
    "future_directions": array[string],
    "key_claims": array[{
      "claim": string,
      "support_type": string,
      "page_reference": string | null
    }],
    "contribution_summary": string,
    "discipline_tags": array[string],
    "extraction_confidence": float
  }
}
```

**Error Responses**:

| Error | Condition |
|-------|-----------|
| PAPER_NOT_FOUND | No paper with given ID exists |
| INDEX_NOT_FOUND | Index directory does not exist |

### 4.3 litris_similar

**Purpose**: Find papers similar to a given paper

**Input Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| paper_id | string | Yes | Source paper identifier |
| top_k | integer | No | Number of similar papers (default: 10) |

**Output Structure**:

```
{
  "source_paper_id": string,
  "source_title": string,
  "result_count": integer,
  "similar_papers": [
    {
      "rank": integer,
      "score": float,
      "paper_id": string,
      "title": string,
      "authors": string,
      "year": integer | null,
      "matched_on": string,
      "extraction": object
    }
  ]
}
```

### 4.4 litris_summary

**Purpose**: Get index statistics and coverage information

**Input Parameters**: None

**Output Structure**:

```
{
  "generated_at": string,
  "total_papers": integer,
  "total_extractions": integer,
  "papers_by_type": object,
  "papers_by_year": object,
  "papers_by_collection": object,
  "top_disciplines": object,
  "vector_store": {
    "total_chunks": integer,
    "unique_papers": integer,
    "chunk_types": object
  },
  "recent_papers": array[object]
}
```

### 4.5 litris_collections

**Purpose**: List all Zotero collections in the index

**Input Parameters**: None

**Output Structure**:

```
{
  "collections": array[string],
  "collection_counts": object
}
```

---

## 5. Server Implementation Requirements

### 5.1 Runtime Environment

| Requirement | Specification |
|-------------|---------------|
| Python Version | 3.10+ |
| MCP SDK | mcp (latest stable) |
| Dependencies | Inherit from LITRIS requirements.txt |
| Configuration | Use existing config.yaml |

### 5.2 Server Lifecycle

1. **Initialization**
   - Load configuration from config.yaml
   - Initialize SearchEngine with index paths
   - Verify index exists and is readable
   - Register tools with MCP runtime

2. **Request Handling**
   - Parse incoming tool calls
   - Validate parameters
   - Execute corresponding SearchEngine method
   - Format and return results

3. **Shutdown**
   - Clean up resources
   - Close database connections

### 5.3 Error Handling Strategy

| Error Category | Handling |
|----------------|----------|
| Configuration | Fail fast with descriptive message |
| Index Missing | Return specific error, suggest build |
| Search Failure | Log error, return partial results if possible |
| Timeout | Return timeout error after configurable limit |

### 5.4 Logging Requirements

- Log all tool invocations with parameters
- Log query execution times
- Log errors with full stack traces
- Support configurable log levels
- Write to data/logs/mcp_server.log

---

## 6. Claude Code Integration

### 6.1 Configuration

The MCP server must be registered in Claude Code settings:

Location: .claude/settings.json or user settings

```
{
  "mcpServers": {
    "litris": {
      "command": "python",
      "args": ["path/to/mcp_server.py"],
      "cwd": "path/to/litris"
    }
  }
}
```

### 6.2 Tool Naming Convention

All tools are prefixed with `mcp__litris__` when invoked:

| Internal Name | Claude Code Tool Name |
|---------------|----------------------|
| litris_search | mcp__litris__litris_search |
| litris_get_paper | mcp__litris__litris_get_paper |
| litris_similar | mcp__litris__litris_similar |
| litris_summary | mcp__litris__litris_summary |
| litris_collections | mcp__litris__litris_collections |

### 6.3 Usage Patterns

**Single Query**:
```
User: Find papers about citation network analysis
AI: [invokes mcp__litris__litris_search with query]
AI: [synthesizes results with inline citations]
```

**Multi-Query Aggregation**:
```
User: Compare GNN and LSTM approaches for citation forecasting
AI: [invokes litris_search for "GNN citation forecasting"]
AI: [invokes litris_search for "LSTM bibliometrics prediction"]
AI: [aggregates, deduplicates, compares findings]
```

**Deep Dive**:
```
AI: [identifies paper needing full text review]
AI: [invokes litris_get_paper for paper_id]
AI: [uses Read tool on pdf_path if needed]
```

---

## 7. Performance Requirements

### 7.1 Response Time Targets

| Operation | Target | Maximum |
|-----------|--------|---------|
| litris_search (k=10) | 500ms | 2000ms |
| litris_get_paper | 100ms | 500ms |
| litris_similar | 1000ms | 3000ms |
| litris_summary | 200ms | 1000ms |
| litris_collections | 100ms | 500ms |

### 7.2 Resource Limits

| Resource | Limit |
|----------|-------|
| Memory | 512MB maximum |
| Concurrent Requests | 1 (stdio is sequential) |
| Result Set Size | 50 papers maximum per query |

---

## 8. Security Considerations

### 8.1 Access Control

- Read-only access to index data
- No write operations to index or Zotero
- No network access beyond local resources
- PDF paths returned but not read by server

### 8.2 Input Validation

- Sanitize all string inputs
- Validate integer ranges
- Reject malformed JSON
- Limit query length to 1000 characters

### 8.3 Data Protection

- No sensitive data in logs
- No API keys in responses
- PDF paths are local filesystem only

---

## 9. Testing Requirements

### 9.1 Unit Tests

| Component | Coverage Target |
|-----------|-----------------|
| Tool parameter validation | 100% |
| Search adapter methods | 90% |
| Result formatting | 90% |
| Error handling | 100% |

### 9.2 Integration Tests

| Scenario | Description |
|----------|-------------|
| Basic search | Query returns expected results |
| Filtered search | Year/collection filters work |
| Paper retrieval | Get paper returns full extraction |
| Similar papers | Similarity search excludes source |
| Empty results | Graceful handling of no matches |
| Invalid paper ID | Proper error response |

### 9.3 End-to-End Tests

| Scenario | Description |
|----------|-------------|
| Claude Code invocation | Tools callable from Claude Code |
| Multi-tool workflow | Aggregate results across queries |
| Error propagation | Errors surface correctly in Claude |

---

## 10. Deployment

### 10.1 Installation Steps

1. Install MCP SDK dependency
2. Create mcp_server.py in src/mcp/
3. Configure Claude Code settings
4. Verify with test invocation

### 10.2 Configuration Files

| File | Purpose |
|------|---------|
| config.yaml | Index paths, model settings |
| .claude/settings.json | MCP server registration |

### 10.3 Verification

- Run server standalone to verify startup
- Test each tool with sample inputs
- Verify Claude Code can discover tools
- Execute sample research workflow

---

## 11. Future Enhancements

### 11.1 Planned Extensions

| Enhancement | Priority | Description |
|-------------|----------|-------------|
| litris_compare | High | Side-by-side comparison of paper sets (methods, datasets, findings) |
| litris_multi_search | High | Batch multiple queries with merge/score logic |
| litris_gaps | Medium | Identify coverage gaps in corpus |
| litris_external_ingest | Medium | Add external papers to sidecar store for comparison |
| litris_export | Low | Export results to bibliography format |

### 11.2 Metadata Enrichment

Enhance extraction schema with additional tags for improved filtering:

| Tag | Values | Purpose |
|-----|--------|---------|
| task | citation_forecasting, clustering, link_prediction, etc. | Filter by research task |
| method | GNN, LSTM, transformer, statistical, etc. | Filter by methodology type |
| domain | scientometrics, bibliometrics, network_science, etc. | Filter by domain |
| dataset | WoS, Scopus, MAG, custom, etc. | Filter by data source |

These can be added to the LLM extraction prompt or post-processed from existing extractions.

### 11.3 Query Decomposition

For compound research questions, implement a query decomposition step:

1. Parse compound question into sub-queries
2. Run litris_search for each sub-query
3. Merge results with score normalization
4. Apply recency boost if specified
5. Deduplicate and rank combined results

### 11.4 External Paper Integration

Support for papers outside the Zotero library:

- Sidecar store for external references
- Normalize to same metadata schema
- Enable side-by-side comparison with indexed papers
- Curated list of canonical papers per domain

### 11.5 Research Skill Integration

A Claude Code skill could wrap common workflows:
- Research question decomposition
- Multi-query execution and aggregation
- Synthesis template with citations
- Gap analysis with web search

---

## 12. Appendices

### A. Extraction Schema Reference

See completed/technical_specification.md for full extraction schema.

### B. Chunk Type Descriptions

| Chunk Type | Content |
|------------|---------|
| abstract | Paper abstract or executive summary |
| thesis | Central thesis statement |
| contribution | Main contribution to field |
| methodology | Research methods and design |
| findings | Key empirical or theoretical findings |
| claims | Specific claims with evidence |
| limitations | Acknowledged limitations |
| future_work | Suggested future research |
| full_summary | Comprehensive paper summary |

### C. Item Type Reference

| Item Type | Description |
|-----------|-------------|
| journalArticle | Peer-reviewed journal article |
| book | Complete book |
| bookSection | Chapter in edited volume |
| conferencePaper | Conference proceedings paper |
| thesis | Doctoral or master's thesis |
| report | Technical or research report |
| preprint | Pre-publication manuscript |
