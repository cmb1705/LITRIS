# Phase 5: Enhanced Features - Project Plan

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Created | 2026-01-22 |
| Status | Active |
| Epic | LITRIS-1h7 |

---

## 1. Project Overview

### 1.1 Objective

Extend LITRIS with advanced research capabilities: citation network visualization, literature gap detection, research question generation, and multi-model LLM consensus.

### 1.2 Scope

| In Scope | Out of Scope |
|----------|--------------|
| Citation network visualization | External citation API integration (OpenAlex) |
| Gap detection algorithm | Full bibliometric analysis suite |
| Research question generation | Automated literature review writing |
| LLM Council consensus | Fine-tuning or training models |
| MCP query auto-save | Real-time collaboration features |

---

## 2. Feature Breakdown

### F1: Citation Network Visualization

**Objective**: Enable users to explore citation relationships between papers in their library.

**Tasks**:

- [ ] LITRIS-vvx: Define citation sources and graph schema
- [ ] LITRIS-4gv: Build graph generation pipeline
- [ ] LITRIS-78s: Streamlit visualization view

**Dependencies**: Existing similarity network (PyVis) provides UI patterns.

**Effort**: Medium (3 tasks, ~2-3 days)

---

### F2: Gap Detection

**Objective**: Identify underexplored areas and research gaps in indexed literature.

**Tasks**:

- [ ] LITRIS-c66: Define algorithm and required inputs
- [ ] LITRIS-xrr: Implement scoring pipeline
- [ ] LITRIS-00o: Output report and evaluation

**Dependencies**: Requires embeddings and extraction data.

**Effort**: Medium (3 tasks, ~2-3 days)

---

### F3: LLM Council Consensus

**Objective**: Aggregate responses from multiple LLM providers for robust extraction.

**Tasks**:

- [ ] LITRIS-q0t: Consensus schema and aggregator design
- [ ] LITRIS-cei: Provider fanout with cost and timeout controls
- [ ] LITRIS-rbh: Integration and tests

**Dependencies**: Existing multi-provider support (Anthropic, OpenAI, Google, Ollama).

**Effort**: High (3 tasks, ~3-4 days)

---

### F4: Research Question Generation

**Objective**: Generate potential research questions from gaps and trends.

**Tasks**:

- [ ] LITRIS-hhn: Prompt templates and config
- [ ] LITRIS-onf: Generation pipeline and ranking
- [ ] LITRIS-q2d: UI/CLI integration

**Dependencies**: Gap detection (F2) should complete first.

**Effort**: Medium (3 tasks, ~2-3 days)

---

### F5: MCP Query Auto-Save

**Objective**: Allow MCP queries to save results to data/query_results/.

**Tasks**:

- [ ] LITRIS-49l: Implement litris_save_query MCP tool

**Dependencies**: None (standalone enhancement).

**Effort**: Low (1 task, ~30 min)

---

## 3. Milestone Roadmap

| Milestone | Features | Target |
|-----------|----------|--------|
| M1: Quick Wins | F5 (MCP Auto-Save) | Immediate |
| M2: Analysis Foundation | F2 (Gap Detection) | Week 1 |
| M3: Visualization | F1 (Citation Network) | Week 2 |
| M4: Generation | F4 (Research Questions) | Week 2-3 |
| M5: Multi-Model | F3 (LLM Council) | Week 3-4 |

---

## 4. Dependency Graph

```
F5 (MCP Save) -----> No dependencies (do first)

F2 (Gap Detection) ---> F4 (Research Questions)
                              |
                              v
                        Depends on gap scores

F1 (Citation Network) ---> Standalone
                              |
                              v
                        Can use existing PyVis patterns

F3 (LLM Council) ---------> Standalone
                              |
                              v
                        Leverages existing provider infra
```

---

## 5. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Citation data sparse in Zotero | Medium | Fall back to title/DOI matching |
| Gap detection subjective | Medium | Document scoring rationale |
| LLM API costs for council | High | Add cost controls and limits |
| Research questions low quality | Medium | Human review step |

---

## 6. Success Criteria

- [ ] Citation network renders for 50+ paper libraries
- [ ] Gap detection produces actionable insights
- [ ] LLM Council reaches consensus on 80%+ of extractions
- [ ] Research questions rated useful by user testing
- [ ] MCP queries persist to query_results/

---

## 7. Detailed Acceptance Criteria

### F1: Citation Network Visualization

| Task | Acceptance Criteria |
|------|---------------------|
| LITRIS-vvx | Schema documented with node/edge types; sample JSON fixture; data source decision recorded |
| LITRIS-4gv | Pipeline generates deterministic graph; handles missing DOIs; tests for edge cases |
| LITRIS-78s | Graph renders in Streamlit; supports hover/click; filters by collection/year |

**Shared Components**: Reuses PyVis from similarity network.

---

### F2: Gap Detection

| Task | Acceptance Criteria |
|------|---------------------|
| LITRIS-c66 | Algorithm documented; inputs/outputs specified; scoring rationale clear |
| LITRIS-xrr | Scoring pipeline runs on full index; produces ranked gap list |
| LITRIS-00o | Report generated in markdown; includes gap descriptions and evidence |

**Shared Components**: Uses embeddings from vector store; extraction data.

---

### F3: LLM Council Consensus

| Task | Acceptance Criteria |
|------|---------------------|
| LITRIS-q0t | Consensus schema defined; aggregation strategy documented |
| LITRIS-cei | Fanout executes with configurable cost/timeout limits; handles provider failures |
| LITRIS-rbh | Integration tests pass; works with 2+ providers |

**Shared Components**: Existing LLM provider abstraction; config.yaml provider settings.

---

### F4: Research Question Generation

| Task | Acceptance Criteria |
|------|---------------------|
| LITRIS-hhn | Prompt templates in config; customizable parameters |
| LITRIS-onf | Pipeline generates and ranks questions; uses gap scores |
| LITRIS-q2d | Questions accessible via CLI and Streamlit UI |

**Shared Components**: Gap detection output (F2); LLM extraction infrastructure.

---

### F5: MCP Query Auto-Save

| Task | Acceptance Criteria |
|------|---------------------|
| LITRIS-49l | New MCP tool saves to query_results/; supports markdown and PDF; updates latest.md |

**Shared Components**: Existing save_results() from retrieval.py.

---

## 8. Cross-Feature Dependencies

```
                    +------------------+
                    |  F5: MCP Save    |  <-- No deps, do first
                    +------------------+

+------------------+     +------------------+
|  F1: Citation    |     |  F2: Gap         |
|  Network         |     |  Detection       |
+------------------+     +------------------+
        |                        |
        v                        v
  (PyVis patterns)         +------------------+
                           |  F4: Research    |
                           |  Questions       |
                           +------------------+
                                  |
                                  v
                           (needs gap scores)

+------------------+
|  F3: LLM Council |  <-- Standalone, leverages provider infra
+------------------+
```

### Task-Level Dependencies

| Task | Depends On |
|------|------------|
| LITRIS-4gv | LITRIS-vvx (schema first) |
| LITRIS-78s | LITRIS-4gv (pipeline first) |
| LITRIS-xrr | LITRIS-c66 (algorithm first) |
| LITRIS-00o | LITRIS-xrr (scoring first) |
| LITRIS-onf | LITRIS-c66, LITRIS-xrr (gap detection) |
| LITRIS-q2d | LITRIS-onf (pipeline first) |
| LITRIS-cei | LITRIS-q0t (schema first) |
| LITRIS-rbh | LITRIS-cei (fanout first) |
