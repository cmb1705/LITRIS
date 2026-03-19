# LITRIS vs Minty: Comprehensive Comparison Report

## Context

**Why this analysis exists**: To evaluate LITRIS (our literature review indexing system) against Minty (MINT Research Lab's agentic research infrastructure) and identify features worth adopting, architectural differences, and strategic gaps in either direction.

**Minty source**: MINT Research Lab's agentic research infrastructure.

---

## Executive Summary

LITRIS and Minty share a core mission -- extracting structured knowledge from academic papers and making it searchable -- but they operate at fundamentally different scales and with different philosophies. LITRIS is a **focused extraction-and-search pipeline** optimized for individual researchers managing their personal Zotero libraries. Minty is a **full research lab operating system** with 18 background daemons, multi-source content discovery, team communication integration, and a persistent AI colleague persona.

The comparison reveals that LITRIS excels in areas Minty does not address (multi-reference-manager support, document type classification, federated search, batch API cost optimization) while Minty offers capabilities LITRIS lacks entirely (content discovery, hierarchical summarization, multi-round agentic search, institutional memory, team collaboration).

---

## 1. What is the Same / Similar

### 1.1 Core Paper Extraction Pipeline

Both systems extract structured knowledge from PDFs using LLMs:

| Aspect | LITRIS | Minty |
|--------|--------|-------|
| PDF to text | PyMuPDF + OCR fallback | 6-tier cascade (Companion, arXiv HTML, ar5iv, Marker, PyMuPDF, Tesseract) |
| LLM extraction | Multi-provider (Anthropic, OpenAI, Google, Ollama, llama.cpp) | Multi-provider (Claude Opus, Codex GPT-5.2) |
| Structured output | 45-field PaperExtraction schema | 41-question semantic analysis |
| Caching | Content-hash based | Manifest database per paper |
| Incremental updates | Change detection via file hash | Backlog queue with state tracking |

### 1.2 Vector Search

Both use embedding-based semantic search over paper corpora:

| Aspect | LITRIS | Minty |
|--------|--------|-------|
| Vector DB | ChromaDB | LanceDB (papers) + ChromaDB (messages, workspace) |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 (384d) | Voyage context-3 (2048d) |
| Chunks per paper | 9 semantic types | Multiple (full text + sections + summaries) |
| Similarity metric | Cosine distance | Cosine distance |

### 1.3 Zotero Integration

Both integrate with Zotero for reference management (LITRIS as primary source, Minty as final cataloging destination in the pipeline).

### 1.4 Claude Code Integration

Both are built to work with Claude Code as the agent runtime:
- Both use CLAUDE.md for agent instructions
- Both define custom skills and commands
- Both use hooks for safety and quality enforcement

### 1.5 Metadata Enrichment

Both pull metadata from external sources (CrossRef, Semantic Scholar in Minty; Zotero metadata in LITRIS).

### 1.6 Quality Validation

Both validate extraction quality -- LITRIS via confidence scores and type-specific required fields; Minty via QA verification phase in deep reviews.

---

## 2. What is Different

### 2.1 Scale and Scope

| Dimension | LITRIS | Minty |
|-----------|--------|-------|
| **Target user** | Individual researcher | Research lab (29 members, 11 institutions) |
| **Corpus size** | 332 papers, 3,746 chunks | 2,400+ papers, 270,858 chunks |
| **Operating model** | On-demand CLI/script execution | 18 always-on background daemons |
| **Platform** | Cross-platform (Windows primary) | macOS primary (launchd-dependent) |
| **Infrastructure** | Local machine, no cloud services | Mac Studio + Cloudflare tunnel + Google Drive + Ghost CMS |

### 2.2 Content Discovery vs Content Processing

**LITRIS**: Starts with papers you already have in your reference manager. No discovery pipeline. You find papers, add them to Zotero, then LITRIS extracts and indexes them.

**Minty**: Active content discovery from 6 sources (Twitter, Bluesky, RSS, arXiv, LinkedIn, email). Classifies ~3,000 items/day. Human curation via Slack reactions. Papers flow into the corpus automatically.

### 2.3 Extraction Schema Philosophy

**LITRIS** (45 fields, structured):
- Thesis statement, research questions, theoretical framework
- Detailed methodology (approach, design, data sources, analysis methods, sample size, time period)
- Key findings with evidence type + significance + page references
- Key claims with support type + strength + page references
- Document type classification with confidence scores
- Discipline tags, keywords, contribution summary

**Minty** (41 questions, analytical):
- Research core (q01-05): research question, thesis, claims, evidence, limitations
- Methodology (q06-10): paradigm, methods, datasets, reproducibility, framework
- Context (q11-14): traditions, citations, assumptions, counterarguments
- AI-specific (q15-18): LLM roles, models discussed, capability/risk claims
- Meta (q19-26): novelty, field, audience, implications, quality rating 1-5
- Advanced (q27-41): publication type, institutions, timing, disciplinary bridges, power dynamics, dual-use concerns

**Key difference**: LITRIS extracts *structured data* (enums, lists, typed fields) optimized for search filtering. Minty asks *analytical questions* optimized for understanding and synthesis. LITRIS is more database-oriented; Minty is more intelligence-analysis-oriented.

### 2.4 Search Architecture

**LITRIS**:
- Single-pass vector search with metadata filters
- Federated multi-index search (query multiple libraries)
- Chunk-type filtering (search only methodology sections, etc.)
- Recency boost weighting
- MCP server for Claude Code integration (6 tools)

**Minty**:
- 7 interactive search modes (fast, deep review, conversation, news, etc.)
- Multi-round search with semantic expansion and gap analysis
- Reciprocal Rank Fusion (RRF) across 49 dimensions
- Agentic search: LLM reasons about results, generates follow-up queries
- Warm search server with pre-loaded LanceDB
- Deep Review: 4-phase pipeline (search -> parallel readers -> synthesis -> QA)

### 2.5 Post-Extraction Processing

**LITRIS**: Extraction -> Embedding -> Index. Done.

**Minty**: Extraction -> RAPTOR hierarchical summarization -> Topic clustering (UMAP + HDBSCAN) -> Pre-computed similarity pairs (11,570) -> Multiple embedding representations -> Interactive paper map visualization.

### 2.6 Memory and Persistence

**LITRIS**: No institutional memory system. Each session is independent. Relies on beads for task tracking.

**Minty**: Three-tier memory:
- FACTS.md (81KB): Durable institutional knowledge, pruned during retrospectives
- RECENT.md (29KB): Rolling 10-session context window
- sessions.db (SQLite + FTS5): Full session history with semantic search

### 2.7 Team Collaboration

**LITRIS**: Single-user system. No collaboration features.

**Minty**: Full team infrastructure:
- Slack integration (3 bots, 6+ channels)
- Daily digest emails to 29 lab members
- Slack message indexing + semantic search
- Workspace file indexing
- Shared corpus accessible to all members

### 2.8 Agent Persona

**LITRIS**: Task-oriented agents (pipeline-engineer, literature-analyst, etc.) with no persistent identity.

**Minty**: Named agent "Minty" with 16,800 words of persona documentation across 21 documents. Treated as intellectual colleague, not tool. Encouraged to express own views. Persistent identity across sessions.

---

## 3. What Minty Does That LITRIS Should Incorporate

### 3.1 HIGH PRIORITY -- Clear value with reasonable effort

#### A. RAPTOR Hierarchical Summarization
**What**: Recursive abstraction creating multi-level summaries (sentence -> paragraph -> section -> full paper). Enables search at different granularity levels.
**Why**: Currently LITRIS chunks are flat. A researcher searching for "network centrality methods" would benefit from matching at summary level vs. detail level depending on intent.
**Effort**: Medium. Add a post-extraction stage that generates 2-3 summary layers per paper.

#### B. Multi-Round Agentic Search
**What**: LLM generates initial query -> searches -> analyzes gaps -> generates follow-up queries -> searches again. Minty does up to 2 gap-analysis rounds.
**Why**: Single-pass vector search often misses relevant papers that use different terminology. An agentic search loop dramatically improves recall.
**Effort**: Medium. Build on top of existing MCP tools. The LLM orchestrating search can call litris_search iteratively.

#### C. Reciprocal Rank Fusion (RRF)
**What**: Combine results from multiple query reformulations into a single ranked list. Minty uses RRF across 49 dimensions.
**Why**: Improves search quality significantly over single-query retrieval. Well-established IR technique.
**Effort**: Low-Medium. Implement RRF scoring in the search engine, generate query variants via LLM.

#### D. Pre-Computed Similarity Pairs
**What**: Minty pre-computes 11,570 paper-to-paper similarity pairs for instant "related papers" lookup.
**Why**: `litris_similar` currently computes similarity at query time. Pre-computing enables instant results and supports network visualization.
**Effort**: Low. Run pairwise similarity on all paper embeddings during index build. Store in structured store.

#### E. Deep Review / Literature Synthesis Pipeline
**What**: 4-phase pipeline: multi-round search -> parallel paper reading -> synthesis -> QA verification. Produces 3,000-5,000 word literature reviews.
**Why**: This is the killer feature for academic users. Going from "search results" to "synthesized review" is the gap between a tool and a research assistant.
**Effort**: High. Requires orchestrating multiple LLM calls, parallel reading, and synthesis. But the existing thematic-comparison skill is a foundation.

#### F. Topic Clustering
**What**: UMAP + HDBSCAN clustering to identify 8 topic clusters with 114 micro-topics across the corpus.
**Why**: Provides structural understanding of the corpus. Enables gap analysis ("which areas are underrepresented?") and visual exploration.
**Effort**: Medium. Both libraries are well-established. Can run as a post-build analysis step.

### 3.2 MEDIUM PRIORITY -- Valuable but less urgent

#### G. Institutional Memory (FACTS.md / RECENT.md Pattern)
**What**: Persistent knowledge files that accumulate learned patterns, preferences, and context across sessions.
**Why**: LITRIS already has CLAUDE.md and beads, but lacks a structured "lessons learned" system that grows with use. Would help the agent remember corpus-specific insights.
**Effort**: Low. Claude Code's auto-memory system already provides this foundation -- could formalize it.

#### H. Quality Rating (1-5 Scale)
**What**: Minty's q24 assigns a 1-5 quality rating to every paper based on methodology, evidence, and contribution.
**Why**: Enables filtering search results by quality. "Show me only high-quality papers on X."
**Effort**: Low. Add a `quality_rating` field to the extraction schema. Include in the LLM prompt.

#### I. Content Discovery Pipeline
**What**: Automated fetching from arXiv, RSS, social media with LLM classification and human curation.
**Why**: Moves LITRIS from "process what you have" to "find what you need." But this is a major scope expansion.
**Effort**: Very High. This is essentially building a separate system. Consider as a future phase.

#### J. Enhanced PDF Conversion Cascade
**What**: 6-tier fallback: Companion -> arXiv HTML -> ar5iv -> Marker -> PyMuPDF -> Tesseract.
**Why**: LITRIS has PyMuPDF + OCR. Adding arXiv HTML and Marker would significantly improve extraction quality for certain paper types.
**Effort**: Medium. Marker is a pip install. arXiv HTML is an API call. Could add as additional fallback tiers.

#### K. Paper Network Visualization
**What**: Interactive HTML network graph showing paper relationships, discipline clustering, and exploration filters.
**Why**: Visual exploration of a corpus is powerful for identifying gaps and connections.
**Effort**: Medium. LITRIS already has PyVis + NetworkX dependencies. Needs pre-computed similarities (item D).

### 3.3 LOW PRIORITY -- Nice to have / scope expansion

#### L. Background Daemons for Continuous Processing
**What**: Always-on services for ingestion, indexing, health monitoring.
**Why**: Currently LITRIS is run manually via scripts. Automation would reduce friction but adds operational complexity.
**Effort**: High. Windows service management is different from launchd. Less natural fit.

#### M. Team Messaging Integration (Slack)
**What**: Index Slack messages, enable corpus search via @mention, share results in channels.
**Why**: Only valuable if LITRIS becomes a multi-user system. Current single-user design doesn't need this.

#### N. Daily Digest Generation
**What**: Automated narrative summary of recent research developments.
**Why**: Useful for teams. Overkill for individual use.

#### O. Session Database with FTS5
**What**: SQLite database tracking all agent sessions with full-text search.
**Why**: Beads already provides persistent task tracking. A session database adds historical search but duplicates some functionality.

---

## 4. What LITRIS Does That Minty Does Not

### 4.1 Multi-Reference-Manager Support
LITRIS supports 7 reference sources (Zotero, BibTeX, Mendeley, EndNote, Paperpile, PDF folders, federated). Minty is Zotero-only for cataloging. This is a significant advantage for researchers using different tools.

### 4.2 Document Type Classification
LITRIS has a two-tier classification system (metadata + content heuristics) producing 7 document types with type-specific extraction strategies and validation rules. Minty classifies by publication type but doesn't adapt extraction behavior per type.

### 4.3 Federated Multi-Index Search
LITRIS can search across multiple independent indexes simultaneously with configurable merge strategies (interleave, concat, rerank) and source weighting. Minty searches a single corpus.

### 4.4 Batch API Cost Optimization
LITRIS supports Anthropic's Batch API (50% cost reduction, 24-hour processing) for bulk extraction. Minty uses real-time API calls only.

### 4.5 Local LLM Support
LITRIS supports Ollama and llama.cpp for fully offline, free extraction. Minty requires cloud API access.

### 4.6 Structured Enum Extraction
LITRIS enforces strict enum validation (evidence_type, support_type, significance, strength) producing machine-queryable structured data. Minty's 41-question system produces more analytical text responses.

### 4.7 MCP Server Integration
LITRIS exposes 6 MCP tools for direct Claude Code integration. Minty uses Slack as its primary interface, with Claude Code as the agent runtime.

### 4.8 Cross-Platform Support
LITRIS runs on Windows, macOS, and Linux. Minty is macOS-first with significant porting effort required for other platforms.

### 4.9 Web UI (Streamlit)
LITRIS provides an interactive Streamlit search workbench. Minty's interface is Slack-based.

### 4.10 Provider Comparison Tooling
LITRIS has dedicated extraction-comparator agents and provider-benchmark skills for side-by-side evaluation of Anthropic vs OpenAI extraction quality. Minty uses multiple providers but doesn't have comparison tooling.

---

## 5. Architectural Differences Summary

| Dimension | LITRIS | Minty |
|-----------|--------|-------|
| **Architecture** | Pipeline (build -> query) | Event-driven (daemons + queues) |
| **Execution model** | On-demand scripts | 18 always-on daemons |
| **Primary interface** | CLI + MCP + Streamlit | Slack + Claude Code |
| **Vector DB** | ChromaDB (384d) | LanceDB (2048d) + ChromaDB |
| **Embedding model** | all-MiniLM-L6-v2 (22MB, free) | Voyage context-3 (paid, higher quality) |
| **LLM strategy** | Multi-provider, cost-optimized | Opus for substance, Sonnet for retrieval |
| **Search** | Single-pass + federated | Multi-round agentic + RRF |
| **Post-processing** | None | RAPTOR + clustering + similarity |
| **Memory** | Beads + auto-memory | FACTS.md + RECENT.md + sessions.db |
| **Collaboration** | Single-user | Team (29 members) |
| **Platform** | Cross-platform (Windows primary) | macOS only |

---

## 6. Embedding Quality Gap

This deserves special attention. LITRIS uses `all-MiniLM-L6-v2` (384 dimensions, free, 22MB). Minty uses `Voyage context-3` (2048 dimensions, paid, higher quality). The 5.3x dimensionality difference and Voyage's superior training on academic text means Minty's search quality is likely meaningfully better.

**Recommendation**: Evaluate upgrading to a higher-quality embedding model. Options:
- `Voyage-3` or `Voyage context-3` (paid, best quality for academic text)
- `nomic-embed-text-v1.5` (free, 768d, strong academic performance)
- `bge-large-en-v1.5` (free, 1024d)
- Keep MiniLM as default but allow configurable model in config.yaml

---

## 7. Recommended Adoption Roadmap

### Phase 1: Quick Wins (1-2 sessions each)
1. **Pre-computed similarity pairs** (item D) -- enables instant `litris_similar`
2. **Quality rating field** (item H) -- add to extraction schema
3. **Formalize institutional memory** (item G) -- structure the existing memory directory

### Phase 2: Search Improvements (2-4 sessions each)
4. **RRF multi-query search** (item C) -- generate query variants, fuse results
5. **Embedding model upgrade** -- evaluate Voyage or nomic-embed alternatives
6. **Multi-round agentic search** (item B) -- iterative search with gap analysis

### Phase 3: Post-Processing Enrichment (3-5 sessions each)
7. **RAPTOR hierarchical summarization** (item A) -- multi-level summaries
8. **Topic clustering** (item F) -- UMAP + HDBSCAN on corpus
9. **Paper network visualization** (item K) -- interactive graph explorer

### Phase 4: Synthesis Capability (5+ sessions)
10. **Deep Review pipeline** (item E) -- the full search -> read -> synthesize -> QA workflow

---

## 8. Verification Plan

After cloning the Minty repository, verify this analysis by:
1. Reading their `SETUP_GUIDE.md` in full (7,000 lines)
2. Examining the actual 41-question prompt template
3. Reviewing the RAPTOR implementation
4. Studying the RRF search implementation
5. Comparing their CLAUDE.md template against ours

---

## 9. Key Takeaway

LITRIS is a strong, well-architected extraction and search system that excels at what it does. Minty is a broader research infrastructure platform. The most impactful adoptions from Minty are in the **search quality** domain (RRF, agentic multi-round search, better embeddings) and **post-extraction enrichment** (RAPTOR summaries, topic clustering, similarity pre-computation). These would transform LITRIS from "find papers by similarity" to "understand your corpus and synthesize knowledge."

The areas where LITRIS already leads (multi-reference-manager support, document type awareness, federated search, cost optimization, cross-platform) should be preserved and strengthened as differentiators.
