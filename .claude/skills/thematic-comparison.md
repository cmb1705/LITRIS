---
name: thematic-comparison
description: Structured thematic comparison across all indexed papers. Searches the corpus for a theme, cross-references extractions and full text, and produces a comparative analysis with verbatim quotes and page numbers.
---

# Thematic Comparison Skill

Produce a structured comparison of how a specific theme, concept, or research
question is treated across all papers in the LITRIS index.

## Usage

```
/thematic-comparison "democratic legitimacy"
/thematic-comparison "generalizability limitations"
/thematic-comparison "citizen participation"
```

If no argument is given, ask the user what theme to compare.

## Data Sources

### LITRIS MCP Tools
- `litris_search(query, top_k)` -- semantic search across indexed chunks
- `litris_get_paper(paper_id)` -- full metadata + extraction for a paper
- `litris_summary()` -- index statistics, paper count, collections

### File-Based Data
- **Paper metadata**: `data/index/papers.json`
- **Structured extractions**: `data/index/extractions.json`
- **Full text cache**: `data/cache/pdf_text/*.txt`
  - Page-delimited with `--- Page N ---` markers
  - First lines contain paper title for identification

## Methodology

### Step 1: Corpus Overview

Call `litris_summary()` to get the corpus size and composition.

### Step 2: Semantic Search

Call `litris_search` with the user's theme, `top_k=20` for broad coverage. This
returns relevant chunks across papers with similarity scores.

### Step 3: Extraction Cross-Reference

Read `data/index/extractions.json` and search each paper's extraction for the
theme across all structured fields: `thesis_statement`, `key_findings`,
`key_claims`, `conclusions`, `limitations`, `future_directions`, and
`theoretical_framework`.

### Step 4: Full-Text Verification

For papers that appear in search results, grep the `data/cache/pdf_text/*.txt`
files for the theme and related terms. Capture surrounding context (5 lines
before and after) to understand how each author treats the concept.

### Step 5: Construct Comparison

For each paper that engages with the theme, extract:

- **Engagement level**: Is the theme central, substantial, peripheral, or absent?
- **Author's position**: What do they claim about it?
- **Evidence type**: Theoretical argument, empirical finding, limitation, or
  future direction?
- **Exact quote**: A representative verbatim passage with page number from
  `--- Page N ---` markers
- **Analytical vocabulary**: What terms does the author use?

## Output Format

```markdown
# Thematic Comparison: [Theme]

**Corpus:** [N] papers indexed in LITRIS
**Papers engaging with theme:** [M] of [N]
**Date:** [current date]

---

## Summary

[2-3 sentence synthesis: where there is consensus, where there is tension,
and what is absent]

## Comparative Table

| Paper | Role of Theme | Position | Evidence | Key Quote (page) |
|-------|--------------|----------|----------|-------------------|
| Author (Year) | Central/Peripheral | [summary] | [type] | "[quote]" (p. X) |

## Detailed Analysis

### [Author (Year)] -- [Paper Title]

**Engagement level:** Central / Substantial / Peripheral / Absent

**How the theme appears:**
[1-2 sentences on where and how the concept enters the paper]

**Key passage (p. X):**
> [Verbatim quote from full text]

**Analytical vocabulary:**
[Terms the author uses related to this theme]

**Connection to other papers:**
[How this paper's treatment relates to or contrasts with others]

---

## Cross-Cutting Observations

1. [Pattern observed across multiple papers]
2. [Tension or contradiction between papers]
3. [Gap -- what the corpus does not address regarding this theme]

## Papers Not Engaging with Theme

[List papers that do not address the theme, with brief note on why]
```

## Execution Strategy

For efficiency, parallelize full-text searches across papers using subagents
(one per paper, up to 4-5 in parallel). Compile results into the unified format.

Offer to save the result via `litris_save_query` if the user wants it persisted
in `data/query_results/`.

## Quality Standards

- Every quote must be verbatim from the full-text cache, not paraphrased
- Every page number must be derived from `--- Page N ---` markers in the cache
- Distinguish between substantive engagement and mere keyword mention
- Note when a paper uses different vocabulary for the same concept
- Flag when extraction summaries differ from actual paper text
