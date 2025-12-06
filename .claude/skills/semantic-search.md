---
name: semantic-search
description: Guide effective query formulation and result interpretation for the semantic search system.
---

# Semantic Search Skill

## Purpose

Guide effective query formulation and result interpretation for the semantic search system.

## When to Use

Invoke this skill when:
- Formulating search queries
- Interpreting search results
- Optimizing retrieval strategies
- Troubleshooting poor results

## How Semantic Search Works

### Embedding-Based Retrieval

1. **Query embedding**: Your query is converted to a vector
2. **Similarity matching**: Compared against stored chunk vectors
3. **Ranking**: Results ordered by cosine similarity
4. **Filtering**: Metadata filters applied
5. **Return**: Top-k results with scores

### What Makes Semantic Search Different

| Keyword Search | Semantic Search |
|---------------|-----------------|
| Exact term matching | Meaning matching |
| "network" only finds "network" | "network" finds "graph", "connections" |
| Boolean operators | Natural language |
| Precise but brittle | Flexible but fuzzy |

## Query Formulation Strategies

### Effective Query Patterns

**Conceptual queries** (best for exploration):
```
"methods for detecting emerging research topics"
"theoretical frameworks for understanding scientific progress"
```

**Specific queries** (best for targeted retrieval):
```
"Leiden community detection algorithm parameters"
"bibliographic coupling vs co-citation analysis"
```

**Question-based queries** (natural and effective):
```
"how do citation networks reveal research influence"
"what factors predict paper impact"
```

### Query Length Guidelines

| Length | Effectiveness | Use Case |
|--------|---------------|----------|
| 3-5 words | Good | Specific concepts |
| 6-15 words | Best | Detailed questions |
| 15-30 words | Good | Complex queries |
| 30+ words | Diminishing returns | Avoid unless necessary |

### Terms to Include

- **Domain terms**: Specific vocabulary of the field
- **Method terms**: How things are studied
- **Concept terms**: What is being studied
- **Relationship terms**: How concepts connect

### Terms to Avoid

- **Stop words**: "the", "a", "is" (embedding handles these)
- **Hedging**: "maybe", "possibly", "might"
- **Meta-language**: "papers about", "research on" (implicit)

## Filter Usage

### Chunk Type Filters

| Filter | What It Matches | When to Use |
|--------|-----------------|-------------|
| abstract | Paper summaries | Broad topic search |
| thesis | Core arguments | Finding main claims |
| methodology | Research methods | Methods comparison |
| finding | Results | Evidence gathering |
| claim | Specific assertions | Argument support |
| limitation | Acknowledged gaps | Gap identification |
| future_work | Suggested research | Opportunity finding |

### Combining Filters

**Narrow to broad:**
1. Start with specific chunk_type
2. If too few results, broaden
3. Add year/collection if needed

**Example progression:**
```
chunk_types=["thesis"], year_min=2020  # Very specific
chunk_types=["thesis", "finding"]       # Broader
chunk_types=None, year_min=2020         # All types, recent
```

## Result Interpretation

### Score Meaning

| Score | Interpretation | Action |
|-------|----------------|--------|
| 0.90+ | Near exact match | High confidence |
| 0.80-0.90 | Strong match | Good result |
| 0.70-0.80 | Related content | Review carefully |
| 0.60-0.70 | Tangential | May be useful |
| <0.60 | Weak match | Likely irrelevant |

### Result Patterns

**Good search:**
- Top results highly relevant
- Scores clustered high (0.75+)
- Multiple papers on topic

**Poor search indicators:**
- Top score < 0.70
- Wide score spread
- Unexpected results in top 5

### Troubleshooting Poor Results

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| No relevant results | Query too specific | Broaden terms |
| Too many results | Query too broad | Add specificity |
| Unexpected results | Wrong domain terms | Use field vocabulary |
| Missing known papers | Wrong chunk type | Try different filters |

## Multi-Query Strategies

### Synonym Expansion

Run multiple queries with synonyms:
```
Query 1: "citation network analysis"
Query 2: "bibliometric network methods"
Query 3: "reference graph analysis"
```
Merge results, keeping highest scores.

### Aspect Decomposition

Break complex topics into aspects:
```
Theory: "theoretical framework [topic]"
Methods: "methodology for [topic]"
Findings: "empirical results [topic]"
```

### Iterative Refinement

1. Run initial query
2. Examine top results
3. Extract useful terms from good results
4. Reformulate query
5. Repeat until satisfied

## Query Examples by Task

### Literature Review

```
"comprehensive review of [topic] literature"
"survey of approaches to [topic]"
"state of the art in [field]"
```

### Gap Identification

```
"limitations of current [topic] research"
"challenges in [field]"
"future research directions [topic]"
```

### Method Finding

```
"methods for analyzing [phenomenon]"
"approaches to measuring [concept]"
"techniques for [task]"
```

### Evidence Gathering

```
"empirical evidence for [claim]"
"studies showing [relationship]"
"findings about [topic]"
```
