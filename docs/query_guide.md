# Query Guide

This guide provides examples and strategies for effective literature searches.

## How Search Works

The system uses semantic search powered by sentence embeddings. This means:

- **Meaning over keywords**: Finds conceptually similar content
- **Multiple chunk types**: Searches across different extraction fields
- **Relevance scoring**: Results ranked by similarity

## Basic Queries

### Topic Search

```bash
# Find papers about a topic
python scripts/query_index.py -q "social network analysis"

# Find papers about a methodology
python scripts/query_index.py -q "qualitative interview methods"

# Find papers about a theory
python scripts/query_index.py -q "actor-network theory applications"
```

### Research Question Style

Frame queries as research questions for better results:

```bash
# How questions
python scripts/query_index.py -q "how do researchers measure network centrality"

# What questions
python scripts/query_index.py -q "what factors influence citation patterns"

# Why questions
python scripts/query_index.py -q "why do academic collaborations form"
```

## Advanced Queries

### Combining Concepts

```bash
# Method + domain
python scripts/query_index.py -q "machine learning applications in bibliometrics"

# Theory + phenomenon
python scripts/query_index.py -q "social capital theory in online communities"
```

### Finding Specific Types

```bash
# Literature reviews
python scripts/query_index.py -q "systematic review of citation analysis methods"

# Methodological papers
python scripts/query_index.py -q "novel approach to measuring research impact"

# Empirical studies
python scripts/query_index.py -q "empirical analysis of collaboration networks"
```

## Filtering Results

### By Publication Year

```bash
# Recent papers only
python scripts/query_index.py -q "your query" --year-min 2020

# Historical papers
python scripts/query_index.py -q "your query" --year-max 2010

# Specific range
python scripts/query_index.py -q "your query" --year-min 2015 --year-max 2020
```

### By Collection

```bash
# Papers in a specific Zotero collection
python scripts/query_index.py -q "your query" --collection "Methods"
```

### Limiting Results

```bash
# Get more results
python scripts/query_index.py -q "your query" -n 50

# Get fewer results
python scripts/query_index.py -q "your query" -n 5
```

## Output Formats

### Human-Readable (Default)

```bash
python scripts/query_index.py -q "network analysis"
```

Output shows:
- Title
- Authors and year
- Relevance score
- Matching text excerpt
- Key extraction details

### JSON for Processing

```bash
python scripts/query_index.py -q "network analysis" -f json > results.json
```

Includes full paper and extraction data.

### CSV for Spreadsheets

```bash
python scripts/query_index.py -q "network analysis" -f csv > results.csv
```

Includes basic metadata fields.

### Markdown for Reports

```bash
python scripts/query_index.py -q "network analysis" -f markdown > results.md
```

Formatted for documentation or reports.

## Query Strategies

### Literature Review Workflow

1. **Broad initial search**
   ```bash
   python scripts/query_index.py -q "your main topic" -n 50 -f json > initial.json
   ```

2. **Refine with specific aspects**
   ```bash
   python scripts/query_index.py -q "specific methodology aspect"
   python scripts/query_index.py -q "theoretical framework aspect"
   ```

3. **Check for gaps**
   ```bash
   python scripts/query_index.py -q "limitations in this field"
   python scripts/query_index.py -q "future research directions"
   ```

### Finding Key Papers

```bash
# Foundational works
python scripts/query_index.py -q "seminal work on topic" --year-max 2010

# State of the art
python scripts/query_index.py -q "recent advances in topic" --year-min 2020

# Methodological foundations
python scripts/query_index.py -q "methodology for topic development"
```

### Cross-Disciplinary Search

```bash
# Find connections across fields
python scripts/query_index.py -q "concept applied in different contexts"

# Method transfer
python scripts/query_index.py -q "technique borrowed from other field"
```

## Understanding Results

### Relevance Scores

- **0.8-1.0**: Highly relevant, direct match
- **0.6-0.8**: Relevant, related concepts
- **0.4-0.6**: Somewhat relevant, tangential
- **Below 0.4**: Low relevance, may not be useful

### Chunk Types

Results may come from different extraction fields:
- `full_summary`: Complete paper summary
- `thesis`: Main argument
- `findings`: Key results
- `methodology`: Methods used
- `claims`: Arguments made

## Tips for Better Results

1. **Be specific**: "network centrality measures" > "networks"
2. **Use domain terms**: Include technical vocabulary
3. **Try variations**: Same concept, different phrasings
4. **Check related papers**: Look at what similar papers cite
5. **Use filters**: Narrow by year or collection
6. **Increase results**: Sometimes relevant papers appear lower

## Examples by Use Case

### Writing a Methods Section

```bash
python scripts/query_index.py -q "how to analyze network data"
python scripts/query_index.py -q "statistical methods for network analysis"
python scripts/query_index.py -q "software tools for network visualization"
```

### Building Theory

```bash
python scripts/query_index.py -q "theoretical frameworks for understanding phenomena"
python scripts/query_index.py -q "conceptual models in field"
python scripts/query_index.py -q "critique of existing theories"
```

### Finding Gaps

```bash
python scripts/query_index.py -q "limitations of current research"
python scripts/query_index.py -q "underexplored aspects of topic"
python scripts/query_index.py -q "calls for future research"
```

### Identifying Methods

```bash
python scripts/query_index.py -q "novel methodological approaches"
python scripts/query_index.py -q "mixed methods studies"
python scripts/query_index.py -q "longitudinal analysis techniques"
```
