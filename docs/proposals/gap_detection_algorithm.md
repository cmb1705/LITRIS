# Gap Detection Algorithm Specification

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Created | 2026-01-22 |
| Task | LITRIS-c66 |
| Implementation | src/analysis/gap_detection.py |

---

## 1. Definition of a Research Gap

A **research gap** in LITRIS is defined as:

> An area, topic, methodology, time period, or future research direction that is **underrepresented** relative to the overall corpus, suggesting potential opportunities for further investigation.

### Gap Categories

| Category | Description | Signal |
|----------|-------------|--------|
| Topic Gap | Subject areas mentioned but not deeply covered | Low topic count relative to corpus |
| Methodology Gap | Research methods referenced but rarely applied | Low methodology count |
| Temporal Gap | Time periods with sparse publications | Years with few papers |
| Future Direction Gap | Author-identified directions not yet pursued | Future directions with low subsequent coverage |

---

## 2. Required Inputs

### 2.1 Paper Metadata

Source: `data/index/papers.json`

| Field | Required | Used For |
|-------|----------|----------|
| paper_id | Yes | Unique identifier |
| title | Yes | Token extraction |
| year | Yes | Temporal analysis |
| abstract | Optional | Extended token coverage |
| collections | Optional | Filtering |

### 2.2 Extraction Data

Source: `data/index/extractions.json`

| Field | Required | Used For |
|-------|----------|----------|
| topics | Yes | Topic gap detection |
| methodology | Yes | Methodology gap detection |
| future_directions | Yes | Future direction gaps |
| thesis_statement | Optional | Context for evidence |

### 2.3 Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_items | 10 | Maximum gaps to report per category |
| min_count | 2 | Minimum mentions to be considered |
| quantile | 0.2 | Percentile threshold for "underrepresented" |
| evidence_limit | 3 | Max evidence papers per gap |
| include_abstracts | false | Include abstract text in token index |
| sparse_year_max_count | 1 | Max papers for year to be "sparse" |
| future_direction_min_mentions | 1 | Min times a direction must appear |
| future_direction_max_coverage | 1 | Max coverage to be "low" |
| token_min_len | 3 | Minimum token length |

---

## 3. Algorithm Description

### 3.1 Topic and Methodology Gaps

```
1. For each paper with extraction:
   a. Extract topics list from extraction.topics
   b. Normalize topics (lowercase, trim)
   c. Increment topic_counts[topic]
   d. Store paper as evidence for topic

2. Calculate threshold:
   threshold = quantile_value(topic_counts, config.quantile)

3. Select underrepresented:
   gaps = topics where:
     - count >= min_count (mentioned enough to be real)
     - count <= threshold (but underrepresented)

4. Sort by count ascending (most underrepresented first)

5. Return top max_items with evidence
```

Same algorithm applies to methodologies.

### 3.2 Temporal Gaps

```
1. Count papers per publication year
2. Identify sparse years:
   sparse = years where count <= sparse_year_max_count
3. Group into ranges (consecutive sparse years)
4. Report ranges with year bounds
```

### 3.3 Future Direction Gaps

```
1. Collect future_directions from all extractions
2. Build token index from all paper content
3. For each unique direction:
   a. Extract key tokens from direction text
   b. Count how many papers cover those tokens
   c. If coverage <= future_direction_max_coverage:
      - Flag as "low coverage" gap
4. Sort by mention count (directions multiple authors identify)
5. Return top max_items with source papers
```

---

## 4. Scoring and Ranking

### 4.1 Topic/Methodology Ranking

Gaps are ranked by **ascending count** (fewer mentions = higher priority).

Rationale: Topics mentioned by 2-3 papers are "known" but unexplored, representing clear opportunities.

### 4.2 Future Direction Ranking

Ranked by **descending mention count** (more authors identify = higher priority).

Rationale: If multiple authors independently identify the same future direction, it signals community-recognized opportunity.

### 4.3 Evidence Quality

Each gap includes evidence papers showing:
- Paper title and year
- Specific extraction text mentioning the gap

---

## 5. Output Schema

```json
{
  "generated_at": "ISO8601 timestamp",
  "corpus": {
    "papers": 347,
    "extractions": 320,
    "collections": ["GNN", "Scientometrics"]
  },
  "parameters": {
    "max_items": 10,
    "min_count": 2,
    "quantile": 0.2
  },
  "topics_underrepresented": [
    {
      "term": "knowledge distillation",
      "count": 3,
      "evidence": [
        {"paper_id": "...", "title": "...", "year": 2023}
      ]
    }
  ],
  "methodologies_underrepresented": [...],
  "year_gaps": {
    "sparse_years": [2015, 2016],
    "ranges": ["2015-2016"]
  },
  "future_directions": [
    {
      "direction": "Apply HGTs to temporal citation networks",
      "mention_count": 3,
      "coverage": 0,
      "source_papers": [...]
    }
  ],
  "notes": ["Interpretation guidance..."]
}
```

---

## 6. Assumptions and Limitations

### Assumptions

1. LLM extractions accurately capture paper topics and methodologies
2. Topics/methodologies are normalized consistently across papers
3. Coverage proxy (token matching) approximates semantic relatedness
4. More mentions of a gap = higher confidence it's real

### Limitations

1. **Extraction quality**: Gaps depend on LLM extraction accuracy
2. **Vocabulary mismatch**: Synonyms may split counts artificially
3. **Corpus bias**: Gaps reflect what's missing from THIS library, not the field
4. **No external validation**: Cannot verify against broader literature
5. **Token matching**: Future direction coverage is approximate

---

## 7. Interpretation Guidance

| Gap Type | Interpretation | Action |
|----------|----------------|--------|
| Topic with count 2-3 | Mentioned but not central | Consider for literature review |
| Methodology with count 1-2 | Rare approach in corpus | Evaluate if applicable to research |
| Year range gap | Publication pause | Check if field-wide or collection gap |
| Future direction, coverage 0 | Unfulfilled research opportunity | High priority for original research |

---

## 8. Example Usage

```python
from src.analysis.gap_detection import (
    GapDetectionConfig,
    load_gap_report,
    format_gap_report_markdown,
)
from pathlib import Path

config = GapDetectionConfig(
    max_items=10,
    min_count=2,
    quantile=0.2,
)

report = load_gap_report(
    index_dir=Path("data/index"),
    config=config,
    collections=["GNN"],  # Optional filter
)

markdown = format_gap_report_markdown(report)
print(markdown)
```
