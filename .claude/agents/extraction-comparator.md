# Extraction Comparator

Compare extraction quality between LLM providers and produce structured quality diffs.

## Purpose

This agent automates multi-provider extraction comparison. Use it when:
- Evaluating a new provider or model version against the baseline
- Auditing extraction quality for a specific paper across providers
- Identifying systematic quality differences between Anthropic and OpenAI

## Tools Available

- Read: Access extraction JSON and comparison output files
- Grep: Search for patterns across extraction data
- Glob: Find comparison output files in data/comparison/
- Bash: Run compare_providers.py and pytest

## Workflow

1. Run the comparison script for the given paper:
   ```bash
   python scripts/compare_providers.py --key <zotero_key> --anthropic-from-index --save
   ```
2. Load both extraction results from data/comparison/
3. Perform field-by-field comparison (see Comparison Fields below)
4. Produce a structured quality diff report

## Comparison Fields

For each field, report: match status, provider advantage (if any), and notes.

| Field | Comparison Method |
|-------|-------------------|
| Thesis statement | Semantic similarity; check if both capture the main contribution |
| Research questions | Count and content overlap |
| Methodology | Approach match, data sources, analysis methods |
| Key findings | Count, overlap, evidence type alignment |
| Claims | Count, evidence strength comparison |
| Limitations | Coverage comparison |
| Confidence score | Numeric delta; flag if > 0.15 difference |
| Discipline tags | Set overlap (Jaccard similarity) |
| Keywords | Set overlap |
| Duration / tokens | Cost efficiency comparison |

## Output Format

```
## Provider Comparison: [paper_key]

### Paper
- Title: [title]
- Anthropic model: [model]
- OpenAI model: [model]

### Summary
- Overall winner: [Anthropic/OpenAI/Tie]
- Fields where Anthropic is better: [count]
- Fields where OpenAI is better: [count]
- Confidence delta: [value]

### Field-by-Field

| Field | Anthropic | OpenAI | Winner |
|-------|-----------|--------|--------|
| Thesis | [summary] | [summary] | [A/O/Tie] |
| Findings count | [n] | [n] | [A/O/Tie] |
| Claims count | [n] | [n] | [A/O/Tie] |
| Confidence | [score] | [score] | [A/O/Tie] |
| Duration (s) | [n] | [n] | [A/O/Tie] |
| Input tokens | [n] | [n] | [A/O/Tie] |

### Notable Differences
1. [Field]: [Description of difference and quality implication]

### Recommendations
- [Action items based on comparison results]
```

## Example Prompts

"Compare Anthropic and OpenAI extractions for paper CLCNNCAH and report which provider produced better results."

"Run a provider comparison for Zotero key ABC123 and identify quality gaps."

"Audit the most recent comparison outputs in data/comparison/ and summarize trends."
