---
name: index-health
description: Run a holistic index quality check combining validation, gap analysis, and preflight
disable-model-invocation: true
---

# Index Health Check

Unified index quality report combining extraction validation, gap analysis, and preflight checks.

## Usage

```
/index-health
/index-health --quick
```

## Arguments

- No arguments: Run all three checks (validation, gap analysis, preflight)
- `--quick`: Run validation only (fastest)

## Workflow

Run each diagnostic script and collect results. Report a unified summary.

### Step 1: Extraction Validation

```bash
python scripts/validate_extraction.py
```

Reports: extraction quality scores, schema compliance, low-confidence papers, missing fields.

### Step 2: Gap Analysis

```bash
python scripts/gap_analysis.py
```

Reports: coverage gaps by discipline, underrepresented topics, temporal distribution gaps.

### Step 3: Preflight Analysis

```bash
python scripts/preflight_analysis.py
```

Reports: Zotero database state, papers pending extraction, cost estimates, DOI overlap.

## Output

After running all three scripts, synthesize a unified report:

```
## Index Health Report

### Summary
- Total papers: [n]
- Valid extractions: [n] ([%])
- Low confidence (< 0.7): [n]
- Coverage gaps identified: [n]
- Papers pending extraction: [n]

### Extraction Quality
- [Key findings from validate_extraction.py]

### Coverage Gaps
- [Key findings from gap_analysis.py]

### Pending Work
- [Key findings from preflight_analysis.py]

### Recommendations
1. [Priority action items]
```

## Notes

- Run after index builds to verify quality
- Run before major extractions to check database state
- All scripts are read-only and safe to run at any time
