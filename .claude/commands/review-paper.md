---
description: Review extraction quality for a specific paper
allowed-tools: Read, Grep, Glob
argument-hint: <paper_id or title>
---

Review the extraction quality for a specific paper.

Paper identifier: $ARGUMENTS

1. Find the paper in papers.json (by ID or title match)
2. Load the corresponding extraction from extractions.json
3. If available, read the original PDF
4. Compare extraction to source and assess:
   - Thesis statement accuracy
   - Methodology completeness
   - Key findings coverage
   - Limitations and future directions
5. Provide quality score and recommendations

Report format:
## Paper: [Title]
### Extraction Quality Assessment
- Thesis: [Accurate/Partially Accurate/Inaccurate]
- Methodology: [Complete/Partial/Missing]
- Findings: [Complete/Partial/Missing]
- Overall confidence: [High/Medium/Low]

### Issues Found
[List any problems]

### Recommendations
[Suggestions for improvement]
