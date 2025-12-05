# Academic Paper Extraction Skill

## Purpose

Guide the extraction of structured information from academic papers, ensuring accurate capture of thesis, methodology, findings, and contributions.

## When to Use

Invoke this skill when:
- Designing or refining extraction prompts
- Validating extraction quality
- Training on paper structure recognition
- Troubleshooting extraction failures

## Paper Structure Recognition

### Standard Academic Paper Sections

| Section | Location | Contains |
|---------|----------|----------|
| Abstract | Beginning | Summary of entire paper |
| Introduction | After abstract | Problem, motivation, thesis |
| Literature Review | Early sections | Prior work, theoretical framework |
| Methods/Methodology | Middle | Research design, data, analysis |
| Results/Findings | After methods | Primary outcomes |
| Discussion | After results | Interpretation, implications |
| Conclusion | End | Summary, contributions, future work |

### Section Identification Patterns

**Introduction indicators:**
- "This paper examines..."
- "We investigate..."
- "The purpose of this study..."
- "This research addresses..."

**Methods indicators:**
- "We collected data..."
- "Participants were..."
- "The analysis employed..."
- "Our methodology..."

**Results indicators:**
- "The findings show..."
- "Results indicate..."
- "We found that..."
- "Analysis revealed..."

**Conclusion indicators:**
- "In conclusion..."
- "This study demonstrates..."
- "Future research should..."
- "Limitations include..."

## Extraction Field Guidelines

### Thesis Statement

**What it is:** The paper's central argument or hypothesis

**Where to find:**
- End of introduction
- Abstract (often summarized)
- Sometimes explicitly labeled

**Quality criteria:**
- Clear and falsifiable
- Specific to this paper
- Not a general statement about the field

**Example:**
- Good: "This study demonstrates that network centrality predicts citation impact in scientometric analysis"
- Bad: "Citation analysis is important for understanding research impact"

### Research Questions

**What they are:** Explicit questions the paper addresses

**Where to find:**
- End of introduction
- Sometimes numbered/bulleted
- May be labeled "RQ1, RQ2..."

**Extraction rules:**
- Only include explicitly stated questions
- Preserve original wording
- Note if implicit (inferred from objectives)

### Methodology

**Components to extract:**

| Component | Description | Example |
|-----------|-------------|---------|
| Approach | Paradigm | Quantitative, qualitative, mixed |
| Design | Study type | Experiment, case study, survey |
| Data sources | Where data came from | Interviews, databases, documents |
| Sample | Who/what was studied | 500 participants, 10 companies |
| Analysis | How data was processed | Regression, thematic analysis |
| Tools | Software/instruments | SPSS, NVivo, custom scripts |

### Key Findings

**What they are:** Primary results and discoveries

**Quality criteria:**
- Directly addresses research questions
- Supported by evidence in paper
- Represents main contribution, not peripheral

**Extraction format:**
```
Finding: [Statement of finding]
Evidence type: [Statistical, qualitative, theoretical]
Significance: [Why it matters]
```

### Limitations

**What they are:** Acknowledged constraints on the research

**Where to find:**
- Dedicated limitations section
- Discussion section
- Conclusion

**Extraction rules:**
- Only author-stated limitations
- Do not add reviewer-perceived limitations
- Preserve nuance and context

### Future Directions

**What they are:** Suggested areas for future research

**Where to find:**
- End of conclusion
- Discussion section
- Sometimes labeled "Future Work"

**Extraction rules:**
- Explicit suggestions only
- Distinguish from limitations
- Note if actionable vs. vague

## Item Type Variations

### Journal Articles

- Structured, predictable format
- Clear section headings
- Primary source for empirical findings

### Theses/Dissertations

- Extended methodology sections
- Comprehensive literature reviews
- Multiple chapters as sections

### Conference Papers

- Condensed format
- Focus on novel contribution
- May lack detailed methods

### Books/Chapters

- Theoretical frameworks
- Comprehensive reviews
- Less structured

### Preprints

- Preliminary findings
- May lack peer review polish
- Cutting-edge but tentative

## Quality Validation

### Extraction Confidence Scoring

| Confidence | Criteria |
|------------|----------|
| High (0.8+) | All sections present, clear structure |
| Medium (0.5-0.8) | Some sections missing, inference needed |
| Low (<0.5) | Poor structure, significant gaps |

### Common Errors to Avoid

1. **Hallucination**: Inventing content not in paper
2. **Conflation**: Mixing content from different sections
3. **Over-extraction**: Including irrelevant details
4. **Under-extraction**: Missing key contributions
5. **Misattribution**: Assigning findings to wrong section
