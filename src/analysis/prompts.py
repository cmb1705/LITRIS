"""Prompts for LLM-based paper extraction."""

EXTRACTION_SYSTEM_PROMPT = """You are an expert academic research analyst specializing in extracting structured information from scholarly papers. Your task is to analyze the provided paper text and extract key information in a structured format.

Guidelines:
1. Extract information directly stated in the text. Do not infer or hallucinate.
2. If information is not clearly present, use null/empty values rather than guessing.
3. Focus on the main arguments and findings, not peripheral details.
4. Maintain academic rigor in your interpretations.
5. When uncertain, note this in the extraction_notes field.
6. Assign confidence scores based on text clarity and completeness."""

EXTRACTION_USER_PROMPT = '''Analyze the following academic paper and extract structured information.

PAPER METADATA:
Title: {title}
Authors: {authors}
Year: {year}
Type: {item_type}

PAPER TEXT:
{text}

Extract the following information and return as JSON:

{{
  "thesis_statement": "The main thesis or central argument (1-2 sentences)",
  "research_questions": ["List of research questions addressed"],
  "theoretical_framework": "Theoretical lens or framework used",
  "methodology": {{
    "approach": "qualitative/quantitative/mixed",
    "design": "research design type",
    "data_sources": ["list of data sources"],
    "analysis_methods": ["methods used"],
    "sample_size": "sample size if applicable",
    "time_period": "time period covered if applicable"
  }},
  "key_findings": [
    {{
      "finding": "Description of finding",
      "evidence_type": "empirical/theoretical/methodological/case_study/survey/experimental/qualitative/quantitative/mixed",
      "significance": "high/medium/low",
      "page_reference": "page number if identifiable"
    }}
  ],
  "key_claims": [
    {{
      "claim": "The claim statement",
      "support_type": "data/citation/logic/example/authority",
      "page_reference": "page number if identifiable",
      "strength": "high/medium/low"
    }}
  ],
  "conclusions": "Main conclusions (2-3 sentences)",
  "limitations": ["List of acknowledged limitations"],
  "future_directions": ["Suggested future research directions"],
  "contribution_summary": "Brief summary of the paper's contribution (1-2 sentences)",
  "discipline_tags": ["relevant topic/discipline tags"],
  "extraction_confidence": 0.0-1.0,
  "extraction_notes": "Any notes about extraction quality or issues"
}}

Respond ONLY with valid JSON. No additional text or markdown formatting.'''


def build_extraction_prompt(
    title: str,
    authors: str,
    year: int | str | None,
    item_type: str,
    text: str,
) -> str:
    """Build the extraction prompt with paper details.

    Args:
        title: Paper title.
        authors: Author string.
        year: Publication year.
        item_type: Type of paper.
        text: Full text content.

    Returns:
        Formatted prompt string.
    """
    return EXTRACTION_USER_PROMPT.format(
        title=title,
        authors=authors,
        year=year or "Unknown",
        item_type=item_type,
        text=text,
    )


VALIDATION_PROMPT = """Review the following extraction for accuracy and completeness.

ORIGINAL TEXT EXCERPT:
{text_excerpt}

EXTRACTION:
{extraction_json}

Identify any:
1. Hallucinated information (not in source)
2. Missing key information
3. Misinterpretations
4. Confidence score accuracy

Respond with JSON:
{{
  "valid": true/false,
  "issues": ["list of issues found"],
  "suggested_corrections": {{}}
}}"""


def build_validation_prompt(text_excerpt: str, extraction_json: str) -> str:
    """Build validation prompt.

    Args:
        text_excerpt: Excerpt from original text.
        extraction_json: JSON string of extraction.

    Returns:
        Formatted validation prompt.
    """
    return VALIDATION_PROMPT.format(
        text_excerpt=text_excerpt,
        extraction_json=extraction_json,
    )
