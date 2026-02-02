"""Prompts for LLM-based paper extraction."""

# Prompt version for tracking extraction compatibility
EXTRACTION_PROMPT_VERSION = "1.2.0"  # Added discipline tag examples and lowercase guidance

EXTRACTION_SYSTEM_PROMPT = """You are an expert academic research analyst specializing in extracting structured information from scholarly papers. Your task is to analyze the provided paper text and extract key information in a structured format.

Guidelines:
1. Extract information directly stated in the text. Do not infer or hallucinate.
2. If information is not clearly present, use null/empty values rather than guessing.
3. Focus on the main arguments and findings, not peripheral details.
4. Maintain academic rigor in your interpretations.
5. When uncertain, note this in the extraction_notes field.
6. Assign confidence scores based on text clarity and completeness.
7. For keywords, extract specific terms that would help researchers find this paper."""

EXTRACTION_USER_PROMPT = '''Analyze the following academic paper and extract structured information.

PAPER METADATA:
Title: {title}
Authors: {authors}
Year: {year}
Type: {item_type}

PAPER TEXT:
{text}

Extract the following information and return as JSON:

Enum rules (use exact tokens only, no extra words or parentheses):
- significance: high/medium/low
- evidence_type: empirical/theoretical/methodological/case_study/survey/experimental/qualitative/quantitative/mixed
- support_type: data/citation/logic/example/authority

{{
  "thesis_statement": "The main thesis or central argument (1-2 sentences)",
  "research_questions": ["List of explicit research questions or objectives addressed"],
  "theoretical_framework": "Primary theoretical lens, framework, or paradigm used (null if not explicitly stated)",
  "methodology": {{
    "approach": "qualitative/quantitative/mixed/theoretical/review",
    "design": "e.g., case study, experiment, survey, ethnography, systematic review, meta-analysis",
    "data_sources": ["list of specific data sources used"],
    "analysis_methods": ["specific analytical methods or techniques"],
    "sample_size": "sample size or N if applicable (null otherwise)",
    "time_period": "time period covered if applicable (null otherwise)"
  }},
  "key_findings": [
    {{
      "finding": "Specific description of finding or result",
      "evidence_type": "empirical/theoretical/methodological/case_study/survey/experimental/qualitative/quantitative/mixed",
      "significance": "high/medium/low",
      "page_reference": "page number if identifiable (null otherwise)"
    }}
  ],
  "key_claims": [
    {{
      "claim": "The claim or argument statement",
      "support_type": "data/citation/logic/example/authority",
      "page_reference": "page number if identifiable (null otherwise)"
    }}
  ],
  "conclusions": "Main conclusions summarized (2-3 sentences)",
  "limitations": ["List of explicitly acknowledged limitations"],
  "future_directions": ["Explicitly suggested future research directions"],
  "contribution_summary": "Brief summary of the paper's primary contribution to the field (1-2 sentences)",
  "keywords": ["5-10 searchable terms: concepts, methods, theories, phenomena studied"],
  "discipline_tags": ["2-5 academic disciplines this paper contributes to. Use lowercase. Examples: 'scientometrics', 'bibliometrics', 'network science', 'science policy', 'information science', 'machine learning', 'complex systems', 'science and technology studies', 'research evaluation', 'computational social science', 'innovation studies', 'public administration', 'military studies', 'defense policy', 'organizational behavior', 'philosophy of science', 'history of science', 'sociology of science', 'economics of innovation', 'technology assessment', 'data science', 'artificial intelligence', 'epidemiology', 'statistical physics', 'graph theory'],
  "extraction_confidence": 0.0-1.0,
  "extraction_notes": "Notes about extraction quality, missing sections, or ambiguous content"
}}

Respond ONLY with valid JSON. No additional text or markdown formatting.'''

SUMMARY_EXTRACTION_USER_PROMPT = '''Analyze the following academic paper and extract structured information.

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
  "research_questions": ["List of explicit research questions or objectives addressed"],
  "theoretical_framework": "Primary theoretical lens, framework, or paradigm used (null if not explicitly stated)",
  "conclusions": "Main conclusions summarized (2-3 sentences)",
  "contribution_summary": "Brief summary of the paper's primary contribution to the field (1-2 sentences)",
  "keywords": ["5-10 searchable terms: concepts, methods, theories, phenomena studied"],
  "discipline_tags": ["2-5 academic disciplines this paper contributes to. Use lowercase. Examples: 'scientometrics', 'bibliometrics', 'network science', 'science policy', 'information science', 'machine learning', 'complex systems', 'science and technology studies', 'research evaluation', 'computational social science', 'innovation studies', 'public administration', 'military studies', 'defense policy', 'organizational behavior', 'philosophy of science', 'history of science', 'sociology of science', 'economics of innovation', 'technology assessment', 'data science', 'artificial intelligence', 'epidemiology', 'statistical physics', 'graph theory'],
  "extraction_confidence": 0.0-1.0,
  "extraction_notes": "Notes about extraction quality, missing sections, or ambiguous content"
}}

Respond ONLY with valid JSON. No additional text or markdown formatting.'''

METHODOLOGY_EXTRACTION_USER_PROMPT = '''Analyze the following academic paper and extract structured information.

PAPER METADATA:
Title: {title}
Authors: {authors}
Year: {year}
Type: {item_type}

PAPER TEXT:
{text}

Extract the following information and return as JSON:

Enum rules (use exact tokens only, no extra words or parentheses):
- significance: high/medium/low
- evidence_type: empirical/theoretical/methodological/case_study/survey/experimental/qualitative/quantitative/mixed
- support_type: data/citation/logic/example/authority

{{
  "methodology": {{
    "approach": "qualitative/quantitative/mixed/theoretical/review",
    "design": "e.g., case study, experiment, survey, ethnography, systematic review, meta-analysis",
    "data_sources": ["list of specific data sources used"],
    "analysis_methods": ["specific analytical methods or techniques"],
    "sample_size": "sample size or N if applicable (null otherwise)",
    "time_period": "time period covered if applicable (null otherwise)"
  }},
  "key_findings": [
    {{
      "finding": "Specific description of finding or result",
      "evidence_type": "empirical/theoretical/methodological/case_study/survey/experimental/qualitative/quantitative/mixed",
      "significance": "high/medium/low",
      "page_reference": "page number if identifiable (null otherwise)"
    }}
  ],
  "key_claims": [
    {{
      "claim": "The claim or argument statement",
      "support_type": "data/citation/logic/example/authority",
      "page_reference": "page number if identifiable (null otherwise)"
    }}
  ],
  "limitations": ["List of explicitly acknowledged limitations"],
  "future_directions": ["Explicitly suggested future research directions"],
  "extraction_confidence": 0.0-1.0,
  "extraction_notes": "Notes about extraction quality, missing sections, or ambiguous content"
}}

Respond ONLY with valid JSON. No additional text or markdown formatting.'''

EXTRACTION_TYPE_FIELDS: dict[str, list[str]] = {
    "summary": [
        "thesis_statement",
        "research_questions",
        "theoretical_framework",
        "conclusions",
        "contribution_summary",
        "keywords",
        "discipline_tags",
    ],
    "methodology": [
        "methodology",
        "key_findings",
        "key_claims",
        "limitations",
        "future_directions",
    ],
}


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


def build_cli_extraction_prompt(
    title: str,
    authors: str,
    year: int | str | None,
    item_type: str,
) -> str:
    """Build the extraction prompt for CLI mode (text provided separately via stdin).

    For CLI mode, the paper text is passed via stdin to handle large documents,
    so this prompt only includes the metadata and instructions.

    Args:
        title: Paper title.
        authors: Author string.
        year: Publication year.
        item_type: Type of paper.

    Returns:
        Formatted prompt string (text placeholder indicates stdin input).
    """
    return EXTRACTION_USER_PROMPT.format(
        title=title,
        authors=authors,
        year=year or "Unknown",
        item_type=item_type,
        text="[PAPER TEXT PROVIDED VIA STDIN - SEE BELOW]",
    )


def build_extraction_prompt_for_type(
    extraction_type: str,
    title: str,
    authors: str,
    year: int | str | None,
    item_type: str,
    text: str,
) -> str:
    """Build an extraction prompt for a specific extraction type.

    Args:
        extraction_type: Extraction type ("summary", "methodology", or "full").
        title: Paper title.
        authors: Author string.
        year: Publication year.
        item_type: Type of paper.
        text: Full text content.

    Returns:
        Formatted prompt string.
    """
    if extraction_type == "summary":
        template = SUMMARY_EXTRACTION_USER_PROMPT
    elif extraction_type == "methodology":
        template = METHODOLOGY_EXTRACTION_USER_PROMPT
    elif extraction_type == "full":
        template = EXTRACTION_USER_PROMPT
    else:
        raise ValueError(
            f"Unknown extraction_type '{extraction_type}'. "
            f"Valid types: {', '.join(EXTRACTION_TYPE_FIELDS.keys())}."
        )

    return template.format(
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
