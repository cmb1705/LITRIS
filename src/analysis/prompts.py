# DEPRECATED: These templates are replaced by semantic_prompts.py. Retained for reference only.
"""Prompts for LLM-based paper extraction."""

# Prompt version for tracking extraction compatibility
EXTRACTION_PROMPT_VERSION = "1.4.0"  # Added reference_list for citation graph ground truth

EXTRACTION_SYSTEM_PROMPT = """You are an expert academic research analyst specializing in extracting structured information from scholarly papers. Your task is to analyze the provided paper text and extract key information in a structured format.

Guidelines:
1. Extract information directly stated in the text. Do not infer or hallucinate.
2. If information is not clearly present, use null/empty values rather than guessing.
3. Focus on the main arguments and findings, not peripheral details.
4. Maintain academic rigor in your interpretations.
5. When uncertain, note this in the extraction_notes field.
6. Assign confidence scores based on text clarity and completeness.
7. For keywords, extract specific terms that would help researchers find this paper.
8. Assess the overall quality of the paper on a 1-5 scale considering: methodology rigor, evidence strength, contribution significance, and writing clarity. This rates the PAPER's quality, not the extraction quality."""

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
  "reference_list": [
    {{
      "raw_text": "The full bibliography entry as it appears in the reference list",
      "parsed_title": "Title of the referenced work",
      "parsed_authors": "Author(s) of the referenced work",
      "parsed_year": 2020,
      "parsed_doi": "10.xxxx/xxxxx or null if not present"
    }}
  ],
  "quality_rating": "1-5 integer (1=poor methodology/weak evidence, 2=below average, 3=competent but unremarkable, 4=strong methodology and evidence, 5=exceptional rigor and contribution)",
  "quality_explanation": "Brief rationale for the quality rating",
  "extraction_confidence": 0.0-1.0,
  "extraction_notes": "Notes about extraction quality, missing sections, or ambiguous content"
}}

IMPORTANT: For reference_list, extract up to 50 entries from the paper's bibliography/references section. Include the DOI if present in the entry. If the paper has no reference list, return an empty array.

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
  "quality_rating": "1-5 integer (1=poor methodology/weak evidence, 2=below average, 3=competent but unremarkable, 4=strong methodology and evidence, 5=exceptional rigor and contribution)",
  "quality_explanation": "Brief rationale for the quality rating",
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
  "quality_rating": "1-5 integer (1=poor methodology/weak evidence, 2=below average, 3=competent but unremarkable, 4=strong methodology and evidence, 5=exceptional rigor and contribution)",
  "quality_explanation": "Brief rationale for the quality rating",
  "extraction_confidence": 0.0-1.0,
  "extraction_notes": "Notes about extraction quality, missing sections, or ambiguous content"
}}

Respond ONLY with valid JSON. No additional text or markdown formatting.'''

BOOK_EXTRACTION_USER_PROMPT = '''Analyze the following book/monograph and extract structured information.
Focus on the central argument, key concepts, and theoretical contribution rather than
empirical methodology.

DOCUMENT METADATA:
Title: {title}
Authors: {authors}
Year: {year}
Type: {item_type}

DOCUMENT TEXT:
{text}

Extract the following information and return as JSON:

Enum rules (use exact tokens only, no extra words or parentheses):
- support_type: data/citation/logic/example/authority

{{
  "thesis_statement": "The central argument or thesis of the book (1-2 sentences)",
  "theoretical_framework": "Primary theoretical lens or paradigm (null if not explicit)",
  "key_claims": [
    {{
      "claim": "A major argument or claim made in the book",
      "support_type": "data/citation/logic/example/authority",
      "page_reference": "page or chapter reference if identifiable (null otherwise)"
    }}
  ],
  "conclusions": "Main conclusions or takeaways (2-3 sentences)",
  "contribution_summary": "The book's primary contribution to the field (1-2 sentences)",
  "keywords": ["5-10 searchable terms: concepts, theories, phenomena covered"],
  "discipline_tags": ["2-5 academic disciplines. Use lowercase."],
  "quality_rating": "1-5 integer (1=poor methodology/weak evidence, 2=below average, 3=competent but unremarkable, 4=strong methodology and evidence, 5=exceptional rigor and contribution)",
  "quality_explanation": "Brief rationale for the quality rating",
  "extraction_confidence": 0.0-1.0,
  "extraction_notes": "Notes about extraction quality or missing content"
}}

Respond ONLY with valid JSON. No additional text or markdown formatting.'''

REPORT_EXTRACTION_USER_PROMPT = '''Analyze the following report/white paper and extract structured information.
Focus on findings, recommendations, and data sources rather than academic methodology.

DOCUMENT METADATA:
Title: {title}
Authors: {authors}
Year: {year}
Type: {item_type}

DOCUMENT TEXT:
{text}

Extract the following information and return as JSON:

Enum rules (use exact tokens only, no extra words or parentheses):
- significance: high/medium/low
- evidence_type: empirical/theoretical/methodological/case_study/survey/experimental/qualitative/quantitative/mixed
- support_type: data/citation/logic/example/authority

{{
  "key_findings": [
    {{
      "finding": "A key finding or recommendation from the report",
      "evidence_type": "empirical/theoretical/methodological/case_study/survey/experimental/qualitative/quantitative/mixed",
      "significance": "high/medium/low",
      "page_reference": "page reference if identifiable (null otherwise)"
    }}
  ],
  "key_claims": [
    {{
      "claim": "A key claim or recommendation",
      "support_type": "data/citation/logic/example/authority",
      "page_reference": "page reference if identifiable (null otherwise)"
    }}
  ],
  "conclusions": "Main conclusions or recommendations (2-3 sentences)",
  "keywords": ["5-10 searchable terms: topics, methods, policy areas"],
  "discipline_tags": ["2-5 relevant fields. Use lowercase."],
  "quality_rating": "1-5 integer (1=poor methodology/weak evidence, 2=below average, 3=competent but unremarkable, 4=strong methodology and evidence, 5=exceptional rigor and contribution)",
  "quality_explanation": "Brief rationale for the quality rating",
  "extraction_confidence": 0.0-1.0,
  "extraction_notes": "Notes about extraction quality or missing content"
}}

Respond ONLY with valid JSON. No additional text or markdown formatting.'''

REVIEW_EXTRACTION_USER_PROMPT = '''Analyze the following review article and extract structured information.
Focus on the synthesis scope, included studies, synthesized findings, and identified gaps.
Treat "methodology" as the review scope and search strategy, not empirical methods.

DOCUMENT METADATA:
Title: {title}
Authors: {authors}
Year: {year}
Type: {item_type}

DOCUMENT TEXT:
{text}

Extract the following information and return as JSON:

Enum rules (use exact tokens only, no extra words or parentheses):
- significance: high/medium/low
- evidence_type: empirical/theoretical/methodological/case_study/survey/experimental/qualitative/quantitative/mixed

{{
  "thesis_statement": "The main objective or scope of this review (1-2 sentences)",
  "research_questions": ["Review questions or objectives addressed"],
  "methodology": {{
    "approach": "review",
    "design": "e.g., systematic review, meta-analysis, scoping review, narrative review",
    "data_sources": ["databases searched, inclusion criteria"],
    "analysis_methods": ["synthesis approach: thematic, statistical, narrative, etc."],
    "sample_size": "number of studies included if stated (null otherwise)",
    "time_period": "time period covered by the review (null otherwise)"
  }},
  "key_findings": [
    {{
      "finding": "A synthesized finding from the reviewed literature",
      "evidence_type": "empirical/theoretical/methodological/case_study/survey/experimental/qualitative/quantitative/mixed",
      "significance": "high/medium/low",
      "page_reference": "page reference if identifiable (null otherwise)"
    }}
  ],
  "conclusions": "Main conclusions and identified gaps (2-3 sentences)",
  "future_directions": ["Gaps or directions identified by the review"],
  "contribution_summary": "How this review contributes to the field (1-2 sentences)",
  "keywords": ["5-10 searchable terms: review topic, methods, phenomena"],
  "discipline_tags": ["2-5 academic disciplines. Use lowercase."],
  "quality_rating": "1-5 integer (1=poor methodology/weak evidence, 2=below average, 3=competent but unremarkable, 4=strong methodology and evidence, 5=exceptional rigor and contribution)",
  "quality_explanation": "Brief rationale for the quality rating",
  "extraction_confidence": 0.0-1.0,
  "extraction_notes": "Notes about extraction quality or missing content"
}}

Respond ONLY with valid JSON. No additional text or markdown formatting.'''

GENERIC_EXTRACTION_USER_PROMPT = '''Analyze the following document and extract whatever structured information is available.
This document may not follow standard academic paper structure.

DOCUMENT METADATA:
Title: {title}
Authors: {authors}
Year: {year}
Type: {item_type}

DOCUMENT TEXT:
{text}

Extract the following information and return as JSON:

{{
  "thesis_statement": "The main point or purpose of this document (null if unclear)",
  "contribution_summary": "Brief summary of what this document covers (1-2 sentences)",
  "keywords": ["3-8 searchable terms related to the content"],
  "discipline_tags": ["1-3 relevant fields if applicable. Use lowercase."],
  "quality_rating": "1-5 integer (1=poor methodology/weak evidence, 2=below average, 3=competent but unremarkable, 4=strong methodology and evidence, 5=exceptional rigor and contribution)",
  "quality_explanation": "Brief rationale for the quality rating",
  "extraction_confidence": 0.0-1.0,
  "extraction_notes": "Notes about the document type and extraction limitations"
}}

Respond ONLY with valid JSON. No additional text or markdown formatting.'''

# Mapping from document-type prompt keys to templates
DOCUMENT_TYPE_PROMPTS: dict[str, str] = {
    "full": EXTRACTION_USER_PROMPT,
    "book": BOOK_EXTRACTION_USER_PROMPT,
    "report": REPORT_EXTRACTION_USER_PROMPT,
    "review": REVIEW_EXTRACTION_USER_PROMPT,
    "generic": GENERIC_EXTRACTION_USER_PROMPT,
}

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


def build_prompt_for_document_type(
    prompt_key: str,
    title: str,
    authors: str,
    year: int | str | None,
    item_type: str,
    text: str,
) -> str:
    """Build an extraction prompt for a specific document type.

    Args:
        prompt_key: Key into DOCUMENT_TYPE_PROMPTS (full, book, report, review, generic).
        title: Paper/document title.
        authors: Author string.
        year: Publication year.
        item_type: Zotero item type.
        text: Full text content.

    Returns:
        Formatted prompt string.

    Raises:
        ValueError: If prompt_key is not recognized.
    """
    template = DOCUMENT_TYPE_PROMPTS.get(prompt_key)
    if template is None:
        raise ValueError(
            f"Unknown prompt_key '{prompt_key}'. "
            f"Valid keys: {', '.join(DOCUMENT_TYPE_PROMPTS.keys())}."
        )

    return template.format(
        title=title,
        authors=authors,
        year=year or "Unknown",
        item_type=item_type,
        text=text,
    )


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
