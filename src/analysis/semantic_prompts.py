"""6-pass prompt templates for SemanticAnalysis extraction.

Each pass targets a thematic group of dimensions (q01-q40). Passes run
sequentially per paper so the model maintains consistent reasoning context.

Prompt version is tracked separately from the legacy prompts.py version.
Cache keys include this version so bumping it invalidates all cached results.
"""

# Prompt version for SemanticAnalysis extraction.
# Bump this when any prompt text changes; cache keys include it.
SEMANTIC_PROMPT_VERSION = "2.0.0"

# -- System prompt (shared across all 6 passes) --------------------------------

SEMANTIC_SYSTEM_PROMPT = """\
You are an expert academic research analyst conducting a structured semantic \
analysis of a scholarly document. Your task is to answer a specific set of \
analytical questions about the document.

Guidelines:
1. Provide thorough prose responses (2-5 sentences per dimension).
2. Extract information directly stated in or clearly supported by the text.
3. Return null for any dimension that genuinely does not apply to this document.
4. Return ONLY valid JSON with the requested keys. No markdown, no commentary.
5. Use extended thinking to reason carefully before answering."""

# -- Document-type framing notes -----------------------------------------------
# Inserted into user prompts so the model calibrates expectations.

DOCUMENT_TYPE_FRAMING: dict[str, str] = {
    "research_paper": (
        "This is primary empirical or theoretical research. "
        "Most dimensions should be answerable."
    ),
    "book": (
        "This is a book-length scholarly work. "
        "Focus on central arguments, theoretical contributions, and scope. "
        "Some empirical dimensions may return null."
    ),
    "monograph": (
        "This is a book-length scholarly work. "
        "Focus on central arguments, theoretical contributions, and scope. "
        "Some empirical dimensions may return null."
    ),
    "review": (
        "This is a systematic review or meta-analysis. "
        "Treat methodology dimensions as describing the review protocol, "
        "not individual study methods."
    ),
    "meta-analysis": (
        "This is a systematic review or meta-analysis. "
        "Treat methodology dimensions as describing the review protocol, "
        "not individual study methods."
    ),
    "report": (
        "This is a policy or technical report. "
        "Focus on findings, recommendations, and data sources. "
        "Academic methodology dimensions may return null."
    ),
    "white_paper": (
        "This is a policy or technical report. "
        "Focus on findings, recommendations, and data sources. "
        "Academic methodology dimensions may return null."
    ),
    "thesis": (
        "This is a graduate research thesis or dissertation. "
        "Expect detailed methodology and literature review sections."
    ),
    "dissertation": (
        "This is a graduate research thesis or dissertation. "
        "Expect detailed methodology and literature review sections."
    ),
    "generic": (
        "This is a non-traditional or unclassified document. "
        "Many dimensions may return null; answer what applies."
    ),
    "non-academic": (
        "This is a non-traditional or unclassified document. "
        "Many dimensions may return null; answer what applies."
    ),
}

# Fallback when document type is not in the mapping.
_DEFAULT_FRAMING = (
    "Analyze this document to the extent each dimension applies. "
    "Return null for inapplicable dimensions."
)


def _get_framing(document_type: str) -> str:
    """Return the framing note for a document type, with fallback."""
    return DOCUMENT_TYPE_FRAMING.get(document_type, _DEFAULT_FRAMING)


# -- Question definitions per pass ----------------------------------------------
# Each entry: (field_name, human-readable question text)

PASS_1_QUESTIONS: list[tuple[str, str]] = [
    ("q01_research_question",
     "What research questions or objectives does this work address?"),
    ("q02_thesis",
     "What is the central thesis or main argument?"),
    ("q03_key_claims",
     "What are the key claims or propositions made?"),
    ("q04_evidence",
     "What evidence is presented and how strong is it?"),
    ("q05_limitations",
     "What limitations are acknowledged or apparent?"),
]

PASS_2_QUESTIONS: list[tuple[str, str]] = [
    ("q06_paradigm",
     "What research paradigm underlies this work? "
     "(positivist, interpretivist, critical, pragmatist, etc.)"),
    ("q07_methods",
     "What methods and analytical techniques are used?"),
    ("q08_data",
     "What data sources, sample sizes, and time periods are involved?"),
    ("q09_reproducibility",
     "How reproducible is this work? Are methods, data, and code available?"),
    ("q10_framework",
     "What theoretical or conceptual framework is used?"),
]

PASS_3_QUESTIONS: list[tuple[str, str]] = [
    ("q11_traditions",
     "What intellectual traditions or schools of thought does this draw from?"),
    ("q12_key_citations",
     "What are the most influential works cited, and how do they shape "
     "this paper?"),
    ("q13_assumptions",
     "What assumptions (stated or unstated) underlie the analysis?"),
    ("q14_counterarguments",
     "What counterarguments or alternative interpretations are addressed?"),
    ("q15_novelty",
     "What is novel or original about this work?"),
    ("q16_stance",
     "What is the author's stance or perspective on the topic?"),
]

PASS_4_QUESTIONS: list[tuple[str, str]] = [
    ("q17_field",
     "What academic field(s) does this work belong to?"),
    ("q18_audience",
     "Who is the intended audience?"),
    ("q19_implications",
     "What are the broader theoretical or practical implications?"),
    ("q20_future_work",
     "What future research directions are suggested?"),
    ("q21_quality",
     "How would you rate the overall quality? "
     "(methodology rigor, evidence strength, contribution significance)"),
    ("q22_contribution",
     "What is the explicit contribution of this work to its field?"),
    ("q23_source_type",
     "What type of document is this? "
     "(empirical study, review, theoretical, report, etc.)"),
    ("q24_other",
     "What else is noteworthy that the above questions don't capture?"),
]

PASS_5_QUESTIONS: list[tuple[str, str]] = [
    ("q25_institutional_context",
     "What institutional or organizational context shaped this work?"),
    ("q26_historical_timing",
     "Why does this work appear now? "
     "What historical/temporal factors are relevant?"),
    ("q27_paradigm_influence",
     "How does this work relate to dominant paradigms in its field?"),
    ("q28_disciplines_bridged",
     "What disciplines does this work bridge or draw from?"),
    ("q29_cross_domain_insights",
     "What insights transfer to or from other domains?"),
    ("q30_cultural_scope",
     "What cultural, geographic, or demographic scope does this cover?"),
    ("q31_philosophical_assumptions",
     "What philosophical assumptions underlie the methodology or claims?"),
]

PASS_6_QUESTIONS: list[tuple[str, str]] = [
    ("q32_deployment_gap",
     "What gap exists between this research and real-world application?"),
    ("q33_infrastructure_contribution",
     "Does this work contribute tools, datasets, frameworks, "
     "or infrastructure?"),
    ("q34_power_dynamics",
     "What power dynamics, inequities, or stakeholder tensions "
     "are relevant?"),
    ("q35_gaps_and_omissions",
     "What important aspects does this work fail to address?"),
    ("q36_dual_use_concerns",
     "Are there dual-use or ethical concerns with the findings or methods?"),
    ("q37_emergence_claims",
     "Does this work describe emergent phenomena or system-level behaviors?"),
    ("q38_remaining_other",
     "What else is significant that no prior question has captured?"),
    ("q39_network_properties",
     "What network structures, metrics, or graph algorithms are central?"),
    ("q40_policy_recommendations",
     "What specific policy recommendations or actionable guidance "
     "is proposed?"),
]

# Ordered list of all passes for iteration.
PASS_DEFINITIONS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Pass 1: Research Core", PASS_1_QUESTIONS),
    ("Pass 2: Methodology", PASS_2_QUESTIONS),
    ("Pass 3: Context & Discourse", PASS_3_QUESTIONS),
    ("Pass 4: Meta & Audience", PASS_4_QUESTIONS),
    ("Pass 5: Scholarly Positioning", PASS_5_QUESTIONS),
    ("Pass 6: Impact, Gaps & Domain", PASS_6_QUESTIONS),
]

# -- Dimension groups (for search filtering) ------------------------------------

DIMENSION_GROUPS: dict[str, list[str]] = {
    "research_core": [q[0] for q in PASS_1_QUESTIONS],
    "methodology": [q[0] for q in PASS_2_QUESTIONS],
    "context": [q[0] for q in PASS_3_QUESTIONS],
    "meta": [q[0] for q in PASS_4_QUESTIONS],
    "scholarly": [q[0] for q in PASS_5_QUESTIONS],
    "impact": [q[0] for q in PASS_6_QUESTIONS],
}

# Reverse mapping: field name -> group name
FIELD_TO_GROUP: dict[str, str] = {
    field: group
    for group, fields in DIMENSION_GROUPS.items()
    for field in fields
}


# -- Prompt builders -----------------------------------------------------------

def _format_questions(questions: list[tuple[str, str]]) -> str:
    """Format question definitions for inclusion in a user prompt."""
    lines = []
    for field_name, question_text in questions:
        lines.append(f'  "{field_name}": "<{question_text}>"')
    return ",\n".join(lines)


def build_pass_user_prompt(
    pass_number: int,
    title: str,
    authors: str,
    year: int | str | None,
    document_type: str,
    text: str,
) -> str:
    """Build the user prompt for a specific extraction pass.

    Args:
        pass_number: 1-6, selecting which question group to use.
        title: Paper title.
        authors: Author string.
        year: Publication year.
        document_type: Document type key for framing note.
        text: Truncated paper text.

    Returns:
        Formatted user prompt string.

    Raises:
        ValueError: If pass_number is not 1-6.
    """
    if not 1 <= pass_number <= 6:
        raise ValueError(f"pass_number must be 1-6, got {pass_number}")

    pass_label, questions = PASS_DEFINITIONS[pass_number - 1]
    framing = _get_framing(document_type)
    formatted_questions = _format_questions(questions)

    return f"""\
Analyze the following document and answer the questions below.

DOCUMENT TYPE NOTE: {framing}

PAPER METADATA:
Title: {title}
Authors: {authors}
Year: {year or "Unknown"}

PAPER TEXT:
{text}

{pass_label} -- Answer each question with 2-5 sentences of prose analysis. \
Return null for dimensions that do not apply to this document.

Return ONLY a JSON object with these keys:

{{
{formatted_questions}
}}"""


def get_pass_fields(pass_number: int) -> list[str]:
    """Return the field names produced by a given pass.

    Args:
        pass_number: 1-6.

    Returns:
        List of field name strings (e.g., ["q01_research_question", ...]).
    """
    if not 1 <= pass_number <= 6:
        raise ValueError(f"pass_number must be 1-6, got {pass_number}")
    return [q[0] for q in PASS_DEFINITIONS[pass_number - 1][1]]


def get_all_dimension_fields() -> list[str]:
    """Return all 40 dimension field names in order."""
    fields = []
    for _, questions in PASS_DEFINITIONS:
        fields.extend(q[0] for q in questions)
    return fields
