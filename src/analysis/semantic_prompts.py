"""6-pass prompt templates for SemanticAnalysis extraction.

Each pass targets a thematic group of dimensions (q01-q40). Passes run
sequentially per paper so the model maintains consistent reasoning context.

Prompt version is tracked separately from the legacy prompts.py version.
Cache keys include this version so bumping it invalidates all cached results.
"""

from src.analysis.dimensions import get_default_dimension_registry

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
# Each entry is ``(output_key, question_text)``. Legacy profiles keep ``qNN``
# output keys; newer profiles can emit canonical IDs directly.


def get_pass_definitions() -> list[tuple[str, list[tuple[str, str]]]]:
    """Return the ordered extraction passes for the active profile."""

    registry = get_default_dimension_registry()
    profile = registry.active_profile
    pass_definitions: list[tuple[str, list[tuple[str, str]]]] = []
    for section in profile.ordered_sections:
        questions = []
        for dimension in profile.dimensions_for_section(section.id):
            output_key = dimension.legacy_field_name or dimension.id
            questions.append((output_key, dimension.question))
        pass_definitions.append((section.display_label, questions))
    return pass_definitions


def get_dimension_groups() -> dict[str, list[str]]:
    """Return group -> output key mappings for the active profile."""

    registry = get_default_dimension_registry()
    profile = registry.active_profile
    groups: dict[str, list[str]] = {}
    for section in profile.ordered_sections:
        groups[section.id] = [
            dimension.legacy_field_name or dimension.id
            for dimension in profile.dimensions_for_section(section.id)
        ]
    return groups


def get_field_to_group() -> dict[str, str]:
    """Return output key -> group mapping for the active profile."""

    return {
        field_name: group_name
        for group_name, fields in get_dimension_groups().items()
        for field_name in fields
    }


def get_dimension_reasoning_effort() -> dict[str, str]:
    """Return output key -> reasoning effort for the active profile."""

    registry = get_default_dimension_registry()
    profile = registry.active_profile
    return {
        (dimension.legacy_field_name or dimension.id): dimension.reasoning_effort
        for dimension in profile.ordered_dimensions
    }


PASS_DEFINITIONS: list[tuple[str, list[tuple[str, str]]]] = get_pass_definitions()
DIMENSION_GROUPS: dict[str, list[str]] = get_dimension_groups()
FIELD_TO_GROUP: dict[str, str] = get_field_to_group()
DIMENSION_REASONING_EFFORT: dict[str, str] = get_dimension_reasoning_effort()


def get_pass_reasoning_effort(pass_number: int) -> str:
    """Return the highest reasoning effort needed for any dimension in a pass.

    If any dimension in the pass requires "xhigh", the whole pass runs at
    "xhigh". Otherwise "high".

    Args:
        pass_number: 1-indexed pass number for the active profile.

    Returns:
        "high" or "xhigh".
    """
    pass_definitions = get_pass_definitions()
    _, questions = pass_definitions[pass_number - 1]
    for field_name, _ in questions:
        if DIMENSION_REASONING_EFFORT.get(field_name) == "xhigh":
            return "xhigh"
    return "high"


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
    pass_definitions = get_pass_definitions()
    if not 1 <= pass_number <= len(pass_definitions):
        raise ValueError(
            f"pass_number must be 1-{len(pass_definitions)}, got {pass_number}"
        )

    pass_label, questions = pass_definitions[pass_number - 1]
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
        pass_number: 1-indexed pass number for the active profile.

    Returns:
        List of field name strings (e.g., ["q01_research_question", ...]).
    """
    pass_definitions = get_pass_definitions()
    if not 1 <= pass_number <= len(pass_definitions):
        raise ValueError(
            f"pass_number must be 1-{len(pass_definitions)}, got {pass_number}"
        )
    return [q[0] for q in pass_definitions[pass_number - 1][1]]


def get_all_dimension_fields() -> list[str]:
    """Return all output keys for the active profile in order."""
    fields = []
    for _, questions in get_pass_definitions():
        fields.extend(q[0] for q in questions)
    return fields
