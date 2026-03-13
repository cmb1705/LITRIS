"""Document type taxonomy and type-specific profiles for extraction and validation."""

from dataclasses import dataclass
from enum import Enum


class DocumentType(str, Enum):
    """Classification of document types in a Zotero library.

    Each type maps to different extraction and validation expectations.
    """

    RESEARCH_PAPER = "research_paper"
    REVIEW_PAPER = "review_paper"
    BOOK_MONOGRAPH = "book_monograph"
    THESIS = "thesis"
    REPORT = "report"
    REFERENCE_MATERIAL = "reference_material"
    NON_ACADEMIC = "non_academic"


@dataclass(frozen=True)
class TypeProfile:
    """Extraction and validation profile for a document type.

    Attributes:
        required_extraction_fields: Fields that must be present for valid extraction.
        recommended_extraction_fields: Fields that should be present but are not required.
        min_confidence_threshold: Minimum extraction confidence for this type.
        extraction_prompt_key: Key into EXTRACTION_PROMPTS for the appropriate template.
        description: Human-readable description of this document type.
    """

    required_extraction_fields: tuple[str, ...] = ()
    recommended_extraction_fields: tuple[str, ...] = ()
    min_confidence_threshold: float = 0.5
    extraction_prompt_key: str = "full"
    description: str = ""


# Registry of type profiles keyed by DocumentType
TYPE_PROFILES: dict[DocumentType, TypeProfile] = {
    DocumentType.RESEARCH_PAPER: TypeProfile(
        required_extraction_fields=(
            "q02_thesis",
            "q07_methods",
            "q04_evidence",
        ),
        recommended_extraction_fields=(
            "q01_research_question",
            "q22_contribution",
        ),
        min_confidence_threshold=0.5,
        extraction_prompt_key="full",
        description="Journal articles, conference papers, preprints",
    ),
    DocumentType.REVIEW_PAPER: TypeProfile(
        required_extraction_fields=(
            "q02_thesis",
            "q03_key_claims",
        ),
        recommended_extraction_fields=(
            "q07_methods",
            "q22_contribution",
        ),
        min_confidence_threshold=0.5,
        extraction_prompt_key="review",
        description="Systematic reviews, meta-analyses, literature reviews",
    ),
    DocumentType.BOOK_MONOGRAPH: TypeProfile(
        required_extraction_fields=(
            "q02_thesis",
            "q03_key_claims",
        ),
        recommended_extraction_fields=(
            "q22_contribution",
            "q21_summary",
        ),
        min_confidence_threshold=0.4,
        extraction_prompt_key="book",
        description="Books, monographs, textbooks, book chapters",
    ),
    DocumentType.THESIS: TypeProfile(
        required_extraction_fields=(
            "q02_thesis",
            "q07_methods",
            "q04_evidence",
            "q01_research_question",
        ),
        recommended_extraction_fields=(
            "q22_contribution",
            "q05_limitations",
        ),
        min_confidence_threshold=0.5,
        extraction_prompt_key="full",
        description="Dissertations, master's theses",
    ),
    DocumentType.REPORT: TypeProfile(
        required_extraction_fields=(
            "q04_evidence",
            "q22_contribution",
        ),
        recommended_extraction_fields=(
            "q03_key_claims",
        ),
        min_confidence_threshold=0.4,
        extraction_prompt_key="report",
        description="Government reports, white papers, technical reports",
    ),
    DocumentType.REFERENCE_MATERIAL: TypeProfile(
        required_extraction_fields=(
            "q17_field",
            "q22_contribution",
        ),
        recommended_extraction_fields=(
            "q02_thesis",
        ),
        min_confidence_threshold=0.3,
        extraction_prompt_key="generic",
        description="Handbooks, encyclopedias, standards, datasets",
    ),
    DocumentType.NON_ACADEMIC: TypeProfile(
        required_extraction_fields=(),
        recommended_extraction_fields=(
            "keywords",
        ),
        min_confidence_threshold=0.0,
        extraction_prompt_key="generic",
        description="Presentations, forms, cheat sheets, web content",
    ),
}


# Direct mapping from Zotero item_type to DocumentType for unambiguous types
ZOTERO_TYPE_MAP: dict[str, DocumentType] = {
    "journalArticle": DocumentType.RESEARCH_PAPER,
    "conferencePaper": DocumentType.RESEARCH_PAPER,
    "preprint": DocumentType.RESEARCH_PAPER,
    "book": DocumentType.BOOK_MONOGRAPH,
    "bookSection": DocumentType.BOOK_MONOGRAPH,
    "thesis": DocumentType.THESIS,
    "report": DocumentType.REPORT,
    "presentation": DocumentType.NON_ACADEMIC,
    "webpage": DocumentType.NON_ACADEMIC,
    "blogPost": DocumentType.NON_ACADEMIC,
    "forumPost": DocumentType.NON_ACADEMIC,
    "film": DocumentType.NON_ACADEMIC,
    "artwork": DocumentType.NON_ACADEMIC,
    "audioRecording": DocumentType.NON_ACADEMIC,
    "videoRecording": DocumentType.NON_ACADEMIC,
    "map": DocumentType.NON_ACADEMIC,
    "encyclopediaArticle": DocumentType.REFERENCE_MATERIAL,
    "dictionaryEntry": DocumentType.REFERENCE_MATERIAL,
    "statute": DocumentType.REFERENCE_MATERIAL,
    "case": DocumentType.REFERENCE_MATERIAL,
    "hearing": DocumentType.REFERENCE_MATERIAL,
    "patent": DocumentType.REFERENCE_MATERIAL,
    "dataset": DocumentType.REFERENCE_MATERIAL,
    # "document" is intentionally absent -- requires Tier 2 classification
}


def get_profile(doc_type: DocumentType) -> TypeProfile:
    """Get the TypeProfile for a given DocumentType.

    Args:
        doc_type: The document type to look up.

    Returns:
        TypeProfile with extraction and validation configuration.
    """
    return TYPE_PROFILES[doc_type]


def get_required_fields(doc_type: str | None) -> list[str]:
    """Get required extraction fields for a document type string.

    Falls back to research_paper requirements for unknown or None types,
    ensuring backward compatibility with existing extractions.

    Args:
        doc_type: Document type string (from DocumentType enum value) or None.

    Returns:
        List of required field names.
    """
    if doc_type is None:
        return list(TYPE_PROFILES[DocumentType.RESEARCH_PAPER].required_extraction_fields)
    try:
        return list(TYPE_PROFILES[DocumentType(doc_type)].required_extraction_fields)
    except ValueError:
        return list(TYPE_PROFILES[DocumentType.RESEARCH_PAPER].required_extraction_fields)


def get_recommended_fields(doc_type: str | None) -> list[str]:
    """Get recommended extraction fields for a document type string.

    Args:
        doc_type: Document type string or None.

    Returns:
        List of recommended field names.
    """
    if doc_type is None:
        return list(
            TYPE_PROFILES[DocumentType.RESEARCH_PAPER].recommended_extraction_fields
        )
    try:
        return list(TYPE_PROFILES[DocumentType(doc_type)].recommended_extraction_fields)
    except ValueError:
        return list(
            TYPE_PROFILES[DocumentType.RESEARCH_PAPER].recommended_extraction_fields
        )
