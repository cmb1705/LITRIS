"""Two-tier document type classifier using metadata and text heuristics."""

import re

from src.analysis.document_types import ZOTERO_TYPE_MAP, DocumentType
from src.utils.logging_config import get_logger
from src.zotero.models import PaperMetadata

logger = get_logger(__name__)

# Tier 2 patterns compiled once at module level
_REVIEW_PATTERNS = re.compile(
    r"\b(systematic\s+review|meta[-\s]?analysis|literature\s+review|scoping\s+review"
    r"|narrative\s+review|integrative\s+review|rapid\s+review|umbrella\s+review)\b",
    re.IGNORECASE,
)

_ACADEMIC_PHRASES = re.compile(
    r"\b(we\s+hypothesiz|this\s+study|our\s+findings|this\s+paper\s+(?:presents|examines|investigates)"
    r"|the\s+results\s+(?:show|indicate|suggest|demonstrate)"
    r"|we\s+(?:propose|argue|demonstrate|analyze|examine|investigate)"
    r"|our\s+(?:analysis|results|approach|method|contribution))\b",
    re.IGNORECASE,
)

_CITATION_BRACKET = re.compile(r"\[\d+\]")
_CITATION_AUTHOR_YEAR = re.compile(r"\([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z]))?,?\s*\d{4}\)")
_WEB_REPORT_TITLE_PATTERNS = re.compile(
    r"\b(guideline|guidance|position statement|framework|report|white paper|policy|"
    r"technology transfer|regulatory science|standard|reference|adaptive cycle)\b",
    re.IGNORECASE,
)
_WEB_REPORT_DOMAIN_PATTERNS = re.compile(
    r"(?:who\.int|ema\.europa\.eu|fda\.gov|tga\.gov\.au|resalliance\.org)",
    re.IGNORECASE,
)


def classify_metadata(paper: PaperMetadata) -> tuple[DocumentType | None, float]:
    """Tier 1: Classify document type from metadata alone.

    Uses Zotero item_type, DOI, ISBN, journal name, and title keywords to
    produce a classification. Returns None when confidence is too low.

    Args:
        paper: Paper metadata from Zotero.

    Returns:
        Tuple of (DocumentType or None, confidence 0.0-1.0).
    """
    if paper.item_type == "webpage":
        title_lower = paper.title.lower() if paper.title else ""
        url_lower = paper.url.lower() if paper.url else ""
        if paper.doi:
            return DocumentType.RESEARCH_PAPER, 0.7
        if _WEB_REPORT_DOMAIN_PATTERNS.search(url_lower) or _WEB_REPORT_TITLE_PATTERNS.search(
            title_lower
        ):
            return DocumentType.REPORT, 0.65
        return DocumentType.REFERENCE_MATERIAL, 0.55

    # Direct mapping from Zotero item_type
    if paper.item_type in ZOTERO_TYPE_MAP:
        doc_type = ZOTERO_TYPE_MAP[paper.item_type]
        confidence = 0.85

        # Refine journalArticle/conferencePaper: check for review indicators
        if doc_type == DocumentType.RESEARCH_PAPER:
            title_lower = paper.title.lower() if paper.title else ""
            if _REVIEW_PATTERNS.search(title_lower):
                doc_type = DocumentType.REVIEW_PAPER
                confidence = 0.9

        # Boost confidence with corroborating signals
        if doc_type == DocumentType.RESEARCH_PAPER:
            if paper.doi:
                confidence = min(confidence + 0.05, 1.0)
            if paper.journal:
                confidence = min(confidence + 0.05, 1.0)
        elif doc_type == DocumentType.BOOK_MONOGRAPH:
            if paper.isbn:
                confidence = min(confidence + 0.1, 1.0)

        return doc_type, confidence

    # Zotero "document" type is ambiguous -- needs Tier 2
    if paper.item_type == "document":
        # Can still try some metadata heuristics
        if paper.doi:
            return DocumentType.RESEARCH_PAPER, 0.6
        if paper.isbn:
            return DocumentType.BOOK_MONOGRAPH, 0.6
        return None, 0.3

    # Unknown item_type -- low confidence guess from available signals
    if paper.doi and paper.journal:
        return DocumentType.RESEARCH_PAPER, 0.7
    if paper.isbn:
        return DocumentType.BOOK_MONOGRAPH, 0.6

    return None, 0.2


def classify_text(
    text: str,
    metadata_type: DocumentType | None = None,
    word_count: int | None = None,
    page_count: int | None = None,
    section_marker_count: int | None = None,
) -> tuple[DocumentType, float]:
    """Tier 2: Refine classification using extracted text statistics.

    Analyzes word count, section markers, citation density, and academic
    language patterns to classify or refine a document type.

    Args:
        text: Cleaned document text (or first ~5000 chars for efficiency).
        metadata_type: Tier 1 classification result (may be None).
        word_count: Pre-computed word count (computed from text if None).
        page_count: Pre-computed page count (defaults to 1 if None).
        section_marker_count: Pre-computed count of IMRaD section markers.

    Returns:
        Tuple of (DocumentType, confidence 0.0-1.0).
    """
    if word_count is None:
        word_count = len(text.split())
    if page_count is None or page_count < 1:
        page_count = 1

    words_per_page = word_count / page_count

    # Compute text-based signals from first portion of text
    sample = text[:5000]

    # Review markers in title/abstract area
    review_match = bool(_REVIEW_PATTERNS.search(sample))

    # Citation density (per 1000 words)
    bracket_cites = len(_CITATION_BRACKET.findall(text[:20000]))
    author_year_cites = len(_CITATION_AUTHOR_YEAR.findall(text[:20000]))
    citation_count = bracket_cites + author_year_cites
    text_sample_words = len(text[:20000].split()) or 1
    citation_density = citation_count / text_sample_words * 1000

    # Academic phrase count
    academic_phrases = len(_ACADEMIC_PHRASES.findall(sample))

    # Very short documents -> likely non-academic
    if word_count < 200:
        return DocumentType.NON_ACADEMIC, 0.85

    # Very low words per page -> likely slides/presentation
    if words_per_page < 50 and page_count > 3:
        return DocumentType.NON_ACADEMIC, 0.8

    # Very long documents -> likely book or thesis
    if word_count > 40000:
        if metadata_type == DocumentType.THESIS:
            return DocumentType.THESIS, 0.95
        if metadata_type == DocumentType.BOOK_MONOGRAPH:
            return DocumentType.BOOK_MONOGRAPH, 0.95
        # Default to book for very long docs
        return DocumentType.BOOK_MONOGRAPH, 0.7

    # Review paper signals
    if review_match:
        if metadata_type in (DocumentType.RESEARCH_PAPER, DocumentType.REVIEW_PAPER, None):
            return DocumentType.REVIEW_PAPER, 0.85

    # If metadata gave us a high-confidence answer, confirm with text signals
    if metadata_type is not None:
        confidence = 0.8

        # Corroborate with text signals
        if metadata_type == DocumentType.RESEARCH_PAPER:
            if section_marker_count is not None and section_marker_count >= 3:
                confidence = min(confidence + 0.1, 1.0)
            if academic_phrases >= 2:
                confidence = min(confidence + 0.05, 1.0)
            if citation_density > 5:
                confidence = min(confidence + 0.05, 1.0)

        return metadata_type, confidence

    # No metadata type -- classify from scratch using text signals
    if section_marker_count is not None and section_marker_count >= 4:
        if citation_density > 5 and academic_phrases >= 2:
            return DocumentType.RESEARCH_PAPER, 0.75
        return DocumentType.RESEARCH_PAPER, 0.65

    if citation_density > 10:
        return DocumentType.RESEARCH_PAPER, 0.6

    if citation_density < 2 and academic_phrases < 2:
        if word_count > 5000:
            return DocumentType.REPORT, 0.5
        return DocumentType.NON_ACADEMIC, 0.6

    # Fallback: moderate academic signals -> research paper
    return DocumentType.RESEARCH_PAPER, 0.5


def classify(
    paper: PaperMetadata,
    text: str | None = None,
    word_count: int | None = None,
    page_count: int | None = None,
    section_marker_count: int | None = None,
    high_confidence_threshold: float = 0.8,
) -> tuple[DocumentType, float]:
    """Full two-tier classification.

    Runs Tier 1 (metadata) first. If confidence is high enough, returns
    immediately. Otherwise runs Tier 2 (text analysis) to refine.

    Args:
        paper: Paper metadata from Zotero.
        text: Cleaned document text (needed for Tier 2).
        word_count: Pre-computed word count for Tier 2.
        page_count: Pre-computed page count for Tier 2.
        section_marker_count: Pre-computed section marker count for Tier 2.
        high_confidence_threshold: Tier 1 confidence above which Tier 2 is skipped.

    Returns:
        Tuple of (DocumentType, confidence 0.0-1.0).
    """
    # Tier 1: metadata
    meta_type, meta_confidence = classify_metadata(paper)

    if meta_type is not None and meta_confidence >= high_confidence_threshold:
        logger.debug(
            f"Tier 1 classified '{paper.title[:50]}' as {meta_type.value} "
            f"(confidence={meta_confidence:.2f})"
        )
        return meta_type, meta_confidence

    # Tier 2: text analysis (requires text)
    if text is None:
        # No text available -- return best Tier 1 result or default
        if meta_type is not None:
            return meta_type, meta_confidence
        return DocumentType.RESEARCH_PAPER, 0.3

    doc_type, text_confidence = classify_text(
        text=text,
        metadata_type=meta_type,
        word_count=word_count,
        page_count=page_count,
        section_marker_count=section_marker_count,
    )

    logger.debug(
        f"Tier 2 classified '{paper.title[:50]}' as {doc_type.value} "
        f"(confidence={text_confidence:.2f}, tier1={meta_type})"
    )

    return doc_type, text_confidence
