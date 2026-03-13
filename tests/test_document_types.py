"""Tests for document type classification and type profiles."""


from src.analysis.document_classifier import (
    classify,
    classify_metadata,
    classify_text,
)
from src.analysis.document_types import (
    TYPE_PROFILES,
    ZOTERO_TYPE_MAP,
    DocumentType,
    get_recommended_fields,
    get_required_fields,
)
from src.zotero.models import PaperMetadata

# --- DocumentType and TypeProfile tests ---


class TestDocumentType:
    """Test DocumentType enum."""

    def test_all_types_have_profiles(self):
        """Every DocumentType must have a corresponding TypeProfile."""
        for doc_type in DocumentType:
            assert doc_type in TYPE_PROFILES, f"Missing profile for {doc_type}"

    def test_type_values_are_snake_case(self):
        """Type values should be snake_case strings."""
        for doc_type in DocumentType:
            assert doc_type.value == doc_type.value.lower()
            assert " " not in doc_type.value

    def test_profile_has_valid_prompt_key(self):
        """Each profile's extraction_prompt_key must be a valid key."""
        from src.analysis.prompts import DOCUMENT_TYPE_PROMPTS

        for doc_type, profile in TYPE_PROFILES.items():
            assert profile.extraction_prompt_key in DOCUMENT_TYPE_PROMPTS, (
                f"{doc_type} has invalid prompt key: {profile.extraction_prompt_key}"
            )


class TestGetRequiredFields:
    """Test get_required_fields helper."""

    def test_none_returns_research_paper_fields(self):
        """None document_type falls back to research_paper requirements."""
        fields = get_required_fields(None)
        assert "q02_thesis" in fields
        assert "q07_methods" in fields
        assert "q04_evidence" in fields

    def test_known_type_returns_correct_fields(self):
        """Known types return their configured fields."""
        fields = get_required_fields("book_monograph")
        assert "q02_thesis" in fields
        assert "q03_key_claims" in fields
        assert "q07_methods" not in fields

    def test_unknown_type_falls_back(self):
        """Unknown type string falls back to research_paper."""
        fields = get_required_fields("unknown_type_xyz")
        assert fields == get_required_fields(None)

    def test_report_fields(self):
        """Reports require q04_evidence and q22_contribution."""
        fields = get_required_fields("report")
        assert "q04_evidence" in fields
        assert "q22_contribution" in fields
        assert "q07_methods" not in fields

    def test_non_academic_has_no_required_fields(self):
        """Non-academic documents have no required extraction fields."""
        fields = get_required_fields("non_academic")
        assert fields == []

    def test_thesis_requires_research_question(self):
        """Theses should require q01_research_question."""
        fields = get_required_fields("thesis")
        assert "q01_research_question" in fields
        assert "q07_methods" in fields


class TestGetRecommendedFields:
    """Test get_recommended_fields helper."""

    def test_none_returns_research_paper_recommended(self):
        fields = get_recommended_fields(None)
        assert "q01_research_question" in fields or "q22_contribution" in fields

    def test_non_academic_recommends_keywords(self):
        fields = get_recommended_fields("non_academic")
        assert "keywords" in fields


# --- Classifier tests ---


def _make_paper(**kwargs) -> PaperMetadata:
    """Create a PaperMetadata with sensible defaults for testing."""
    defaults = {
        "paper_id": "test-001",
        "zotero_key": "ABC123",
        "zotero_item_id": 1,
        "title": "Test Paper",
        "authors": [],
        "item_type": "journalArticle",
        "date_added": "2024-01-01",
        "date_modified": "2024-01-01",
    }
    defaults.update(kwargs)
    return PaperMetadata(**defaults)


class TestClassifyMetadata:
    """Test Tier 1 (metadata-only) classification."""

    def test_journal_article(self):
        """journalArticle maps to RESEARCH_PAPER."""
        paper = _make_paper(item_type="journalArticle")
        doc_type, confidence = classify_metadata(paper)
        assert doc_type == DocumentType.RESEARCH_PAPER
        assert confidence >= 0.8

    def test_journal_article_with_doi_boosts_confidence(self):
        """DOI presence should boost confidence."""
        paper_no_doi = _make_paper(item_type="journalArticle")
        paper_doi = _make_paper(item_type="journalArticle", doi="10.1234/test")
        _, conf_no = classify_metadata(paper_no_doi)
        _, conf_doi = classify_metadata(paper_doi)
        assert conf_doi > conf_no

    def test_book(self):
        """book maps to BOOK_MONOGRAPH."""
        paper = _make_paper(item_type="book")
        doc_type, confidence = classify_metadata(paper)
        assert doc_type == DocumentType.BOOK_MONOGRAPH
        assert confidence >= 0.8

    def test_book_with_isbn_boosts_confidence(self):
        """ISBN presence should boost confidence for books."""
        paper = _make_paper(item_type="book", isbn="978-0-123456-47-2")
        doc_type, confidence = classify_metadata(paper)
        assert doc_type == DocumentType.BOOK_MONOGRAPH
        assert confidence >= 0.9

    def test_thesis(self):
        """thesis maps to THESIS."""
        paper = _make_paper(item_type="thesis")
        doc_type, _ = classify_metadata(paper)
        assert doc_type == DocumentType.THESIS

    def test_report(self):
        """report maps to REPORT."""
        paper = _make_paper(item_type="report")
        doc_type, _ = classify_metadata(paper)
        assert doc_type == DocumentType.REPORT

    def test_presentation(self):
        """presentation maps to NON_ACADEMIC."""
        paper = _make_paper(item_type="presentation")
        doc_type, _ = classify_metadata(paper)
        assert doc_type == DocumentType.NON_ACADEMIC

    def test_review_in_title(self):
        """Title containing 'systematic review' should classify as REVIEW_PAPER."""
        paper = _make_paper(
            item_type="journalArticle",
            title="A Systematic Review of Citation Analysis Methods",
        )
        doc_type, confidence = classify_metadata(paper)
        assert doc_type == DocumentType.REVIEW_PAPER
        assert confidence >= 0.8

    def test_meta_analysis_in_title(self):
        """Title containing 'meta-analysis' should classify as REVIEW_PAPER."""
        paper = _make_paper(
            item_type="journalArticle",
            title="A Meta-Analysis of Network Effects",
        )
        doc_type, _ = classify_metadata(paper)
        assert doc_type == DocumentType.REVIEW_PAPER

    def test_document_type_uncertain(self):
        """Zotero 'document' type should return low confidence."""
        paper = _make_paper(item_type="document")
        doc_type, confidence = classify_metadata(paper)
        assert confidence < 0.8  # Needs Tier 2

    def test_document_with_doi(self):
        """Zotero 'document' with DOI leans toward RESEARCH_PAPER."""
        paper = _make_paper(item_type="document", doi="10.1234/test")
        doc_type, confidence = classify_metadata(paper)
        assert doc_type == DocumentType.RESEARCH_PAPER

    def test_encyclopedia_article(self):
        """encyclopediaArticle maps to REFERENCE_MATERIAL."""
        paper = _make_paper(item_type="encyclopediaArticle")
        doc_type, _ = classify_metadata(paper)
        assert doc_type == DocumentType.REFERENCE_MATERIAL

    def test_all_mapped_types_resolve(self):
        """Every type in ZOTERO_TYPE_MAP should produce a valid classification."""
        for zotero_type, expected_doc_type in ZOTERO_TYPE_MAP.items():
            paper = _make_paper(item_type=zotero_type)
            doc_type, confidence = classify_metadata(paper)
            assert doc_type == expected_doc_type, f"Failed for {zotero_type}"
            assert confidence > 0


class TestClassifyText:
    """Test Tier 2 (text-based) classification."""

    def test_very_short_text_is_non_academic(self):
        """Documents under 200 words should classify as non-academic."""
        text = " ".join(["word"] * 100)
        doc_type, confidence = classify_text(text, word_count=100)
        assert doc_type == DocumentType.NON_ACADEMIC
        assert confidence >= 0.8

    def test_slides_detected_by_low_words_per_page(self):
        """Low words per page with many pages -> presentation/non-academic."""
        text = "slide content " * 40
        doc_type, _ = classify_text(
            text, word_count=200, page_count=10  # 20 words/page
        )
        assert doc_type == DocumentType.NON_ACADEMIC

    def test_very_long_document_is_book(self):
        """Documents over 40k words default to book."""
        text = "academic content " * 100
        doc_type, _ = classify_text(text, word_count=50000, page_count=300)
        assert doc_type == DocumentType.BOOK_MONOGRAPH

    def test_very_long_thesis_stays_thesis(self):
        """Long document with thesis metadata stays as thesis."""
        text = "academic content " * 100
        doc_type, confidence = classify_text(
            text,
            metadata_type=DocumentType.THESIS,
            word_count=50000,
            page_count=200,
        )
        assert doc_type == DocumentType.THESIS
        assert confidence >= 0.9

    def test_review_markers_in_text(self):
        """Text containing review markers should classify as review."""
        text = (
            "This systematic review examines the literature on citation analysis. "
            "We conducted a comprehensive search of databases. " * 50
        )
        doc_type, _ = classify_text(text, word_count=1000)
        assert doc_type == DocumentType.REVIEW_PAPER

    def test_metadata_type_confirmed_by_text(self):
        """High section markers should confirm research_paper metadata."""
        text = "academic content " * 500
        doc_type, confidence = classify_text(
            text,
            metadata_type=DocumentType.RESEARCH_PAPER,
            word_count=5000,
            page_count=20,
            section_marker_count=5,
        )
        assert doc_type == DocumentType.RESEARCH_PAPER
        assert confidence >= 0.8


class TestClassifyFull:
    """Test full two-tier classification."""

    def test_high_confidence_metadata_skips_tier2(self):
        """High-confidence Tier 1 result should not need text."""
        paper = _make_paper(
            item_type="journalArticle",
            doi="10.1234/test",
            journal="Nature",
        )
        doc_type, confidence = classify(paper)
        assert doc_type == DocumentType.RESEARCH_PAPER
        assert confidence >= 0.8

    def test_low_confidence_metadata_uses_text(self):
        """Low-confidence Tier 1 should use Tier 2 when text available."""
        paper = _make_paper(item_type="document")
        text = (
            "Abstract Introduction Methods Results Discussion Conclusion "
            "This study investigates the relationship between X and Y. "
            "Our findings demonstrate that... [1] [2] [3] " * 50
        )
        doc_type, confidence = classify(
            paper,
            text=text,
            word_count=5000,
            page_count=20,
            section_marker_count=5,
        )
        assert doc_type == DocumentType.RESEARCH_PAPER

    def test_no_text_returns_best_tier1(self):
        """Without text, should return best Tier 1 result."""
        paper = _make_paper(item_type="document")
        doc_type, confidence = classify(paper, text=None)
        # Should return something, even if low confidence
        assert isinstance(doc_type, DocumentType)

    def test_non_academic_classification(self):
        """Presentation should classify as non-academic."""
        paper = _make_paper(item_type="presentation")
        doc_type, confidence = classify(paper)
        assert doc_type == DocumentType.NON_ACADEMIC
        assert confidence >= 0.8


class TestValidationIntegration:
    """Test that type-aware validation works correctly."""

    def test_book_valid_without_methods(self):
        """A book extraction should be valid without q07_methods."""
        fields = get_required_fields("book_monograph")
        assert "q07_methods" not in fields

    def test_report_valid_without_research_question(self):
        """A report should be valid without q01_research_question."""
        fields = get_required_fields("report")
        assert "q01_research_question" not in fields

    def test_research_paper_requires_methods(self):
        """Research papers should still require q07_methods."""
        fields = get_required_fields("research_paper")
        assert "q07_methods" in fields

    def test_reference_material_minimal_requirements(self):
        """Reference materials have minimal requirements."""
        fields = get_required_fields("reference_material")
        assert "q17_field" in fields
        assert "q22_contribution" in fields
        assert len(fields) == 2
