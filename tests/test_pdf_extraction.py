"""Tests for PDF extraction and text cleaning."""


import pymupdf
import pytest

from src.extraction.pdf_extractor import PDFExtractionError, PDFExtractor
from src.extraction.text_cleaner import TextCleaner, TextStats


class TestPDFExtractor:
    """Tests for PDFExtractor class."""

    @pytest.fixture
    def sample_pdf(self, tmp_path):
        """Create a sample PDF for testing."""
        pdf_path = tmp_path / "sample.pdf"
        doc = pymupdf.open()

        # Add pages with text
        page1 = doc.new_page()
        page1.insert_text((50, 50), "This is page one.\nWith some text content.")

        page2 = doc.new_page()
        page2.insert_text((50, 50), "This is page two.\nWith more content here.")

        doc.save(pdf_path)
        doc.close()
        return pdf_path

    @pytest.fixture
    def extractor(self, tmp_path):
        """Create extractor with cache."""
        cache_dir = tmp_path / "cache"
        return PDFExtractor(cache_dir=cache_dir)

    def test_extract_text_basic(self, extractor, sample_pdf):
        """Test basic text extraction."""
        text = extractor.extract_text(sample_pdf)
        assert "page one" in text
        assert "page two" in text

    def test_extract_text_with_cache(self, extractor, sample_pdf):
        """Test that extraction is cached."""
        # First extraction
        text1 = extractor.extract_text(sample_pdf)

        # Second extraction should use cache
        text2 = extractor.extract_text(sample_pdf)

        assert text1 == text2
        # Cache file should exist
        assert len(list(extractor.cache_dir.glob("*.txt"))) == 1

    def test_extract_text_missing_file(self, extractor, tmp_path):
        """Test extraction of non-existent file."""
        with pytest.raises(PDFExtractionError, match="not found"):
            extractor.extract_text(tmp_path / "nonexistent.pdf")

    def test_extract_with_metadata(self, extractor, sample_pdf):
        """Test extraction with metadata."""
        result = extractor.extract_with_metadata(sample_pdf)

        assert "text" in result
        assert result["page_count"] == 2
        assert result["file_size"] > 0
        assert "metadata" in result

    def test_get_page_count(self, extractor, sample_pdf):
        """Test page count retrieval."""
        count = extractor.get_page_count(sample_pdf)
        assert count == 2

    def test_clear_cache(self, extractor, sample_pdf):
        """Test cache clearing."""
        # Create some cache
        extractor.extract_text(sample_pdf)
        assert len(list(extractor.cache_dir.glob("*.txt"))) == 1

        # Clear cache
        count = extractor.clear_cache()
        assert count == 1
        assert len(list(extractor.cache_dir.glob("*.txt"))) == 0

    def test_no_cache_mode(self, sample_pdf, tmp_path):
        """Test extraction without caching."""
        extractor = PDFExtractor(cache_dir=None)
        text = extractor.extract_text(sample_pdf)
        assert "page one" in text


class TestTextCleaner:
    """Tests for TextCleaner class."""

    @pytest.fixture
    def cleaner(self):
        """Create text cleaner."""
        return TextCleaner()

    def test_fix_hyphenation(self, cleaner):
        """Test hyphenated line break fixing."""
        text = "This is a hyph-\nenated word."
        result = cleaner.clean(text)
        assert "hyphenated" in result

    def test_remove_multiple_spaces(self, cleaner):
        """Test multiple space normalization."""
        text = "This  has    multiple   spaces."
        result = cleaner.clean(text)
        assert "  " not in result

    def test_remove_multiple_newlines(self, cleaner):
        """Test multiple newline normalization."""
        text = "Line one.\n\n\n\n\nLine two."
        result = cleaner.clean(text)
        assert "\n\n\n" not in result

    def test_get_stats(self, cleaner):
        """Test text statistics."""
        text = "--- Page 1 ---\nThis is some text.\n\n--- Page 2 ---\nMore text here."
        stats = cleaner.get_stats(text)

        assert isinstance(stats, TextStats)
        assert stats.page_count == 2
        assert stats.word_count > 0
        assert stats.line_count > 0

    def test_is_valid_extraction_sufficient(self, cleaner):
        """Test valid extraction detection with sufficient content."""
        text = " ".join(["word"] * 150)
        assert cleaner.is_valid_extraction(text, min_words=100)

    def test_is_valid_extraction_insufficient(self, cleaner):
        """Test valid extraction detection with insufficient content."""
        text = "Short text."
        assert not cleaner.is_valid_extraction(text, min_words=100)

    def test_is_valid_extraction_empty(self, cleaner):
        """Test valid extraction detection with empty content."""
        assert not cleaner.is_valid_extraction("")
        assert not cleaner.is_valid_extraction(None)

    def test_truncate_for_llm_no_truncation_needed(self, cleaner):
        """Test truncation when text is short enough."""
        text = "Short text that doesn't need truncation."
        result = cleaner.truncate_for_llm(text, max_chars=1000)
        assert result == text

    def test_truncate_for_llm_truncates(self, cleaner):
        """Test truncation of long text."""
        text = "A" * 200000
        result = cleaner.truncate_for_llm(text, max_chars=50000)

        assert len(result) < 200000
        assert "[... content truncated ...]" in result

    def test_extract_sections_abstract(self, cleaner):
        """Test section extraction for abstract."""
        text = """
        ABSTRACT

        This paper examines the relationship between citation networks and research impact.
        We find significant correlations.

        INTRODUCTION

        Research impact measurement is important.
        """
        sections = cleaner.extract_sections(text)
        assert "abstract" in sections

    def test_clean_preserves_content(self, cleaner):
        """Test that cleaning preserves meaningful content."""
        text = """
        This is meaningful content that should be preserved.
        It contains important information for analysis.
        Multiple sentences with various points.
        """
        result = cleaner.clean(text)
        assert "meaningful content" in result
        assert "important information" in result


class TestTextCleanerConfiguration:
    """Tests for TextCleaner configuration options."""

    def test_min_line_length_filtering(self):
        """Test minimum line length filtering."""
        cleaner = TextCleaner(min_line_length=20)
        text = "Short\nThis is a longer line that should be kept.\nTiny"
        result = cleaner.clean(text)

        assert "longer line" in result

    def test_disable_hyphenation_fix(self):
        """Test disabling hyphenation fixing."""
        cleaner = TextCleaner(fix_hyphenation=False)
        text = "hyph-\nenated"
        result = cleaner.clean(text)
        # Should not be fixed
        assert "hyphenated" not in result
