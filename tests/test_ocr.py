"""Tests for OCR handler and integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.ocr_handler import (
    OCRError,
    OCRHandler,
    OCRResult,
    get_ocr_handler,
)
from src.extraction.pdf_extractor import PDFExtractor


class TestOCRHandler:
    """Test OCR handler functionality."""

    def test_ocr_result_named_tuple(self):
        """Test OCRResult structure."""
        result = OCRResult(
            text="Sample text",
            pages_processed=5,
            method="ocr",
        )
        assert result.text == "Sample text"
        assert result.pages_processed == 5
        assert result.method == "ocr"

    def test_check_dependencies_returns_dict(self):
        """Test dependency check returns proper structure."""
        deps = OCRHandler.check_dependencies()
        assert isinstance(deps, dict)
        assert "pytesseract_installed" in deps
        assert "tesseract_available" in deps
        assert "tesseract_path" in deps
        assert "pdf2image_installed" in deps
        assert "pillow_installed" in deps
        assert "poppler_path" in deps
        assert "poppler_available" in deps

    def test_needs_ocr_low_word_count(self):
        """Test OCR detection for low word density."""
        handler = OCRHandler()
        # 10 words for 5 pages = 2 words/page (well below threshold of 50)
        text = "word " * 10
        assert handler.needs_ocr(text, page_count=5) is True

    def test_needs_ocr_sufficient_text(self):
        """Test OCR not needed for text-rich PDFs."""
        handler = OCRHandler()
        # 500 words for 5 pages = 100 words/page (above threshold)
        text = "--- Page 1 ---\n" + "word " * 100
        text += "--- Page 2 ---\n" + "word " * 100
        text += "--- Page 3 ---\n" + "word " * 100
        text += "--- Page 4 ---\n" + "word " * 100
        text += "--- Page 5 ---\n" + "word " * 100
        assert handler.needs_ocr(text, page_count=5) is False

    def test_needs_ocr_zero_pages(self):
        """Test OCR with zero pages returns False."""
        handler = OCRHandler()
        assert handler.needs_ocr("some text", page_count=0) is False

    def test_needs_ocr_low_text_page_ratio(self):
        """Test OCR detection for missing page text."""
        handler = OCRHandler()
        # Only 1 page marker for 5-page document
        text = "--- Page 1 ---\n" + "word " * 100
        assert handler.needs_ocr(text, page_count=5) is True

    @patch("src.extraction.ocr_handler.TESSERACT_AVAILABLE", False)
    def test_is_available_no_tesseract(self):
        """Test availability check when Tesseract missing."""
        # Need to reload to pick up patched value
        with patch("src.extraction.ocr_handler.TESSERACT_AVAILABLE", False):
            assert OCRHandler.is_available() is False

    def test_extract_text_missing_file(self):
        """Test extraction raises error for missing file or missing dependencies."""
        handler = OCRHandler()
        with pytest.raises(OCRError):
            # Will fail with "not found" if OCR available, or "dependencies not available" if not
            handler.extract_text(Path("/nonexistent/file.pdf"))


class TestOCRHandlerIntegration:
    """Integration tests for OCR with PDFExtractor."""

    def test_pdf_extractor_ocr_disabled_by_default(self):
        """Test OCR is disabled by default."""
        extractor = PDFExtractor()
        assert extractor.enable_ocr is False
        assert extractor.ocr_handler is None

    def test_pdf_extractor_enable_ocr(self):
        """Test OCR can be enabled."""
        # This may or may not work depending on Tesseract installation
        extractor = PDFExtractor(enable_ocr=True)
        assert extractor.enable_ocr is True
        # Handler will be None if Tesseract not installed

    def test_pdf_extractor_ocr_config(self):
        """Test OCR configuration is passed through."""
        config = {"dpi": 150, "lang": "deu"}
        extractor = PDFExtractor(enable_ocr=True, ocr_config=config)
        if extractor.ocr_handler:
            assert extractor.ocr_handler.dpi == 150
            assert extractor.ocr_handler.lang == "deu"


class TestGetOCRHandler:
    """Test get_ocr_handler factory function."""

    def test_get_ocr_handler_returns_handler_or_none(self):
        """Test factory returns handler when available, None otherwise."""
        handler = get_ocr_handler()
        # Will be None if Tesseract not installed, OCRHandler otherwise
        if handler is not None:
            assert isinstance(handler, OCRHandler)

    def test_get_ocr_handler_passes_kwargs(self):
        """Test factory passes configuration."""
        handler = get_ocr_handler(dpi=200, lang="fra")
        if handler is not None:
            assert handler.dpi == 200
            assert handler.lang == "fra"


class TestOCRMocked:
    """Tests with mocked OCR dependencies."""

    @patch("src.extraction.ocr_handler.TESSERACT_AVAILABLE", True)
    @patch("src.extraction.ocr_handler.PDF2IMAGE_AVAILABLE", True)
    @patch("src.extraction.ocr_handler.convert_from_path")
    @patch("src.extraction.ocr_handler.pytesseract")
    def test_extract_text_with_mocked_ocr(self, mock_tesseract, mock_convert, *args):
        """Test OCR extraction with mocked dependencies."""
        # Setup mocks
        mock_image = MagicMock()
        mock_convert.return_value = [mock_image, mock_image]
        mock_tesseract.image_to_string.return_value = "OCR extracted text"

        handler = OCRHandler()
        # Create a temp file for testing
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 test")
            temp_path = Path(f.name)

        try:
            # Mock is_available to return True for this test
            with patch.object(OCRHandler, 'is_available', return_value=True):
                result = handler.extract_text(temp_path)
                assert result.method == "ocr"
                assert result.pages_processed == 2
                assert "OCR extracted text" in result.text
        finally:
            temp_path.unlink(missing_ok=True)


class TestPDFExtractorWithOCRFallback:
    """Test PDFExtractor with OCR fallback behavior."""

    def test_extract_text_with_method_returns_tuple(self, tmp_path):
        """Test extract_text_with_method returns method info."""
        # Create a minimal PDF for testing
        pdf_path = tmp_path / "test.pdf"

        # Create using pymupdf
        import pymupdf
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Test content for extraction")
        doc.save(str(pdf_path))
        doc.close()

        extractor = PDFExtractor()
        text, method = extractor.extract_text_with_method(pdf_path)

        assert isinstance(text, str)
        assert method == "pymupdf"
        assert "Test content" in text

    def test_extract_text_still_returns_string(self, tmp_path):
        """Test extract_text maintains backward compatibility."""
        # Create a minimal PDF
        pdf_path = tmp_path / "test.pdf"

        import pymupdf
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Test content")
        doc.save(str(pdf_path))
        doc.close()

        extractor = PDFExtractor()
        result = extractor.extract_text(pdf_path)

        assert isinstance(result, str)
        assert "Test content" in result
