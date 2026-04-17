"""Tests for multi-tier PDF extraction cascade."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.arxiv_extractor import (
    _extract_text_from_html,
    detect_arxiv_id,
)
from src.extraction.cascade import (
    MIN_EXTRACTION_WORDS,
    ExtractionCascade,
)
from src.extraction.opendataloader_extractor import OpenDataLoaderHybridConfig
from src.extraction.pdf_extractor import PDFExtractionError

# --- arXiv ID Detection ---


class TestDetectArxivId:
    """Tests for arXiv ID detection from metadata."""

    def test_detects_new_format_from_doi(self):
        assert detect_arxiv_id(doi="10.48550/arXiv.2301.12345") == "2301.12345"

    def test_detects_old_format_from_doi(self):
        assert detect_arxiv_id(doi="10.48550/arXiv.hep-th/0601001") == "hep-th/0601001"

    def test_detects_from_url(self):
        assert detect_arxiv_id(url="https://arxiv.org/abs/2301.12345") == "2301.12345"

    def test_detects_from_pdf_url(self):
        assert detect_arxiv_id(url="https://arxiv.org/pdf/2301.12345.pdf") == "2301.12345"

    def test_detects_from_pdf_path(self):
        assert detect_arxiv_id(pdf_path=Path("2301.12345v2.pdf")) == "2301.12345"

    def test_returns_none_for_non_arxiv(self):
        assert detect_arxiv_id(doi="10.1000/xyz123", url="https://example.com") is None

    def test_returns_none_for_empty_input(self):
        assert detect_arxiv_id() is None

    def test_strips_version_suffix(self):
        result = detect_arxiv_id(url="https://arxiv.org/abs/2301.12345v3")
        assert result == "2301.12345"


# --- HTML Text Extraction ---


class TestExtractTextFromHtml:
    """Tests for HTML to text conversion."""

    def test_strips_tags(self):
        html = "<p>Hello <b>world</b></p>"
        text = _extract_text_from_html(html)
        assert "Hello" in text
        assert "world" in text
        assert "<" not in text

    def test_removes_script_blocks(self):
        html = "<p>Before</p><script>var x = 1;</script><p>After</p>"
        text = _extract_text_from_html(html)
        assert "Before" in text
        assert "After" in text
        assert "var x" not in text

    def test_removes_style_blocks(self):
        html = "<style>.red { color: red; }</style><p>Content</p>"
        text = _extract_text_from_html(html)
        assert "Content" in text
        assert "color" not in text

    def test_decodes_html_entities(self):
        html = "<p>A &amp; B &lt; C &gt; D</p>"
        text = _extract_text_from_html(html)
        assert "A & B < C > D" in text

    def test_handles_empty_html(self):
        assert _extract_text_from_html("") == ""

    def test_collapses_whitespace(self):
        html = "<p>Word   one</p>   <p>Word   two</p>"
        text = _extract_text_from_html(html)
        # Should not have excessive whitespace
        assert "  " not in text.replace("\n\n", "")


# --- Marker Availability ---


class TestMarkerExtractor:
    """Tests for Marker extractor (mocked since Marker is not installed)."""

    @patch("src.extraction.marker_extractor.MARKER_AVAILABLE", False)
    def test_is_available_returns_false_when_not_installed(self):
        from src.extraction import marker_extractor

        # Reload to pick up patched value
        assert not marker_extractor.is_available()

    @patch("src.extraction.marker_extractor.MARKER_AVAILABLE", False)
    def test_extract_returns_none_when_not_installed(self):
        from src.extraction import marker_extractor

        result = marker_extractor.extract_with_marker(Path("test.pdf"))
        assert result is None


# --- Extraction Cascade ---


class TestExtractionCascade:
    """Tests for the multi-tier extraction cascade."""

    # Text with >100 words for sufficiency checks
    _LONG_TEXT = " ".join(f"word{i}" for i in range(150))

    @pytest.fixture
    def mock_pdf_extractor(self):
        """Create a mock PDFExtractor."""
        extractor = MagicMock()
        extractor.extract_text_with_method.return_value = (
            self._LONG_TEXT,
            "pymupdf",
        )
        return extractor

    @patch("src.extraction.cascade.opendataloader_extractor.is_available")
    def test_cascade_uses_pymupdf_when_odl_unavailable(
        self,
        mock_odl_available,
        mock_pdf_extractor,
    ):
        """Non-arXiv paper falls back to PyMuPDF when ODL not available."""
        mock_odl_available.return_value = False

        cascade = ExtractionCascade(
            pdf_extractor=mock_pdf_extractor,
            enable_arxiv=False,
            enable_opendataloader=True,
            enable_marker=False,
        )
        result = cascade.extract_text(Path("paper.pdf"))

        assert result.method == "pymupdf"
        assert "pymupdf" in result.tiers_attempted
        assert "opendataloader" not in result.tiers_attempted
        mock_pdf_extractor.extract_text_with_method.assert_called_once()

    @patch("src.extraction.cascade.arxiv_extractor.detect_arxiv_id")
    @patch("src.extraction.cascade.arxiv_extractor.fetch_arxiv_html")
    def test_cascade_tries_arxiv_html_first(
        self,
        mock_fetch,
        mock_detect,
        mock_pdf_extractor,
    ):
        """arXiv paper tries HTML first."""
        mock_detect.return_value = "2301.12345"
        mock_fetch.return_value = self._LONG_TEXT

        cascade = ExtractionCascade(
            pdf_extractor=mock_pdf_extractor,
            enable_opendataloader=False,
            enable_marker=False,
        )
        result = cascade.extract_text(Path("2301.12345.pdf"), doi="10.48550/arXiv.2301.12345")

        assert result.method == "arxiv_html"
        assert "arxiv_html" in result.tiers_attempted
        # PyMuPDF should NOT have been called
        mock_pdf_extractor.extract_text_with_method.assert_not_called()

    @patch("src.extraction.cascade.arxiv_extractor.detect_arxiv_id")
    @patch("src.extraction.cascade.arxiv_extractor.fetch_arxiv_html")
    @patch("src.extraction.cascade.arxiv_extractor.fetch_ar5iv_html")
    def test_cascade_falls_through_to_ar5iv(
        self,
        mock_ar5iv,
        mock_arxiv,
        mock_detect,
        mock_pdf_extractor,
    ):
        """If arXiv HTML fails, tries ar5iv."""
        mock_detect.return_value = "2301.12345"
        mock_arxiv.return_value = None  # arXiv HTML fails
        mock_ar5iv.return_value = self._LONG_TEXT

        cascade = ExtractionCascade(
            pdf_extractor=mock_pdf_extractor,
            enable_opendataloader=False,
            enable_marker=False,
        )
        result = cascade.extract_text(Path("paper.pdf"), doi="10.48550/arXiv.2301.12345")

        assert result.method == "ar5iv"
        assert "arxiv_html" in result.tiers_attempted
        assert "ar5iv" in result.tiers_attempted

    @patch("src.extraction.cascade.opendataloader_extractor.is_available")
    @patch("src.extraction.cascade.arxiv_extractor.detect_arxiv_id")
    @patch("src.extraction.cascade.arxiv_extractor.fetch_arxiv_html")
    @patch("src.extraction.cascade.arxiv_extractor.fetch_ar5iv_html")
    def test_cascade_falls_through_arxiv_to_pymupdf(
        self,
        mock_ar5iv,
        mock_arxiv,
        mock_detect,
        mock_odl_available,
        mock_pdf_extractor,
    ):
        """If arXiv tiers fail and ODL unavailable, falls through to PyMuPDF."""
        mock_detect.return_value = "2301.12345"
        mock_arxiv.return_value = None
        mock_ar5iv.return_value = None
        mock_odl_available.return_value = False

        cascade = ExtractionCascade(
            pdf_extractor=mock_pdf_extractor,
            enable_opendataloader=True,
            enable_marker=False,
        )
        result = cascade.extract_text(Path("paper.pdf"), doi="10.48550/arXiv.2301.12345")

        assert result.method == "pymupdf"
        mock_pdf_extractor.extract_text_with_method.assert_called_once()

    def test_cascade_raises_when_all_fail(self):
        """Raises PDFExtractionError when all tiers fail."""
        mock_extractor = MagicMock()
        mock_extractor.extract_text_with_method.side_effect = PDFExtractionError("fail")

        cascade = ExtractionCascade(
            pdf_extractor=mock_extractor,
            enable_arxiv=False,
            enable_opendataloader=False,
            enable_marker=False,
        )
        with pytest.raises(PDFExtractionError, match="All extraction tiers failed"):
            cascade.extract_text(Path("bad.pdf"))

    def test_cascade_rejects_insufficient_text(self, mock_pdf_extractor):
        """Text below min_words threshold is rejected."""
        mock_pdf_extractor.extract_text_with_method.return_value = ("Short text.", "pymupdf")

        cascade = ExtractionCascade(
            pdf_extractor=mock_pdf_extractor,
            enable_arxiv=False,
            enable_opendataloader=False,
            enable_marker=False,
            min_words=100,
        )
        with pytest.raises(PDFExtractionError):
            cascade.extract_text(Path("paper.pdf"))

    @patch("src.extraction.cascade.marker_extractor.is_available")
    @patch("src.extraction.cascade.marker_extractor.extract_with_marker")
    def test_cascade_uses_marker_as_fallback(
        self,
        mock_extract,
        mock_available,
        mock_pdf_extractor,
    ):
        """Marker fires as fallback when PyMuPDF produces insufficient text."""
        mock_available.return_value = True
        mock_extract.return_value = self._LONG_TEXT
        # PyMuPDF returns insufficient text, triggering Marker fallback
        mock_pdf_extractor.extract_text_with_method.return_value = ("Short.", "pymupdf")

        cascade = ExtractionCascade(
            pdf_extractor=mock_pdf_extractor,
            enable_arxiv=False,
            enable_opendataloader=False,
            enable_marker=True,
        )
        # Force enable_marker since we're mocking is_available after init
        cascade.enable_marker = True

        result = cascade.extract_text(Path("paper.pdf"))

        assert result.method == "marker"
        assert "marker" in result.tiers_attempted
        assert "pymupdf" in result.tiers_attempted

    @patch("src.extraction.cascade.opendataloader_extractor.get_last_attempt_result")
    @patch("src.extraction.cascade.opendataloader_extractor.is_available")
    @patch("src.extraction.cascade.opendataloader_extractor.extract_with_opendataloader")
    def test_cascade_uses_opendataloader_as_primary(
        self,
        mock_extract,
        mock_available,
        mock_last_attempt,
        mock_pdf_extractor,
    ):
        """OpenDataLoader is the primary PDF tier (before PyMuPDF)."""
        mock_available.return_value = True
        mock_extract.return_value = self._LONG_TEXT
        mock_last_attempt.return_value = None

        cascade = ExtractionCascade(
            pdf_extractor=mock_pdf_extractor,
            enable_arxiv=False,
            enable_opendataloader=True,
            enable_marker=False,
        )
        cascade.enable_opendataloader = True

        result = cascade.extract_text(Path("paper.pdf"))

        assert result.method == "opendataloader"
        assert result.is_markdown is True
        assert "opendataloader" in result.tiers_attempted
        # PyMuPDF should NOT have been called (ODL succeeded first)
        mock_pdf_extractor.extract_text_with_method.assert_not_called()

    @patch("src.extraction.cascade.opendataloader_extractor.get_last_attempt_result")
    @patch("src.extraction.cascade.opendataloader_extractor.ensure_hybrid_server")
    @patch("src.extraction.cascade.opendataloader_extractor.is_available")
    @patch("src.extraction.cascade.opendataloader_extractor.extract_with_opendataloader")
    def test_cascade_uses_hybrid_fallback_after_fast_failure(
        self,
        mock_extract,
        mock_available,
        mock_ensure_hybrid,
        mock_last_attempt,
        mock_pdf_extractor,
    ):
        """Hybrid fallback runs after fast ODL when a backend is reachable."""
        mock_available.return_value = True
        mock_ensure_hybrid.return_value = OpenDataLoaderHybridConfig(enabled=True)
        mock_extract.side_effect = [None, self._LONG_TEXT]
        mock_last_attempt.side_effect = [None, None]

        cascade = ExtractionCascade(
            pdf_extractor=mock_pdf_extractor,
            enable_arxiv=False,
            enable_opendataloader=True,
            enable_marker=False,
            opendataloader_hybrid=OpenDataLoaderHybridConfig(enabled=True),
            opendataloader_hybrid_fallback=True,
        )
        result = cascade.extract_text(Path("paper.pdf"))

        assert result.method == "opendataloader_hybrid"
        assert "opendataloader" in result.tiers_attempted
        assert "opendataloader_hybrid" in result.tiers_attempted
        mock_pdf_extractor.extract_text_with_method.assert_not_called()

    @patch("src.extraction.cascade.opendataloader_extractor.get_last_attempt_result")
    @patch("src.extraction.cascade.opendataloader_extractor.ensure_hybrid_server")
    @patch("src.extraction.cascade.opendataloader_extractor.is_available")
    @patch("src.extraction.cascade.opendataloader_extractor.extract_with_opendataloader")
    def test_cascade_downgrades_primary_hybrid_to_fast_when_backend_unavailable(
        self,
        mock_extract,
        mock_available,
        mock_ensure_hybrid,
        mock_last_attempt,
        mock_pdf_extractor,
    ):
        """Primary ODL hybrid mode downgrades to fast mode if backend is absent."""
        mock_available.return_value = True
        mock_ensure_hybrid.return_value = None
        mock_extract.return_value = self._LONG_TEXT
        mock_last_attempt.return_value = None

        cascade = ExtractionCascade(
            pdf_extractor=mock_pdf_extractor,
            enable_arxiv=False,
            enable_opendataloader=True,
            enable_marker=False,
            opendataloader_mode="hybrid",
            opendataloader_hybrid=OpenDataLoaderHybridConfig(enabled=True),
        )
        result = cascade.extract_text(Path("paper.pdf"))

        assert result.method == "opendataloader"
        assert cascade.opendataloader_mode == "fast"
        mock_pdf_extractor.extract_text_with_method.assert_not_called()

    @patch("src.extraction.cascade.opendataloader_extractor.get_last_attempt_result")
    @patch("src.extraction.cascade.opendataloader_extractor.ensure_hybrid_server")
    @patch("src.extraction.cascade.opendataloader_extractor.is_available")
    @patch("src.extraction.cascade.opendataloader_extractor.extract_with_opendataloader")
    def test_cascade_upgrades_fast_odl_to_hybrid_for_formula_docs(
        self,
        mock_extract,
        mock_available,
        mock_ensure_hybrid,
        mock_last_attempt,
        mock_pdf_extractor,
    ):
        """Fast ODL can be upgraded to hybrid when enrichment heuristics fire."""
        mock_available.return_value = True
        mock_ensure_hybrid.return_value = OpenDataLoaderHybridConfig(
            enabled=True,
            enrich_formula=True,
        )
        mock_extract.side_effect = [
            "This paper presents Equation 1 and a proof with theorem details. " * 20,
            self._LONG_TEXT,
        ]
        mock_last_attempt.side_effect = [None, None]

        cascade = ExtractionCascade(
            pdf_extractor=mock_pdf_extractor,
            enable_arxiv=False,
            enable_opendataloader=True,
            enable_marker=False,
            opendataloader_hybrid=OpenDataLoaderHybridConfig(
                enabled=True,
                enrich_formula=True,
            ),
        )
        result = cascade.extract_text(Path("paper.pdf"))

        assert result.method == "opendataloader_hybrid"
        assert "opendataloader_hybrid" in result.tiers_attempted

    @patch("src.extraction.cascade.opendataloader_extractor.get_last_attempt_result")
    @patch("src.extraction.cascade.opendataloader_extractor.is_available")
    @patch("src.extraction.cascade.opendataloader_extractor.extract_with_opendataloader")
    def test_cascade_falls_through_opendataloader_to_pymupdf(
        self,
        mock_extract,
        mock_available,
        mock_last_attempt,
        mock_pdf_extractor,
    ):
        """When OpenDataLoader fails, cascade falls through to PyMuPDF."""
        mock_available.return_value = True
        mock_extract.return_value = None  # ODL fails
        mock_last_attempt.return_value = None

        cascade = ExtractionCascade(
            pdf_extractor=mock_pdf_extractor,
            enable_arxiv=False,
            enable_opendataloader=True,
            enable_marker=False,
        )
        cascade.enable_opendataloader = True

        result = cascade.extract_text(Path("paper.pdf"))

        assert result.method == "pymupdf"
        assert "opendataloader" in result.tiers_attempted
        assert "pymupdf" in result.tiers_attempted

    @patch("src.extraction.cascade.opendataloader_extractor.get_last_attempt_result")
    @patch("src.extraction.cascade.opendataloader_extractor.is_available")
    @patch("src.extraction.cascade.opendataloader_extractor.extract_with_opendataloader")
    @patch("src.extraction.cascade.marker_extractor.is_available")
    @patch("src.extraction.cascade.marker_extractor.extract_with_marker")
    def test_cascade_falls_through_all_to_marker(
        self,
        mock_marker_extract,
        mock_marker_avail,
        mock_odl_extract,
        mock_odl_avail,
        mock_last_attempt,
        mock_pdf_extractor,
    ):
        """When ODL and PyMuPDF both fail, cascade falls through to Marker."""
        mock_odl_avail.return_value = True
        mock_odl_extract.return_value = None  # ODL fails
        mock_last_attempt.return_value = None
        mock_marker_avail.return_value = True
        mock_marker_extract.return_value = self._LONG_TEXT
        mock_pdf_extractor.extract_text_with_method.return_value = ("Short.", "pymupdf")

        cascade = ExtractionCascade(
            pdf_extractor=mock_pdf_extractor,
            enable_arxiv=False,
            enable_opendataloader=True,
            enable_marker=True,
        )
        cascade.enable_opendataloader = True
        cascade.enable_marker = True

        result = cascade.extract_text(Path("paper.pdf"))

        assert result.method == "marker"
        assert "opendataloader" in result.tiers_attempted
        assert "pymupdf" in result.tiers_attempted
        assert "marker" in result.tiers_attempted

    @patch("src.extraction.cascade.opendataloader_extractor.get_last_attempt_result")
    @patch("src.extraction.cascade.opendataloader_extractor.ensure_hybrid_server")
    @patch("src.extraction.cascade.opendataloader_extractor.is_available")
    @patch("src.extraction.cascade.opendataloader_extractor.extract_with_opendataloader")
    def test_cascade_records_hybrid_failure_diagnostics(
        self,
        mock_extract,
        mock_available,
        mock_ensure_hybrid,
        mock_last_attempt,
        mock_pdf_extractor,
    ):
        """Hybrid failures are preserved on the cascade result when fallback wins."""
        from src.extraction.opendataloader_extractor import OpenDataLoaderAttemptResult

        mock_available.return_value = True
        mock_ensure_hybrid.return_value = OpenDataLoaderHybridConfig(enabled=True)
        mock_extract.return_value = None
        mock_last_attempt.return_value = OpenDataLoaderAttemptResult(
            mode="hybrid",
            text=None,
            error="cli exit code 1",
        )

        cascade = ExtractionCascade(
            pdf_extractor=mock_pdf_extractor,
            enable_arxiv=False,
            enable_opendataloader=True,
            enable_marker=False,
            opendataloader_mode="hybrid",
            opendataloader_hybrid=OpenDataLoaderHybridConfig(enabled=True),
        )
        cascade.enable_opendataloader = True
        cascade.opendataloader_hybrid = OpenDataLoaderHybridConfig(enabled=True)
        result = cascade.extract_text(Path("paper.pdf"))

        assert result.method == "pymupdf"
        assert result.tier_errors["opendataloader_hybrid"] == "cli exit code 1"

    @patch("src.extraction.cascade.opendataloader_extractor.get_last_attempt_result")
    @patch("src.extraction.cascade.opendataloader_extractor.ensure_hybrid_server")
    @patch("src.extraction.cascade.opendataloader_extractor.is_available")
    @patch("src.extraction.cascade.opendataloader_extractor.extract_with_opendataloader")
    def test_cascade_uses_managed_pool_url_for_matching_hybrid_profile(
        self,
        mock_extract,
        mock_available,
        mock_ensure_hybrid,
        mock_last_attempt,
        mock_pdf_extractor,
    ):
        """Managed hybrid pool routing should preserve the matched fixed URL."""
        from src.extraction.opendataloader_extractor import ManagedHybridServerSpec

        managed_config = OpenDataLoaderHybridConfig(
            enabled=True,
            enrich_formula=True,
            managed_servers=(
                ManagedHybridServerSpec(name="base", url="http://127.0.0.1:5002"),
                ManagedHybridServerSpec(
                    name="formula",
                    url="http://127.0.0.1:5004",
                    enrich_formula=True,
                ),
            ),
        )
        mock_available.return_value = True
        mock_ensure_hybrid.return_value = managed_config
        mock_extract.return_value = self._LONG_TEXT
        mock_last_attempt.return_value = None

        cascade = ExtractionCascade(
            pdf_extractor=mock_pdf_extractor,
            enable_arxiv=False,
            enable_opendataloader=True,
            enable_marker=False,
            opendataloader_mode="hybrid",
            opendataloader_hybrid=managed_config,
        )
        result = cascade.extract_text(Path("paper.pdf"))

        assert result.method == "opendataloader_hybrid"
        assert cascade.opendataloader_hybrid is not None
        assert cascade.opendataloader_hybrid.resolved_server_url() == "http://127.0.0.1:5004"
        mock_extract.assert_called_once()

    @patch("src.extraction.cascade.opendataloader_extractor.is_available")
    def test_cascade_skips_opendataloader_when_unavailable(
        self,
        mock_available,
        mock_pdf_extractor,
    ):
        """OpenDataLoader tier skipped when package/Java not available."""
        mock_available.return_value = False

        cascade = ExtractionCascade(
            pdf_extractor=mock_pdf_extractor,
            enable_arxiv=False,
            enable_opendataloader=True,
            enable_marker=False,
        )
        result = cascade.extract_text(Path("paper.pdf"))

        assert result.method == "pymupdf"
        assert "opendataloader" not in result.tiers_attempted

    def test_cascade_disables_arxiv(self, mock_pdf_extractor):
        """arXiv tiers are skipped when disabled."""
        cascade = ExtractionCascade(
            pdf_extractor=mock_pdf_extractor,
            enable_arxiv=False,
            enable_opendataloader=False,
            enable_marker=False,
        )
        result = cascade.extract_text(
            Path("paper.pdf"),
            doi="10.48550/arXiv.2301.12345",
        )

        assert result.method == "pymupdf"
        assert "arxiv_html" not in result.tiers_attempted

    def test_cascade_result_includes_word_count(self, mock_pdf_extractor):
        """CascadeResult includes accurate word count."""
        cascade = ExtractionCascade(
            pdf_extractor=mock_pdf_extractor,
            enable_arxiv=False,
            enable_opendataloader=False,
            enable_marker=False,
        )
        result = cascade.extract_text(Path("paper.pdf"))

        assert result.word_count > 0
        assert result.word_count == len(result.text.split())

    def test_cascade_tracks_ocr_method(self):
        """When PDFExtractor returns OCR method, cascade tracks it."""
        mock_extractor = MagicMock()
        mock_extractor.extract_text_with_method.return_value = (
            self._LONG_TEXT,
            "ocr",
        )

        cascade = ExtractionCascade(
            pdf_extractor=mock_extractor,
            enable_arxiv=False,
            enable_opendataloader=False,
            enable_marker=False,
        )
        result = cascade.extract_text(Path("scanned.pdf"))

        assert result.method == "ocr"
        assert "ocr" in result.tiers_attempted

    def test_min_extraction_words_constant(self):
        """MIN_EXTRACTION_WORDS is a reasonable default."""
        assert MIN_EXTRACTION_WORDS == 100
