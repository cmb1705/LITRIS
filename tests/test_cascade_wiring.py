"""Tests for extraction cascade wiring and companion tier."""

from unittest.mock import MagicMock, patch

from src.extraction.cascade import CascadeResult, ExtractionCascade
from src.extraction.opendataloader_extractor import OpenDataLoaderHybridConfig
from src.extraction.pdf_extractor import PDFExtractor


class TestCascadeResultDataclass:
    def test_cascade_result_has_is_markdown(self):
        """CascadeResult includes is_markdown field defaulting to False."""
        result = CascadeResult(
            text="hello world " * 50,
            method="pymupdf",
            word_count=100,
            tiers_attempted=["pymupdf"],
        )
        assert result.is_markdown is False

    def test_cascade_result_is_markdown_true(self):
        """CascadeResult can set is_markdown=True."""
        result = CascadeResult(
            text="# Title\n\nBody",
            method="companion",
            word_count=2,
            tiers_attempted=["companion"],
            is_markdown=True,
        )
        assert result.is_markdown is True
        assert result.method == "companion"

    def test_companion_is_valid_cascade_method(self):
        """'companion' is a valid CascadeMethod literal value."""
        result = CascadeResult(
            text="test",
            method="companion",
            word_count=1,
            tiers_attempted=["companion"],
            is_markdown=True,
        )
        assert result.method == "companion"


class TestCompanionTier:
    def test_companion_tier_finds_md_file(self, tmp_path):
        """Companion tier reads .md file alongside PDF."""
        pdf = tmp_path / "paper.pdf"
        pdf.write_text("dummy pdf")
        md = tmp_path / "paper.md"
        md.write_text("# Title\n\n" + "Body text. " * 50)

        pdf_extractor = MagicMock(spec=PDFExtractor)
        cascade = ExtractionCascade(pdf_extractor, companion_dir=None)

        result = cascade.extract_text(pdf)
        assert result.method == "companion"
        assert result.is_markdown is True
        assert "Body text" in result.text
        pdf_extractor.extract_text_with_method.assert_not_called()

    def test_companion_tier_checks_companion_dir(self, tmp_path):
        """Companion tier checks configured companion_dir."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        pdf = pdf_dir / "paper.pdf"
        pdf.write_text("dummy pdf")

        comp_dir = tmp_path / "companions"
        comp_dir.mkdir()
        md = comp_dir / "paper.md"
        md.write_text("# Title\n\n" + "Content here. " * 50)

        pdf_extractor = MagicMock(spec=PDFExtractor)
        cascade = ExtractionCascade(pdf_extractor, companion_dir=comp_dir)

        result = cascade.extract_text(pdf)
        assert result.method == "companion"
        assert result.is_markdown is True

    def test_companion_tier_skips_insufficient_text(self, tmp_path):
        """Companion tier falls through when .md has too few words."""
        pdf = tmp_path / "paper.pdf"
        pdf.write_text("dummy pdf")
        md = tmp_path / "paper.md"
        md.write_text("Short.")  # Below min_words threshold

        pdf_extractor = MagicMock(spec=PDFExtractor)
        pdf_extractor.extract_text_with_method.return_value = (
            "Fallback text. " * 50,
            "pymupdf",
        )

        cascade = ExtractionCascade(pdf_extractor, companion_dir=None)
        result = cascade.extract_text(pdf)
        assert result.method == "pymupdf"
        assert "companion" in result.tiers_attempted

    def test_companion_tier_not_found_falls_through(self, tmp_path):
        """No companion .md file -- cascade falls through to next tier."""
        pdf = tmp_path / "paper.pdf"
        pdf.write_text("dummy pdf")

        pdf_extractor = MagicMock(spec=PDFExtractor)
        pdf_extractor.extract_text_with_method.return_value = (
            "Extracted text. " * 50,
            "pymupdf",
        )

        cascade = ExtractionCascade(
            pdf_extractor, enable_arxiv=False, enable_marker=False,
        )
        result = cascade.extract_text(pdf)
        assert result.method == "pymupdf"
        assert "companion" not in result.tiers_attempted


class TestSectionExtractorCascadeWiring:
    @patch("src.analysis.section_extractor.create_llm_client")
    def test_section_extractor_creates_cascade_when_enabled(self, mock_factory):
        """SectionExtractor creates ExtractionCascade when cascade_enabled."""
        from src.analysis.section_extractor import SectionExtractor

        mock_factory.return_value = MagicMock()
        extractor = SectionExtractor(cascade_enabled=True)
        assert hasattr(extractor, "cascade")
        assert extractor.cascade is not None

    @patch("src.analysis.section_extractor.create_llm_client")
    def test_section_extractor_no_cascade_when_disabled(self, mock_factory):
        """SectionExtractor uses PDFExtractor directly when cascade disabled."""
        from src.analysis.section_extractor import SectionExtractor

        mock_factory.return_value = MagicMock()
        extractor = SectionExtractor(cascade_enabled=False)
        assert extractor.cascade is None

    @patch("src.analysis.section_extractor.create_llm_client")
    def test_section_extractor_accepts_hybrid_config(self, mock_factory):
        """SectionExtractor accepts hybrid configuration for cascade wiring."""
        from src.analysis.section_extractor import SectionExtractor

        mock_factory.return_value = MagicMock()
        extractor = SectionExtractor(
            cascade_enabled=True,
            opendataloader_hybrid_config=OpenDataLoaderHybridConfig(enabled=True),
            opendataloader_hybrid_fallback=True,
        )
        assert extractor.cascade is not None
