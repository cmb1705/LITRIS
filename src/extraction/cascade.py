"""Multi-tier PDF extraction cascade.

Orchestrates extraction across multiple backends in quality-priority order:

For arXiv papers:
  1. arXiv HTML (highest quality, cleanest text)
  2. ar5iv HTML5 rendering
  3. Marker PDF-to-markdown (if installed)
  4. PyMuPDF (standard text extraction)
  5. OCR (fallback for scanned PDFs)

For non-arXiv papers:
  1. Marker PDF-to-markdown (if installed)
  2. PyMuPDF (standard text extraction)
  3. OCR (fallback for scanned PDFs)

Each tier is tried in order; the first to produce text with sufficient
word count wins. This minimizes API calls and external dependencies
while maximizing extraction quality.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.extraction import arxiv_extractor, marker_extractor
from src.extraction.pdf_extractor import PDFExtractionError, PDFExtractor
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

CascadeMethod = Literal[
    "arxiv_html",
    "ar5iv",
    "marker",
    "pymupdf",
    "ocr",
    "hybrid",
]

# Minimum words for an extraction to be considered successful
MIN_EXTRACTION_WORDS = 100


@dataclass
class CascadeResult:
    """Result of cascade extraction with provenance."""

    text: str
    method: CascadeMethod
    word_count: int
    tiers_attempted: list[str]


class ExtractionCascade:
    """Multi-tier PDF text extraction with quality-ordered fallbacks.

    Wraps PDFExtractor and adds arXiv HTML and Marker tiers on top.
    Can be used as a drop-in replacement where PDFExtractor.extract_text()
    is currently called.
    """

    def __init__(
        self,
        pdf_extractor: PDFExtractor,
        enable_arxiv: bool = True,
        enable_marker: bool = True,
        arxiv_timeout: int = 30,
        min_words: int = MIN_EXTRACTION_WORDS,
    ):
        """Initialize cascade.

        Args:
            pdf_extractor: Existing PDFExtractor instance (handles PyMuPDF + OCR).
            enable_arxiv: Enable arXiv HTML and ar5iv tiers.
            enable_marker: Enable Marker PDF-to-markdown tier.
            arxiv_timeout: Timeout for arXiv/ar5iv HTTP requests.
            min_words: Minimum word count to accept an extraction.
        """
        self.pdf_extractor = pdf_extractor
        self.enable_arxiv = enable_arxiv
        self.enable_marker = enable_marker and marker_extractor.is_available()
        self.arxiv_timeout = arxiv_timeout
        self.min_words = min_words

        if enable_marker and not marker_extractor.is_available():
            logger.info("Marker not installed; Marker tier disabled")

    def extract_text(
        self,
        pdf_path: Path,
        doi: str | None = None,
        url: str | None = None,
    ) -> CascadeResult:
        """Extract text using the cascade.

        Tries each tier in quality order until one succeeds.

        Args:
            pdf_path: Path to PDF file.
            doi: Paper DOI (used for arXiv detection).
            url: Paper URL (used for arXiv detection).

        Returns:
            CascadeResult with extracted text and provenance.

        Raises:
            PDFExtractionError: If all tiers fail.
        """
        tiers_attempted: list[str] = []

        # Detect arXiv ID for arXiv-specific tiers
        arxiv_id = None
        if self.enable_arxiv:
            arxiv_id = arxiv_extractor.detect_arxiv_id(
                doi=doi, url=url, pdf_path=pdf_path,
            )
            if arxiv_id:
                logger.info(f"Detected arXiv paper: {arxiv_id}")

        # Tier 1: arXiv HTML
        if arxiv_id:
            tiers_attempted.append("arxiv_html")
            text = arxiv_extractor.fetch_arxiv_html(arxiv_id, timeout=self.arxiv_timeout)
            if text and self._is_sufficient(text):
                return CascadeResult(
                    text=text,
                    method="arxiv_html",
                    word_count=len(text.split()),
                    tiers_attempted=tiers_attempted,
                )

        # Tier 2: ar5iv
        if arxiv_id:
            tiers_attempted.append("ar5iv")
            text = arxiv_extractor.fetch_ar5iv_html(arxiv_id, timeout=self.arxiv_timeout)
            if text and self._is_sufficient(text):
                return CascadeResult(
                    text=text,
                    method="ar5iv",
                    word_count=len(text.split()),
                    tiers_attempted=tiers_attempted,
                )

        # Tier 3: Marker
        if self.enable_marker:
            tiers_attempted.append("marker")
            text = marker_extractor.extract_with_marker(pdf_path)
            if text and self._is_sufficient(text):
                return CascadeResult(
                    text=text,
                    method="marker",
                    word_count=len(text.split()),
                    tiers_attempted=tiers_attempted,
                )

        # Tier 4-5: PyMuPDF + OCR (via existing PDFExtractor)
        tiers_attempted.append("pymupdf")
        try:
            text, method = self.pdf_extractor.extract_text_with_method(pdf_path)
            cascade_method: CascadeMethod = "pymupdf"
            if method in ("ocr", "hybrid"):
                tiers_attempted.append(method)
                cascade_method = method

            if text and self._is_sufficient(text):
                return CascadeResult(
                    text=text,
                    method=cascade_method,
                    word_count=len(text.split()),
                    tiers_attempted=tiers_attempted,
                )
        except PDFExtractionError:
            pass

        raise PDFExtractionError(
            f"All extraction tiers failed for {pdf_path.name}. "
            f"Tiers attempted: {', '.join(tiers_attempted)}"
        )

    def _is_sufficient(self, text: str) -> bool:
        """Check if extracted text meets minimum quality threshold.

        Args:
            text: Extracted text to check.

        Returns:
            True if text has enough words.
        """
        return len(text.split()) >= self.min_words
