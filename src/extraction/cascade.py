"""Multi-tier PDF extraction cascade.

Orchestrates extraction across multiple backends in speed-priority order,
reserving heavy ML tiers (Marker) for cases where fast extraction fails:

For arXiv papers:
  1. arXiv HTML (highest quality, cleanest text)
  2. ar5iv HTML5 rendering
  3. PyMuPDF (fast text extraction)
  4. Marker PDF-to-markdown (ML fallback for complex layouts)
  5. OCR (last resort for scanned PDFs)

For non-arXiv papers:
  1. PyMuPDF (fast text extraction)
  2. Marker PDF-to-markdown (ML fallback for complex layouts)
  3. OCR (last resort for scanned PDFs)

Each tier is tried in order; the first to produce text with sufficient
word count wins. Marker is intentionally placed after PyMuPDF because
it runs ML models on every page (10-30s/page) and is only needed when
PyMuPDF cannot extract text (scanned PDFs, image-heavy layouts).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.extraction import arxiv_extractor, marker_extractor
from src.extraction.pdf_extractor import PDFExtractionError, PDFExtractor
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

CascadeMethod = Literal[
    "companion",
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
    is_markdown: bool = False


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
        companion_dir: Path | None = None,
    ):
        """Initialize cascade.

        Args:
            pdf_extractor: Existing PDFExtractor instance (handles PyMuPDF + OCR).
            enable_arxiv: Enable arXiv HTML and ar5iv tiers.
            enable_marker: Enable Marker PDF-to-markdown tier.
            arxiv_timeout: Timeout for arXiv/ar5iv HTTP requests.
            min_words: Minimum word count to accept an extraction.
            companion_dir: Optional directory for pre-extracted .md files.
        """
        self.pdf_extractor = pdf_extractor
        self.enable_arxiv = enable_arxiv
        self.enable_marker = enable_marker and marker_extractor.is_available()
        self.arxiv_timeout = arxiv_timeout
        self.min_words = min_words
        self.companion_dir = companion_dir

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

        # Tier 0 (Priority 1): Companion .md file
        companion_path = self._find_companion(pdf_path)
        if companion_path:
            tiers_attempted.append("companion")
            try:
                text = companion_path.read_text(encoding="utf-8")
                if self._is_sufficient(text):
                    logger.info(f"Companion file found: {companion_path.name}")
                    return CascadeResult(
                        text=text,
                        method="companion",
                        word_count=len(text.split()),
                        tiers_attempted=tiers_attempted,
                        is_markdown=True,
                    )
                else:
                    logger.debug(
                        f"Companion file {companion_path.name} has insufficient text"
                    )
            except OSError as e:
                logger.warning(f"Failed to read companion file: {e}")

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

        # Tier 3: PyMuPDF (fast, handles most PDFs with embedded text)
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

        # Tier 4: Marker (ML fallback for complex layouts, scanned docs)
        if self.enable_marker:
            tiers_attempted.append("marker")
            text = marker_extractor.extract_with_marker(pdf_path)
            if text and self._is_sufficient(text):
                return CascadeResult(
                    text=text,
                    method="marker",
                    word_count=len(text.split()),
                    tiers_attempted=tiers_attempted,
                    is_markdown=True,
                )

        raise PDFExtractionError(
            f"All extraction tiers failed for {pdf_path.name}. "
            f"Tiers attempted: {', '.join(tiers_attempted)}"
        )

    def _find_companion(self, pdf_path: Path) -> Path | None:
        """Find a companion .md file for the given PDF.

        Checks:
        1. Same directory as PDF with .md extension
        2. companion_dir (if configured) with same stem

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Path to companion .md file, or None if not found.
        """
        # Check alongside PDF
        md_path = pdf_path.with_suffix(".md")
        if md_path.is_file():
            return md_path

        # Check companion directory
        if self.companion_dir and self.companion_dir.is_dir():
            md_path = self.companion_dir / f"{pdf_path.stem}.md"
            if md_path.is_file():
                return md_path

        return None

    def _is_sufficient(self, text: str) -> bool:
        """Check if extracted text meets minimum quality threshold.

        Args:
            text: Extracted text to check.

        Returns:
            True if text has enough words.
        """
        return len(text.split()) >= self.min_words
