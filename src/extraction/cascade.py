"""Multi-tier PDF extraction cascade.

Orchestrates extraction across multiple backends in quality-priority order:

For arXiv papers:
  1. arXiv HTML (highest quality, cleanest text)
  2. ar5iv HTML5 rendering
  3. OpenDataLoader PDF fast mode (primary: better reading order, +4-15%
     word yield)
  4. OpenDataLoader hybrid (optional fallback for complex/scanned PDFs)
  5. PyMuPDF (fallback: instant, no Java dependency)
  6. Marker PDF-to-markdown (ML fallback for complex layouts)
  7. OCR (last resort for scanned PDFs)

For non-arXiv papers:
  1. OpenDataLoader PDF fast mode (primary: better reading order, +4-15%
     word yield)
  2. OpenDataLoader hybrid (optional fallback for complex/scanned PDFs)
  3. PyMuPDF (fallback: instant, no Java dependency)
  4. Marker PDF-to-markdown (ML fallback for complex layouts)
  5. OCR (last resort for scanned PDFs)

Each tier is tried in order; the first to produce text with sufficient
word count wins. OpenDataLoader is the primary PDF tier because it
consistently extracts more words than PyMuPDF (especially on multi-column
papers) at a modest speed cost (~1-3s/paper). PyMuPDF is kept as the
fallback for environments without Java 11+.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.extraction import arxiv_extractor, marker_extractor, opendataloader_extractor
from src.extraction.opendataloader_extractor import OpenDataLoaderHybridConfig
from src.extraction.pdf_extractor import PDFExtractionError, PDFExtractor
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

CascadeMethod = Literal[
    "companion",
    "arxiv_html",
    "ar5iv",
    "opendataloader",
    "opendataloader_hybrid",
    "marker",
    "pymupdf",
    "ocr",
    "hybrid",
]

# Minimum words for an extraction to be considered successful
MIN_EXTRACTION_WORDS = 100
FORMULA_SIGNAL = re.compile(
    r"\b(equation|formula|theorem|lemma|proof|corollary|optimization)\b",
    re.IGNORECASE,
)
PICTURE_SIGNAL = re.compile(
    r"\b(fig(?:ure)?\.?|chart|plot|diagram|image|visualization)\b",
    re.IGNORECASE,
)


@dataclass
class CascadeResult:
    """Result of cascade extraction with provenance."""

    text: str
    method: CascadeMethod
    word_count: int
    tiers_attempted: list[str]
    is_markdown: bool = False
    tier_errors: dict[str, str] = field(default_factory=dict)


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
        enable_opendataloader: bool = True,
        enable_marker: bool = True,
        arxiv_timeout: int = 30,
        min_words: int = MIN_EXTRACTION_WORDS,
        companion_dir: Path | None = None,
        opendataloader_mode: str = "fast",
        opendataloader_hybrid: OpenDataLoaderHybridConfig | None = None,
        opendataloader_hybrid_fallback: bool = False,
    ):
        """Initialize cascade.

        Args:
            pdf_extractor: Existing PDFExtractor instance (handles PyMuPDF + OCR).
            enable_arxiv: Enable arXiv HTML and ar5iv tiers.
            enable_opendataloader: Enable OpenDataLoader PDF tier.
            enable_marker: Enable Marker PDF-to-markdown tier.
            arxiv_timeout: Timeout for arXiv/ar5iv HTTP requests.
            min_words: Minimum word count to accept an extraction.
            companion_dir: Optional directory for pre-extracted .md files.
            opendataloader_mode: OpenDataLoader primary mode ("fast" or
                "hybrid").
            opendataloader_hybrid: Optional hybrid backend configuration.
            opendataloader_hybrid_fallback: Try hybrid after fast ODL yields
                insufficient text.
        """
        self.pdf_extractor = pdf_extractor
        self.enable_arxiv = enable_arxiv
        self.enable_opendataloader = (
            enable_opendataloader and opendataloader_extractor.is_available()
        )
        self.enable_marker = enable_marker and marker_extractor.is_available()
        self.arxiv_timeout = arxiv_timeout
        self.min_words = min_words
        self.companion_dir = companion_dir
        self.opendataloader_mode = opendataloader_mode
        self.opendataloader_hybrid = None
        self.enable_opendataloader_hybrid_fallback = False

        if self.enable_opendataloader and (
            opendataloader_mode == "hybrid"
            or opendataloader_hybrid_fallback
            or opendataloader_hybrid is not None
        ):
            requested_hybrid = opendataloader_hybrid or OpenDataLoaderHybridConfig(enabled=True)
            self.opendataloader_hybrid = opendataloader_extractor.ensure_hybrid_server(
                requested_hybrid
            )
            if self.opendataloader_hybrid is None:
                if opendataloader_mode == "hybrid":
                    logger.info(
                        "OpenDataLoader hybrid requested but backend unavailable; "
                        "downgrading to fast mode"
                    )
                    self.opendataloader_mode = "fast"
                elif opendataloader_hybrid_fallback:
                    logger.info(
                        "OpenDataLoader hybrid fallback requested but backend "
                        "unavailable; skipping hybrid fallback"
                    )
            elif opendataloader_hybrid_fallback and opendataloader_mode != "hybrid":
                self.enable_opendataloader_hybrid_fallback = True

        if enable_opendataloader and not opendataloader_extractor.is_available():
            logger.info(
                "OpenDataLoader PDF not available (missing Java 11+ or package); "
                "OpenDataLoader tier disabled"
            )
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
        tier_errors: dict[str, str] = {}

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
                        tier_errors=tier_errors,
                    )
                else:
                    logger.debug(f"Companion file {companion_path.name} has insufficient text")
            except OSError as e:
                logger.warning(f"Failed to read companion file: {e}")

        # Detect arXiv ID for arXiv-specific tiers
        arxiv_id = None
        if self.enable_arxiv:
            arxiv_id = arxiv_extractor.detect_arxiv_id(
                doi=doi,
                url=url,
                pdf_path=pdf_path,
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
                    tier_errors=tier_errors,
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
                    tier_errors=tier_errors,
                )

        # Tier 3: OpenDataLoader PDF (primary PDF tier -- better reading order)
        if self.enable_opendataloader:
            odl_tier: CascadeMethod = (
                "opendataloader_hybrid"
                if self.opendataloader_mode == "hybrid"
                else "opendataloader"
            )
            tiers_attempted.append(odl_tier)
            text = opendataloader_extractor.extract_with_opendataloader(
                pdf_path,
                mode=self.opendataloader_mode,
                hybrid_config=self.opendataloader_hybrid,
            )
            attempt = opendataloader_extractor.get_last_attempt_result()
            if attempt and attempt.error:
                tier_errors[odl_tier] = attempt.error
            if text and self._is_sufficient(text):
                if (
                    odl_tier == "opendataloader"
                    and self.opendataloader_hybrid is not None
                    and self._should_upgrade_fast_odl_to_hybrid(text)
                ):
                    tiers_attempted.append("opendataloader_hybrid")
                    hybrid_text = opendataloader_extractor.extract_with_opendataloader(
                        pdf_path,
                        mode="hybrid",
                        hybrid_config=self.opendataloader_hybrid,
                    )
                    hybrid_attempt = opendataloader_extractor.get_last_attempt_result()
                    if hybrid_attempt and hybrid_attempt.error:
                        tier_errors["opendataloader_hybrid"] = hybrid_attempt.error
                    if hybrid_text and self._is_sufficient(hybrid_text):
                        return CascadeResult(
                            text=hybrid_text,
                            method="opendataloader_hybrid",
                            word_count=len(hybrid_text.split()),
                            tiers_attempted=tiers_attempted,
                            is_markdown=True,
                            tier_errors=tier_errors,
                        )
                return CascadeResult(
                    text=text,
                    method=odl_tier,
                    word_count=len(text.split()),
                    tiers_attempted=tiers_attempted,
                    is_markdown=True,
                    tier_errors=tier_errors,
                )

        # Tier 4: OpenDataLoader hybrid fallback for complex/scanned PDFs.
        if self.enable_opendataloader_hybrid_fallback and self.opendataloader_hybrid:
            tiers_attempted.append("opendataloader_hybrid")
            text = opendataloader_extractor.extract_with_opendataloader(
                pdf_path,
                mode="hybrid",
                hybrid_config=self.opendataloader_hybrid,
            )
            attempt = opendataloader_extractor.get_last_attempt_result()
            if attempt and attempt.error:
                tier_errors["opendataloader_hybrid"] = attempt.error
            if text and self._is_sufficient(text):
                return CascadeResult(
                    text=text,
                    method="opendataloader_hybrid",
                    word_count=len(text.split()),
                    tiers_attempted=tiers_attempted,
                    is_markdown=True,
                    tier_errors=tier_errors,
                )

        # Tier 5: PyMuPDF (fallback for when ODL/Java unavailable)
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
                    tier_errors=tier_errors,
                )
        except PDFExtractionError:
            pass

        # Tier 6: Marker (ML fallback for complex layouts, scanned docs)
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
                    tier_errors=tier_errors,
                )

        diagnostics = ""
        if tier_errors:
            diagnostics = " Diagnostics: " + "; ".join(
                f"{tier}={error}" for tier, error in sorted(tier_errors.items())
            )
        raise PDFExtractionError(
            f"All extraction tiers failed for {pdf_path.name}. "
            f"Tiers attempted: {', '.join(tiers_attempted)}.{diagnostics}"
        )

    def _should_upgrade_fast_odl_to_hybrid(self, text: str) -> bool:
        """Return True when fast ODL output suggests hybrid enrichments are useful."""
        if self.opendataloader_hybrid is None:
            return False

        lowered = text.lower()
        wants_formula = (
            self.opendataloader_hybrid.enrich_formula and FORMULA_SIGNAL.search(lowered) is not None
        )
        wants_picture = (
            self.opendataloader_hybrid.enrich_picture_description
            and PICTURE_SIGNAL.search(lowered) is not None
        )
        return wants_formula or wants_picture

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
