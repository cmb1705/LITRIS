"""PDF text extraction using PyMuPDF with optional OCR fallback."""

import hashlib
from pathlib import Path
from typing import Literal

import pymupdf

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

ExtractionMethod = Literal["pymupdf", "ocr", "hybrid"]


class PDFExtractionError(Exception):
    """Error during PDF extraction."""

    pass


class PDFExtractor:
    """Extract text from PDF files using PyMuPDF with optional OCR fallback."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        enable_ocr: bool = False,
        ocr_config: dict | None = None,
    ):
        """Initialize extractor.

        Args:
            cache_dir: Directory for caching extracted text.
            enable_ocr: Enable OCR fallback for scanned PDFs.
            ocr_config: Configuration dict for OCR handler (tesseract_cmd, poppler_path, dpi, lang).
        """
        self.cache_dir = cache_dir
        self.enable_ocr = enable_ocr
        self.ocr_handler = None

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        if enable_ocr:
            from src.extraction.ocr_handler import get_ocr_handler

            self.ocr_handler = get_ocr_handler(**(ocr_config or {}))
            if self.ocr_handler:
                logger.info("OCR fallback enabled")
            else:
                logger.warning("OCR requested but dependencies unavailable")

    def extract_text(
        self,
        pdf_path: Path,
        use_cache: bool = True,
    ) -> str:
        """Extract text from a PDF file.

        Args:
            pdf_path: Path to PDF file.
            use_cache: Whether to use cached extraction if available.

        Returns:
            Extracted text content.

        Raises:
            PDFExtractionError: If extraction fails.
        """
        text, _ = self.extract_text_with_method(pdf_path, use_cache)
        return text

    def extract_text_with_method(
        self,
        pdf_path: Path,
        use_cache: bool = True,
    ) -> tuple[str, ExtractionMethod]:
        """Extract text from a PDF file with method information.

        Args:
            pdf_path: Path to PDF file.
            use_cache: Whether to use cached extraction if available.

        Returns:
            Tuple of (extracted text, extraction method used).

        Raises:
            PDFExtractionError: If extraction fails.
        """
        if not pdf_path.exists():
            raise PDFExtractionError(f"PDF file not found: {pdf_path}")

        # Check cache
        if use_cache and self.cache_dir:
            cached = self._get_cached(pdf_path)
            if cached is not None:
                logger.debug(f"Using cached text for {pdf_path.name}")
                # Cache doesn't track method, assume pymupdf
                return cached, "pymupdf"

        # Extract text with PyMuPDF first
        try:
            text = self._extract_with_pymupdf(pdf_path)
            page_count = self.get_page_count(pdf_path)
            method: ExtractionMethod = "pymupdf"
        except Exception as e:
            # If PyMuPDF fails completely, try OCR if available
            if self.ocr_handler:
                logger.warning(f"PyMuPDF failed, attempting OCR: {e}")
                try:
                    result = self.ocr_handler.extract_text(pdf_path)
                    text = result.text
                    method = "ocr"
                except Exception as ocr_error:
                    raise PDFExtractionError(
                        f"Both PyMuPDF and OCR failed for {pdf_path}: "
                        f"PyMuPDF: {e}, OCR: {ocr_error}"
                    )
            else:
                raise PDFExtractionError(f"Failed to extract text from {pdf_path}: {e}")
        else:
            # Check if OCR fallback is needed
            if self.ocr_handler and self.ocr_handler.needs_ocr(text, page_count):
                logger.info(f"Text quality low, attempting OCR for {pdf_path.name}")
                try:
                    result = self.ocr_handler.extract_hybrid(pdf_path, text, page_count)
                    text = result.text
                    method = result.method
                except Exception as ocr_error:
                    logger.warning(f"OCR fallback failed, using original text: {ocr_error}")
                    # Keep original text and method

        # Cache result
        if use_cache and self.cache_dir:
            self._save_to_cache(pdf_path, text)

        return text, method

    def _extract_with_pymupdf(self, pdf_path: Path) -> str:
        """Extract text using PyMuPDF.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Extracted text content.
        """
        doc = pymupdf.open(pdf_path)
        try:
            pages = []
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                if text.strip():
                    pages.append(f"--- Page {page_num} ---\n{text}")

            return "\n\n".join(pages)
        finally:
            doc.close()

    def extract_with_metadata(self, pdf_path: Path) -> dict:
        """Extract text and metadata from a PDF.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Dictionary with text, page_count, and metadata.
        """
        if not pdf_path.exists():
            raise PDFExtractionError(f"PDF file not found: {pdf_path}")

        doc = pymupdf.open(pdf_path)
        try:
            pages = []
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                if text.strip():
                    pages.append(f"--- Page {page_num} ---\n{text}")

            return {
                "text": "\n\n".join(pages),
                "page_count": len(doc),
                "metadata": doc.metadata,
                "file_size": pdf_path.stat().st_size,
            }
        finally:
            doc.close()

    def get_page_count(self, pdf_path: Path) -> int:
        """Get page count without full extraction.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Number of pages.
        """
        doc = pymupdf.open(pdf_path)
        try:
            return len(doc)
        finally:
            doc.close()

    def _get_cache_path(self, pdf_path: Path) -> Path:
        """Get cache file path for a PDF.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Path to cache file.
        """
        # Use hash of absolute path to create unique cache key
        path_hash = hashlib.sha256(str(pdf_path.absolute()).encode()).hexdigest()[:16]
        return self.cache_dir / f"{path_hash}.txt"

    def _get_cached(self, pdf_path: Path) -> str | None:
        """Get cached extraction if valid.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Cached text or None if not cached or stale.
        """
        if not self.cache_dir:
            return None

        cache_path = self._get_cache_path(pdf_path)
        if not cache_path.exists():
            return None

        # Check if cache is stale (PDF modified after cache)
        pdf_mtime = pdf_path.stat().st_mtime
        cache_mtime = cache_path.stat().st_mtime
        if pdf_mtime > cache_mtime:
            logger.debug(f"Cache stale for {pdf_path.name}")
            return None

        try:
            return cache_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to read cache for {pdf_path.name}: {e}")
            return None

    def _save_to_cache(self, pdf_path: Path, text: str) -> None:
        """Save extracted text to cache.

        Args:
            pdf_path: Path to PDF file.
            text: Extracted text content.
        """
        if not self.cache_dir:
            return

        cache_path = self._get_cache_path(pdf_path)
        try:
            cache_path.write_text(text, encoding="utf-8")
            logger.debug(f"Cached text for {pdf_path.name}")
        except Exception as e:
            logger.warning(f"Failed to cache text for {pdf_path.name}: {e}")

    def clear_cache(self) -> int:
        """Clear all cached extractions.

        Returns:
            Number of cache files removed.
        """
        if not self.cache_dir or not self.cache_dir.exists():
            return 0

        count = 0
        for cache_file in self.cache_dir.glob("*.txt"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")

        logger.info(f"Cleared {count} cached extractions")
        return count
