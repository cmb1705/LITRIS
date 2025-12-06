"""PDF text extraction using PyMuPDF."""

import hashlib
from pathlib import Path

import pymupdf

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class PDFExtractionError(Exception):
    """Error during PDF extraction."""

    pass


class PDFExtractor:
    """Extract text from PDF files using PyMuPDF."""

    def __init__(self, cache_dir: Path | None = None):
        """Initialize extractor.

        Args:
            cache_dir: Directory for caching extracted text.
        """
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

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
        if not pdf_path.exists():
            raise PDFExtractionError(f"PDF file not found: {pdf_path}")

        # Check cache
        if use_cache and self.cache_dir:
            cached = self._get_cached(pdf_path)
            if cached is not None:
                logger.debug(f"Using cached text for {pdf_path.name}")
                return cached

        # Extract text
        try:
            text = self._extract_with_pymupdf(pdf_path)
        except Exception as e:
            raise PDFExtractionError(f"Failed to extract text from {pdf_path}: {e}")

        # Cache result
        if use_cache and self.cache_dir:
            self._save_to_cache(pdf_path, text)

        return text

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
        path_hash = hashlib.md5(str(pdf_path.absolute()).encode()).hexdigest()[:16]
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
