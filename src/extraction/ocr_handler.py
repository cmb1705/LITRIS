"""OCR handler for scanned PDF extraction using Tesseract."""

import os
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _find_tesseract() -> str | None:
    """Find tesseract executable in PATH or common locations.

    Searches standard PATH first, then platform-specific default locations:
    - Windows: Program Files, Chocolatey, Scoop
    - macOS: Homebrew (Intel and Apple Silicon), MacPorts
    - Linux: apt/dpkg, snap, manual /usr/local installs
    """
    # Check PATH first
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        return tesseract_path

    common_paths: list[str] = []

    if sys.platform == "win32":
        # Windows installation paths
        common_paths = [
            # Official installer default
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            # User-local install
            os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
            # Chocolatey
            r"C:\ProgramData\chocolatey\bin\tesseract.exe",
            # Scoop
            os.path.expandvars(r"%USERPROFILE%\scoop\apps\tesseract\current\tesseract.exe"),
            # winget typical location
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\tesseract-ocr.tesseract_*\tesseract.exe"),
        ]

    elif sys.platform == "darwin":
        # macOS installation paths
        common_paths = [
            # Homebrew Apple Silicon (M1/M2/M3)
            "/opt/homebrew/bin/tesseract",
            # Homebrew Intel
            "/usr/local/bin/tesseract",
            # MacPorts
            "/opt/local/bin/tesseract",
            # Manual install
            "/usr/local/Cellar/tesseract/*/bin/tesseract",  # Homebrew Cellar
        ]

    else:
        # Linux installation paths
        common_paths = [
            # apt/dpkg (Debian, Ubuntu)
            "/usr/bin/tesseract",
            # Manual /usr/local install
            "/usr/local/bin/tesseract",
            # Snap
            "/snap/bin/tesseract",
            # Flatpak (less common for CLI tools)
            os.path.expanduser("~/.local/bin/tesseract"),
            # AppImage extracted
            "/opt/tesseract/tesseract",
        ]

    for path in common_paths:
        # Handle glob patterns (for version-specific paths)
        if "*" in path:
            import glob
            matches = glob.glob(path)
            for match in matches:
                if os.path.isfile(match) and os.access(match, os.X_OK):
                    return match
        elif os.path.exists(path):
            return path

    return None


def _find_poppler() -> str | None:
    """Find poppler bin directory for pdf2image.

    Poppler is required by pdf2image to convert PDFs to images.
    On Linux, it's usually installed via apt and found automatically.
    On Windows/macOS, we need to locate the bin directory.
    """
    # Check if pdftoppm (poppler utility) is in PATH
    if shutil.which("pdftoppm"):
        # pdf2image can find it, no need to specify path
        return None

    common_paths: list[str] = []

    if sys.platform == "win32":
        # Windows poppler paths
        common_paths = [
            # Manual download from GitHub releases
            r"C:\Program Files\poppler\Library\bin",
            r"C:\Program Files\poppler\bin",
            r"C:\Program Files (x86)\poppler\Library\bin",
            # Chocolatey
            r"C:\ProgramData\chocolatey\lib\poppler\tools\Library\bin",
            # Scoop
            os.path.expandvars(r"%USERPROFILE%\scoop\apps\poppler\current\Library\bin"),
            # Common manual install locations
            r"C:\poppler\Library\bin",
            r"C:\poppler\bin",
            os.path.expandvars(r"%LOCALAPPDATA%\poppler\Library\bin"),
            # WinGet package location
            os.path.expandvars(
                r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\*Poppler*\poppler-*\Library\bin"
            ),
        ]

    elif sys.platform == "darwin":
        # macOS poppler paths (usually handled by Homebrew)
        common_paths = [
            # Homebrew Apple Silicon
            "/opt/homebrew/bin",
            # Homebrew Intel
            "/usr/local/bin",
            # MacPorts
            "/opt/local/bin",
        ]

    # Linux usually has poppler in PATH via apt, so no need for special paths

    for path in common_paths:
        pdftoppm = os.path.join(path, "pdftoppm.exe" if sys.platform == "win32" else "pdftoppm")
        if "*" in path:
            import glob
            matches = glob.glob(path)
            for match in matches:
                candidate = os.path.join(
                    match, "pdftoppm.exe" if sys.platform == "win32" else "pdftoppm"
                )
                if os.path.exists(candidate):
                    return match
        elif os.path.exists(pdftoppm):
            return path

    return None


# Auto-discover poppler path
_poppler_path = _find_poppler()
if _poppler_path:
    logger.debug(f"Poppler found at: {_poppler_path}")


# Check for optional dependencies
try:
    import pytesseract
    from PIL import Image

    _tesseract_cmd = _find_tesseract()
    TESSERACT_AVAILABLE = _tesseract_cmd is not None
    if _tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
        logger.debug(f"Tesseract found at: {_tesseract_cmd}")
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None
    Image = None

try:
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError

    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    convert_from_path = None


class OCRResult(NamedTuple):
    """Result of OCR extraction."""

    text: str
    pages_processed: int
    method: str  # "ocr" or "hybrid"


class OCRError(Exception):
    """Error during OCR processing."""

    pass


class OCRHandler:
    """Handle OCR extraction for scanned PDFs.

    Requires:
    - Tesseract OCR installed and in PATH
    - Poppler installed (for pdf2image on Windows)
    """

    # Minimum words per page to consider text extraction successful
    MIN_WORDS_PER_PAGE = 50

    # Minimum ratio of pages with text to skip OCR
    MIN_TEXT_PAGE_RATIO = 0.5

    def __init__(
        self,
        tesseract_cmd: str | None = None,
        poppler_path: str | None = None,
        dpi: int = 300,
        lang: str = "eng",
    ):
        """Initialize OCR handler.

        Args:
            tesseract_cmd: Path to tesseract executable (auto-detected if None).
            poppler_path: Path to poppler bin directory (auto-detected if None).
            dpi: DPI for PDF to image conversion.
            lang: Tesseract language code.
        """
        self.dpi = dpi
        self.lang = lang
        # Use auto-discovered poppler path if none provided
        self.poppler_path = poppler_path or _poppler_path

        if tesseract_cmd and pytesseract:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    @classmethod
    def is_available(cls) -> bool:
        """Check if OCR dependencies are available.

        Returns:
            True if Tesseract and pdf2image are available.
        """
        return TESSERACT_AVAILABLE and PDF2IMAGE_AVAILABLE

    @classmethod
    def check_dependencies(cls) -> dict[str, bool | str | None]:
        """Check status of OCR dependencies.

        Returns:
            Dictionary with dependency status and discovered paths.
        """
        deps: dict[str, bool | str | None] = {
            "pytesseract_installed": pytesseract is not None,
            "tesseract_available": TESSERACT_AVAILABLE,
            "tesseract_path": _tesseract_cmd if TESSERACT_AVAILABLE else None,
            "pdf2image_installed": PDF2IMAGE_AVAILABLE,
            "pillow_installed": Image is not None,
            "poppler_path": _poppler_path,
        }

        # Check poppler availability
        if PDF2IMAGE_AVAILABLE:
            # Poppler is available if pdftoppm is in PATH or we found a path
            deps["poppler_available"] = (
                shutil.which("pdftoppm") is not None or _poppler_path is not None
            )
        else:
            deps["poppler_available"] = False

        return deps

    def needs_ocr(self, extracted_text: str, page_count: int) -> bool:
        """Determine if a PDF needs OCR based on extracted text quality.

        Args:
            extracted_text: Text already extracted via PyMuPDF.
            page_count: Total number of pages in PDF.

        Returns:
            True if OCR is recommended.
        """
        if page_count == 0:
            return False

        # Count words in extracted text
        words = extracted_text.split()
        words_per_page = len(words) / page_count

        # If very few words per page, likely scanned
        if words_per_page < self.MIN_WORDS_PER_PAGE:
            logger.debug(
                f"Low text density ({words_per_page:.1f} words/page), OCR recommended"
            )
            return True

        # Check for pages with no text (could indicate mixed document)
        pages_with_text = extracted_text.count("--- Page ")
        text_page_ratio = pages_with_text / page_count if page_count > 0 else 0

        if text_page_ratio < self.MIN_TEXT_PAGE_RATIO:
            logger.debug(
                f"Low text page ratio ({text_page_ratio:.1%}), OCR recommended"
            )
            return True

        return False

    def extract_text(self, pdf_path: Path) -> OCRResult:
        """Extract text from PDF using OCR.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            OCRResult with extracted text.

        Raises:
            OCRError: If OCR fails or dependencies missing.
        """
        if not self.is_available():
            raise OCRError(
                "OCR dependencies not available. "
                "Install Tesseract and ensure it's in PATH."
            )

        if not pdf_path.exists():
            raise OCRError(f"PDF file not found: {pdf_path}")

        logger.info(f"Starting OCR extraction for {pdf_path.name}")

        try:
            # Convert PDF to images
            images = self._pdf_to_images(pdf_path)

            # OCR each image
            pages = []
            for page_num, image in enumerate(images, 1):
                text = self._ocr_image(image)
                if text.strip():
                    pages.append(f"--- Page {page_num} ---\n{text}")

                # Clean up image
                image.close()

            extracted_text = "\n\n".join(pages)
            logger.info(
                f"OCR complete: {len(pages)} pages extracted from {pdf_path.name}"
            )

            return OCRResult(
                text=extracted_text,
                pages_processed=len(images),
                method="ocr",
            )

        except PDFInfoNotInstalledError as e:
            raise OCRError(
                "Poppler not installed. Required for PDF to image conversion. "
                "On Windows, download from: https://github.com/osber/poppler-windows"
            ) from e
        except Exception as e:
            raise OCRError(f"OCR extraction failed: {e}") from e

    def extract_hybrid(
        self,
        pdf_path: Path,
        existing_text: str,
        page_count: int,
    ) -> OCRResult:
        """Extract text using hybrid approach - OCR only pages with no text.

        Args:
            pdf_path: Path to PDF file.
            existing_text: Text already extracted via PyMuPDF.
            page_count: Total page count.

        Returns:
            OCRResult with combined text.
        """
        if not self.is_available():
            raise OCRError("OCR dependencies not available")

        # Parse existing text to find pages with content
        pages_with_text = set()
        existing_pages = {}

        for section in existing_text.split("--- Page "):
            if not section.strip():
                continue
            try:
                first_line = section.split("\n")[0]
                page_num = int(first_line.split()[0].replace("---", "").strip())
                page_text = "\n".join(section.split("\n")[1:])

                # Check if page has substantial text
                if len(page_text.split()) >= self.MIN_WORDS_PER_PAGE:
                    pages_with_text.add(page_num)
                    existing_pages[page_num] = page_text
            except (ValueError, IndexError):
                continue

        # If all pages have text, return existing
        if len(pages_with_text) == page_count:
            return OCRResult(
                text=existing_text,
                pages_processed=0,
                method="hybrid",
            )

        logger.info(
            f"Hybrid OCR: {len(pages_with_text)}/{page_count} pages have text, "
            f"OCR needed for {page_count - len(pages_with_text)} pages"
        )

        # Convert PDF to images
        images = self._pdf_to_images(pdf_path)

        # Build combined result
        pages = []
        ocr_count = 0

        for page_num, image in enumerate(images, 1):
            if page_num in existing_pages:
                # Use existing text
                pages.append(f"--- Page {page_num} ---\n{existing_pages[page_num]}")
            else:
                # OCR this page
                text = self._ocr_image(image)
                if text.strip():
                    pages.append(f"--- Page {page_num} ---\n{text}")
                ocr_count += 1

            image.close()

        logger.info(f"Hybrid OCR complete: {ocr_count} pages OCR'd")

        return OCRResult(
            text="\n\n".join(pages),
            pages_processed=ocr_count,
            method="hybrid",
        )

    def _pdf_to_images(self, pdf_path: Path) -> list:
        """Convert PDF pages to images.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            List of PIL Image objects.
        """
        kwargs = {"dpi": self.dpi}
        if self.poppler_path:
            kwargs["poppler_path"] = self.poppler_path

        return convert_from_path(str(pdf_path), **kwargs)

    def _ocr_image(self, image) -> str:
        """OCR a single image.

        Args:
            image: PIL Image object.

        Returns:
            Extracted text.
        """
        return pytesseract.image_to_string(image, lang=self.lang)


def get_ocr_handler(**kwargs) -> OCRHandler | None:
    """Get OCR handler if available, otherwise None.

    Args:
        **kwargs: Arguments to pass to OCRHandler.

    Returns:
        OCRHandler instance or None if unavailable.
    """
    if OCRHandler.is_available():
        return OCRHandler(**kwargs)

    logger.warning(
        "OCR not available. Install Tesseract and add to PATH for OCR support."
    )
    return None
