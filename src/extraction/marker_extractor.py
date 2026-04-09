"""Marker PDF-to-markdown extraction.

Marker (https://github.com/VikParuchuri/marker) converts PDFs to
high-quality markdown using deep learning models. It handles:
- Complex layouts (multi-column, tables, figures)
- Mathematical equations (LaTeX)
- Code blocks
- Headers and structure preservation

Marker is an optional dependency; this module gracefully degrades
if it is not installed.
"""

from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Lazy: ``marker`` transitively imports torch/transformers/google.genai
# (~5s). Only probe for availability on first call so the MCP server and
# other search-only consumers of ``src.extraction.cascade`` don't pay the
# cost at import time. Tests patch ``MARKER_AVAILABLE`` directly to force
# the unavailable path.
MARKER_AVAILABLE: bool | None = None
PdfConverter = None
create_model_dict = None


def is_available() -> bool:
    """Check if Marker is installed and available.

    Returns:
        True if Marker can be used for extraction.
    """
    global MARKER_AVAILABLE
    if MARKER_AVAILABLE is None:
        try:
            import marker.converters.pdf  # noqa: F401
            import marker.models  # noqa: F401

            MARKER_AVAILABLE = True
        except ImportError:
            MARKER_AVAILABLE = False
    return MARKER_AVAILABLE


def extract_with_marker(pdf_path: Path, timeout: int = 120) -> str | None:
    """Extract text from PDF using Marker.

    Marker produces markdown output which preserves document structure
    better than raw text extraction.

    Args:
        pdf_path: Path to PDF file.
        timeout: Not used directly (Marker doesn't support timeouts),
            but kept for interface consistency.

    Returns:
        Extracted markdown text, or None if extraction fails.
    """
    if not is_available():
        logger.debug("Marker not installed, skipping")
        return None

    if not pdf_path.exists():
        logger.warning(f"PDF not found: {pdf_path}")
        return None

    logger.info(f"Marker extraction: {pdf_path.name}")

    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        models = create_model_dict()
        converter = PdfConverter(artifact_dict=models)
        result = converter(str(pdf_path))

        # Marker returns a ConverterResult; extract the markdown text
        text = result.markdown if hasattr(result, "markdown") else str(result)

        if text and len(text.split()) > 100:
            logger.info(f"Marker: extracted {len(text.split())} words from {pdf_path.name}")
            return text

        logger.debug(f"Marker: insufficient text from {pdf_path.name}")
        return None

    except Exception as e:
        logger.warning(f"Marker extraction failed for {pdf_path.name}: {e}")
        return None
