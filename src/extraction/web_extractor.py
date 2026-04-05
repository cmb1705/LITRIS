"""HTML attachment extraction for webpage-backed Zotero items.

Uses ``trafilatura`` when available to strip boilerplate from saved HTML pages.
Falls back to a conservative tag-stripper when the dependency is unavailable
or extraction fails.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(
    r"<(?:script|style|noscript|svg)[^>]*>.*?</(?:script|style|noscript|svg)>",
    flags=re.DOTALL | re.IGNORECASE,
)
_BOILERPLATE_RE = re.compile(
    r"<(?:nav|header|footer|aside|form)[^>]*>.*?</(?:nav|header|footer|aside|form)>",
    flags=re.DOTALL | re.IGNORECASE,
)
_BLOCK_RE = re.compile(
    r"<(?:p|div|section|article|main|br|h[1-6]|li|tr)[^>]*>",
    flags=re.IGNORECASE,
)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)


@dataclass(frozen=True)
class WebExtractionResult:
    """Clean text extracted from a saved webpage attachment."""

    text: str
    method: str
    source_path: Path

    @property
    def word_count(self) -> int:
        """Return the extracted word count."""
        return len(self.text.split())


def extract_html_attachment(html_path: Path, url: str | None = None) -> WebExtractionResult:
    """Extract canonical text from a saved HTML attachment.

    Args:
        html_path: Local HTML file captured by Zotero/SingleFile.
        url: Canonical source URL when known.

    Returns:
        ``WebExtractionResult`` with cleaned body text.

    Raises:
        FileNotFoundError: If the attachment file does not exist.
        ValueError: If extraction does not yield meaningful text.
    """
    if not html_path.exists():
        raise FileNotFoundError(f"HTML attachment not found: {html_path}")

    raw_html = html_path.read_text(encoding="utf-8", errors="ignore")
    text = _extract_with_trafilatura(raw_html, url=url)
    method = "trafilatura"
    if not text:
        text = _fallback_extract_text(raw_html)
        method = "html_fallback_strip"

    normalized = _normalize_text(text)
    if len(normalized.split()) < 50:
        raise ValueError(
            f"HTML attachment {html_path} did not yield enough article text"
        )

    return WebExtractionResult(text=normalized, method=method, source_path=html_path)


def _extract_with_trafilatura(raw_html: str, url: str | None = None) -> str | None:
    """Use trafilatura when installed."""
    try:
        import trafilatura
    except ImportError:
        logger.debug("trafilatura not installed; using HTML fallback extraction")
        return None

    try:
        return trafilatura.extract(
            raw_html,
            output_format="txt",
            include_comments=False,
            include_tables=True,
            include_links=False,
            favor_precision=True,
            url=url,
        )
    except Exception as exc:
        logger.debug("trafilatura extraction failed: %s", exc)
        return None


def _fallback_extract_text(raw_html: str) -> str:
    """Extract readable text using a conservative regex-based fallback."""
    text = _HTML_COMMENT_RE.sub(" ", raw_html)
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _BOILERPLATE_RE.sub(" ", text)
    text = _BLOCK_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return text


def _normalize_text(text: str) -> str:
    """Normalize whitespace and trivial boilerplate from extracted HTML text."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n\s*\n\s*\n+", "\n\n", normalized)
    normalized = re.sub(r"Page saved with SingleFile", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()
