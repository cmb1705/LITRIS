"""arXiv HTML and ar5iv extraction for arXiv papers.

For papers sourced from arXiv, the original LaTeX-rendered HTML is
significantly higher quality than PDF text extraction. This module
provides two tiers:

1. arXiv HTML API (abs page source) - direct from arXiv
2. ar5iv (ar5iv.labs.arxiv.org) - HTML5 rendering of arXiv LaTeX source

Both require network access. Detection of arXiv papers uses DOI, URL,
or arXiv ID patterns in metadata.
"""

import re
from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Pattern to detect arXiv IDs: old format (hep-th/0601001) and new (2301.12345)
_ARXIV_ID_PATTERNS = [
    re.compile(r"(\d{4}\.\d{4,5})(v\d+)?"),  # New format
    re.compile(r"([a-z-]+/\d{7})(v\d+)?"),  # Old format
]

# HTML tag stripping (simple; avoids BeautifulSoup dependency)
_TAG_RE = re.compile(r"<[^>]+>")


def detect_arxiv_id(
    doi: str | None = None,
    url: str | None = None,
    pdf_path: Path | None = None,
    title: str | None = None,
) -> str | None:
    """Detect arXiv ID from paper metadata.

    Args:
        doi: Paper DOI (may contain arXiv DOI like 10.48550/arXiv.2301.12345).
        url: Paper URL (may be an arXiv abs or PDF URL).
        pdf_path: Path to PDF (filename may contain arXiv ID).
        title: Paper title (last resort, rarely contains arXiv ID).

    Returns:
        arXiv ID string (e.g., "2301.12345") or None.
    """
    sources = []
    if doi:
        sources.append(doi)
    if url:
        sources.append(url)
    if pdf_path:
        sources.append(str(pdf_path.stem))
        sources.append(str(pdf_path))
    if title:
        sources.append(title)

    for source in sources:
        for pattern in _ARXIV_ID_PATTERNS:
            match = pattern.search(source)
            if match:
                return match.group(1)

    return None


def fetch_arxiv_html(arxiv_id: str, timeout: int = 30) -> str | None:
    """Fetch paper text from arXiv HTML page.

    Uses the arXiv HTML rendering at https://arxiv.org/html/{id}.
    Falls back gracefully if the paper has no HTML version.

    Args:
        arxiv_id: arXiv paper ID (e.g., "2301.12345").
        timeout: Request timeout in seconds.

    Returns:
        Extracted text content, or None if unavailable.
    """
    import urllib.error
    import urllib.request

    url = f"https://arxiv.org/html/{arxiv_id}"
    logger.info(f"Fetching arXiv HTML: {url}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LITRIS/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                logger.debug(f"arXiv HTML returned status {response.status}")
                return None

            html = response.read().decode("utf-8", errors="replace")
            text = _extract_text_from_html(html)

            if text and len(text.split()) > 100:
                logger.info(f"arXiv HTML: extracted {len(text.split())} words")
                return text

            logger.debug("arXiv HTML: insufficient text extracted")
            return None

    except urllib.error.HTTPError as e:
        logger.debug(f"arXiv HTML not available (HTTP {e.code})")
        return None
    except Exception as e:
        logger.debug(f"arXiv HTML fetch failed: {e}")
        return None


def fetch_ar5iv_html(arxiv_id: str, timeout: int = 30) -> str | None:
    """Fetch paper text from ar5iv HTML5 rendering.

    ar5iv provides high-quality HTML5 renderings of arXiv papers at
    https://ar5iv.labs.arxiv.org/html/{id}.

    Args:
        arxiv_id: arXiv paper ID (e.g., "2301.12345").
        timeout: Request timeout in seconds.

    Returns:
        Extracted text content, or None if unavailable.
    """
    import urllib.error
    import urllib.request

    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    logger.info(f"Fetching ar5iv HTML: {url}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LITRIS/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                logger.debug(f"ar5iv returned status {response.status}")
                return None

            html = response.read().decode("utf-8", errors="replace")
            text = _extract_text_from_html(html)

            if text and len(text.split()) > 100:
                logger.info(f"ar5iv: extracted {len(text.split())} words")
                return text

            logger.debug("ar5iv: insufficient text extracted")
            return None

    except urllib.error.HTTPError as e:
        logger.debug(f"ar5iv not available (HTTP {e.code})")
        return None
    except Exception as e:
        logger.debug(f"ar5iv fetch failed: {e}")
        return None


def _extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML content.

    Simple tag-stripping approach that avoids requiring BeautifulSoup.
    Handles common HTML entities and collapses whitespace.

    Args:
        html: Raw HTML string.

    Returns:
        Cleaned text content.
    """
    # Remove script and style blocks
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<nav[^>]*>.*?</nav>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<header[^>]*>.*?</header>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<footer[^>]*>.*?</footer>", " ", text, flags=re.DOTALL | re.IGNORECASE)

    # Replace block elements with newlines
    text = re.sub(r"<(?:p|div|br|h[1-6]|li|tr)[^>]*>", "\n", text, flags=re.IGNORECASE)

    # Strip remaining tags
    text = _TAG_RE.sub(" ", text)

    # Decode common HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&nbsp;", " ")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")

    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()
