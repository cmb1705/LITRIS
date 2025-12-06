"""Text cleaning utilities for extracted PDF content."""

import re
from dataclasses import dataclass

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TextStats:
    """Statistics about text content."""

    char_count: int
    word_count: int
    line_count: int
    page_count: int
    avg_line_length: float


class TextCleaner:
    """Clean and normalize extracted PDF text."""

    # Patterns for cleaning
    MULTIPLE_SPACES = re.compile(r" {2,}")
    MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
    HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
    PAGE_NUMBERS = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
    HEADER_FOOTER = re.compile(r"^.{0,80}(page \d+|^\d+$).{0,80}$", re.MULTILINE | re.IGNORECASE)

    def __init__(
        self,
        min_line_length: int = 10,
        remove_headers_footers: bool = True,
        fix_hyphenation: bool = True,
    ):
        """Initialize cleaner.

        Args:
            min_line_length: Minimum line length to keep.
            remove_headers_footers: Whether to remove detected headers/footers.
            fix_hyphenation: Whether to fix hyphenated line breaks.
        """
        self.min_line_length = min_line_length
        self.remove_headers_footers = remove_headers_footers
        self.fix_hyphenation = fix_hyphenation

    def clean(self, text: str) -> str:
        """Clean extracted text.

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned text.
        """
        if not text:
            return ""

        # Fix hyphenated line breaks
        if self.fix_hyphenation:
            text = self.HYPHEN_LINEBREAK.sub(r"\1\2", text)

        # Remove page numbers and headers/footers
        if self.remove_headers_footers:
            text = self.PAGE_NUMBERS.sub("", text)
            text = self.HEADER_FOOTER.sub("", text)

        # Normalize whitespace
        text = self.MULTIPLE_SPACES.sub(" ", text)
        text = self.MULTIPLE_NEWLINES.sub("\n\n", text)

        # Filter short lines (often artifacts)
        lines = text.split("\n")
        lines = [
            line for line in lines
            if len(line.strip()) >= self.min_line_length or not line.strip()
        ]
        text = "\n".join(lines)

        return text.strip()

    def get_stats(self, text: str) -> TextStats:
        """Get statistics about text content.

        Args:
            text: Text to analyze.

        Returns:
            TextStats object.
        """
        lines = text.split("\n")
        words = text.split()

        # Count pages (look for page markers)
        page_markers = re.findall(r"--- Page \d+ ---", text)
        page_count = len(page_markers) if page_markers else 1

        # Calculate average line length (non-empty lines)
        non_empty_lines = [line for line in lines if line.strip()]
        avg_line_length = (
            sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
            if non_empty_lines
            else 0
        )

        return TextStats(
            char_count=len(text),
            word_count=len(words),
            line_count=len(lines),
            page_count=page_count,
            avg_line_length=avg_line_length,
        )

    def is_valid_extraction(self, text: str, min_words: int = 100) -> bool:
        """Check if extraction is valid (enough content).

        Args:
            text: Extracted text.
            min_words: Minimum word count.

        Returns:
            True if extraction appears valid.
        """
        if not text:
            return False

        stats = self.get_stats(text)
        return stats.word_count >= min_words

    def extract_sections(self, text: str) -> dict[str, str]:
        """Attempt to extract common sections from academic paper.

        Args:
            text: Full paper text.

        Returns:
            Dictionary mapping section names to content.
        """
        sections = {}

        # Common section patterns
        section_patterns = [
            (r"abstract", r"(?i)\babstract\b[:\s]*\n?(.+?)(?=\n\n|\n[A-Z]|\n\d\.|$)"),
            (r"introduction", r"(?i)\b(?:1\.?\s*)?introduction\b[:\s]*\n?(.+?)(?=\n\n[A-Z]|\n2\.|$)"),
            (r"methodology", r"(?i)\b(?:method(?:ology|s)?|research (?:design|method))\b[:\s]*\n?(.+?)(?=\n\n[A-Z]|\n\d\.|$)"),
            (r"results", r"(?i)\b(?:results?|findings?)\b[:\s]*\n?(.+?)(?=\n\n[A-Z]|\n\d\.|$)"),
            (r"discussion", r"(?i)\bdiscussion\b[:\s]*\n?(.+?)(?=\n\n[A-Z]|\n\d\.|$)"),
            (r"conclusion", r"(?i)\bconclusions?\b[:\s]*\n?(.+?)(?=\n\n[A-Z]|\nreferences|$)"),
            (r"references", r"(?i)\breferences\b[:\s]*\n?(.+?)$"),
        ]

        for name, pattern in section_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(1).strip()
                if len(content) > 50:  # Only keep substantial sections
                    sections[name] = content

        return sections

    def truncate_for_llm(
        self,
        text: str,
        max_chars: int = 150000,
        preserve_start: int = 20000,
        preserve_end: int = 10000,
    ) -> str:
        """Truncate text for LLM input while preserving key parts.

        Args:
            text: Full text.
            max_chars: Maximum characters.
            preserve_start: Characters to preserve from start.
            preserve_end: Characters to preserve from end.

        Returns:
            Truncated text.
        """
        if len(text) <= max_chars:
            return text

        logger.debug(f"Truncating text from {len(text)} to ~{max_chars} chars")

        # Preserve start and end
        middle_size = max_chars - preserve_start - preserve_end - 100  # Buffer for marker

        if middle_size <= 0:
            # Just take start and end
            return (
                text[:preserve_start]
                + "\n\n[... content truncated ...]\n\n"
                + text[-preserve_end:]
            )

        # Take start, some middle, and end
        start = text[:preserve_start]
        end = text[-preserve_end:]

        # Try to get middle from around center
        middle_start = len(text) // 2 - middle_size // 2
        middle = text[middle_start : middle_start + middle_size]

        return (
            start
            + "\n\n[... content truncated ...]\n\n"
            + middle
            + "\n\n[... content truncated ...]\n\n"
            + end
        )
