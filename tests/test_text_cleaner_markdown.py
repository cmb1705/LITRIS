"""Tests for TextCleaner markdown preservation."""

from src.extraction.text_cleaner import TextCleaner


def test_clean_preserves_short_lines_in_markdown():
    """preserve_markdown=True keeps short lines (table rows, list items)."""
    cleaner = TextCleaner()
    text = "# Title\n\n- item 1\n- item 2\n- item 3\n\n| A | B |\n|---|---|\n| 1 | 2 |"
    result = cleaner.clean(text, preserve_markdown=True)
    assert "- item 1" in result
    assert "| A | B |" in result


def test_clean_removes_short_lines_by_default():
    """Default clean() removes short lines as artifacts."""
    cleaner = TextCleaner()
    text = "This is a long enough line that should be kept.\n- short\n- tiny"
    result = cleaner.clean(text)
    assert "- short" not in result  # Below min_line_length=10


def test_clean_preserves_markdown_keeps_hyphen_fix():
    """preserve_markdown still fixes hyphenated line breaks."""
    cleaner = TextCleaner()
    text = "hyphen-\nated word"
    result = cleaner.clean(text, preserve_markdown=True)
    assert "hyphenated" in result


def test_clean_preserves_markdown_skips_header_footer_removal():
    """preserve_markdown skips HEADER_FOOTER regex removal."""
    cleaner = TextCleaner()
    # "page 1" matches HEADER_FOOTER pattern in normal mode
    text = "# Introduction\n\nSome body text that is long enough to keep.\n\npage 1"
    result_md = cleaner.clean(text, preserve_markdown=True)
    # In markdown mode, "page 1" content not stripped by HEADER_FOOTER
    # (PAGE_NUMBERS may still remove bare numbers, but "page 1" is the footer pattern)
    assert "Introduction" in result_md
