"""Tests for web UI utility functions extracted from scripts/web_ui.py."""

from html import escape

from scripts.web_ui import highlight_query_terms


class TestHighlightQueryTerms:
    """Tests for the highlight_query_terms function."""

    def test_escapes_html_in_text(self):
        """XSS-bearing text is escaped before rendering."""
        result = highlight_query_terms("<script>alert('xss')</script>", "other")
        assert "<script>" not in result
        assert escape("<script>") in result

    def test_escapes_html_in_matched_term(self):
        """Even matched terms are escaped inside the <mark> tag."""
        result = highlight_query_terms("Use <b>bold</b> tags", "bold")
        # The word 'bold' is inside an HTML tag in the source text;
        # after escaping, '<b>' becomes '&lt;b&gt;'
        assert "&lt;b&gt;" in result

    def test_highlights_matching_terms(self):
        """Query terms found in text get wrapped in <mark> tags."""
        result = highlight_query_terms(
            "Network analysis of citation graphs", "network citation"
        )
        assert "<mark" in result
        assert "Network" in result
        assert "citation" in result

    def test_empty_text_returns_empty(self):
        """Empty or None text returns empty string."""
        assert highlight_query_terms("", "query") == ""
        assert highlight_query_terms(None, "query") == ""

    def test_empty_query_returns_escaped_text(self):
        """Empty query returns escaped text without highlights."""
        result = highlight_query_terms("Some <em>text</em>", "")
        assert "<mark" not in result
        assert "&lt;em&gt;" in result

    def test_short_query_terms_ignored(self):
        """Query terms shorter than 3 characters are ignored."""
        result = highlight_query_terms("I am here", "I am")
        assert "<mark" not in result

    def test_case_insensitive_matching(self):
        """Matching is case-insensitive."""
        result = highlight_query_terms("NETWORK analysis", "network")
        assert "<mark" in result
        assert "NETWORK" in result
