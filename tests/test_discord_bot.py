"""Tests for the LITRIS Discord bot.

Tests formatters and bot logic without requiring discord.py to be installed.
"""

from unittest.mock import patch

import pytest

from src.discord_bot.formatters import (
    RESULTS_PER_PAGE,
    _truncate,
    format_paper_embed,
    format_search_page,
    format_search_result_embed,
    format_summary_embed,
)


class TestTruncate:
    """Tests for text truncation helper."""

    def test_short_text_unchanged(self):
        assert _truncate("hello", 10) == "hello"

    def test_exact_length_unchanged(self):
        assert _truncate("hello", 5) == "hello"

    def test_long_text_truncated(self):
        result = _truncate("hello world", 8)
        assert result == "hello..."
        assert len(result) == 8

    def test_empty_string(self):
        assert _truncate("", 10) == ""


class TestFormatPaperEmbed:
    """Tests for single paper embed formatting."""

    def test_basic_paper(self):
        paper = {
            "paper_id": "abc123",
            "found": True,
            "paper": {
                "title": "Test Paper",
                "author_string": "Smith, J.",
                "publication_year": 2024,
                "doi": "10.1234/test",
                "journal": "Test Journal",
                "abstract": "This is a test abstract.",
                "collections": ["ML", "NLP"],
            },
            "extraction": {
                "q02_thesis": "Tests are important.",
                "q07_methods": "Mixed methods approach",
                "q03_key_claims": "Claim 1. Claim 2.",
                "quality_rating": 4,
                "extraction_confidence": 0.85,
            },
        }

        embed = format_paper_embed(paper)

        assert embed["title"] == "Test Paper"
        assert "Smith, J." in embed["description"]
        assert "10.1234/test" in embed["description"]
        assert embed["color"] == 0x3498DB
        assert any(f["name"] == "Thesis" for f in embed["fields"])
        assert any(f["name"] == "Key Claims" for f in embed["fields"])
        assert "abc123" in embed["footer"]["text"]

    def test_minimal_paper(self):
        paper = {
            "paper_id": "min1",
            "paper": {},
            "extraction": None,
        }

        embed = format_paper_embed(paper)

        assert embed["title"] == "Unknown Title"
        assert embed["fields"] == []

    def test_paper_with_methods(self):
        paper = {
            "paper_id": "meth1",
            "paper": {"title": "Methods Paper"},
            "extraction": {
                "q07_methods": "Quantitative regression analysis",
            },
        }

        embed = format_paper_embed(paper)

        methods_field = next(
            f for f in embed["fields"] if f["name"] == "Methods"
        )
        assert "Quantitative" in methods_field["value"]


class TestFormatSearchResultEmbed:
    """Tests for search result embed formatting."""

    def test_basic_result(self):
        result = {
            "title": "Result Paper",
            "authors": "Jones, A.",
            "year": 2023,
            "score": 0.8765,
            "paper_id": "res1",
            "matched_text": "Matched content here.",
            "extraction": {"q02_thesis": "Main argument."},
        }

        embed = format_search_result_embed(result, rank=1)

        assert "#1" in embed["title"]
        assert "Result Paper" in embed["title"]
        assert "0.8765" in embed["description"]
        assert "Jones, A." in embed["description"]
        assert "res1" in embed["footer"]["text"]

    def test_without_rank(self):
        result = {
            "title": "No Rank",
            "authors": "Doe",
            "year": 2024,
            "score": 0.5,
            "paper_id": "nr1",
            "matched_text": "",
        }

        embed = format_search_result_embed(result)

        assert "#" not in embed["title"]


class TestFormatSearchPage:
    """Tests for paginated search result formatting."""

    def test_single_page(self):
        results = [
            {"title": f"Paper {i}", "authors": "Auth", "year": 2024,
             "score": 0.9 - i * 0.1, "paper_id": f"p{i}", "matched_text": ""}
            for i in range(3)
        ]

        embeds = format_search_page(results, "test query", 0, 3)

        # Header + 3 results
        assert len(embeds) == 4
        assert "test query" in embeds[0]["title"]
        assert "Page 1/1" in embeds[0]["description"]

    def test_multi_page_info(self):
        results = [
            {"title": f"Paper {i}", "authors": "Auth", "year": 2024,
             "score": 0.5, "paper_id": f"p{i}", "matched_text": ""}
            for i in range(RESULTS_PER_PAGE)
        ]

        # Page 2 of 3 total (15 results)
        embeds = format_search_page(results, "query", 1, 15)

        assert "Page 2/3" in embeds[0]["description"]
        # Rank should start at 6 for page 2
        assert f"#{RESULTS_PER_PAGE + 1}" in embeds[1]["title"]


class TestFormatSummaryEmbed:
    """Tests for index summary embed formatting."""

    def test_basic_summary(self):
        summary = {
            "total_papers": 332,
            "total_extractions": 310,
            "vector_store": {"total_documents": 3746},
            "papers_by_collection": {
                "Network Analysis": 50,
                "Scientometrics": 45,
                "ML": 30,
            },
            "top_disciplines": {
                "network science": 80,
                "bibliometrics": 60,
            },
            "generated_at": "2026-03-07T12:00:00",
        }

        embed = format_summary_embed(summary)

        assert "LITRIS" in embed["title"]
        assert "332" in embed["description"]
        assert embed["color"] == 0xE67E22
        # Check fields contain stats
        paper_field = next(
            f for f in embed["fields"] if f["name"] == "Papers"
        )
        assert paper_field["value"] == "332"

    def test_empty_summary(self):
        summary = {
            "total_papers": 0,
            "total_extractions": 0,
            "vector_store": {},
        }

        embed = format_summary_embed(summary)

        assert "0 papers" in embed["description"]


class TestBotCreation:
    """Tests for bot creation and configuration."""

    @patch("src.discord_bot.bot.HAS_DISCORD", False)
    def test_raises_without_discord(self):
        from src.discord_bot.bot import create_bot

        with pytest.raises(ImportError, match="discord.py is required"):
            create_bot()

    @patch("src.discord_bot.bot.HAS_DISCORD", False)
    def test_run_bot_raises_without_discord(self):
        from src.discord_bot.bot import run_bot

        with pytest.raises(ImportError, match="discord.py is required"):
            run_bot(token="fake-token")

    def test_run_bot_raises_without_token(self):
        from src.discord_bot.bot import run_bot

        with patch("src.discord_bot.bot.HAS_DISCORD", True), \
             patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Discord bot token required"):
                run_bot(token=None)
