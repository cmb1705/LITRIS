"""Tests for CLI-based LLM extraction."""

import logging
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.analysis.cli_executor import (
    ClaudeCliExecutor,
    CliExecutionError,
    ParseError,
)
from src.analysis.progress_tracker import ProgressTracker
from src.analysis.rate_limit_handler import RateLimitExceededError, RateLimitHandler
from src.zotero.models import Author


class TestClaudeCliExecutor:
    """Test CLI executor functionality."""

    def test_verify_auth_with_api_key_warns(self, monkeypatch, caplog, tmp_path):
        """Test that API key presence warns about billing when no OAuth available."""
        from src.analysis.cli_executor import ClaudeCliAuthenticator

        # Enable log propagation for test capture (both parent and child loggers)
        parent_logger = logging.getLogger("lit_review")
        cli_logger = logging.getLogger("lit_review.analysis.cli_executor")
        original_parent_propagate = parent_logger.propagate
        original_cli_propagate = cli_logger.propagate
        parent_logger.propagate = True
        cli_logger.propagate = True
        caplog.set_level(logging.WARNING, logger="lit_review")

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

        try:
            with patch("shutil.which") as mock_which:
                mock_which.return_value = "/usr/bin/claude"

                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0, stdout="1.0.0")

                    # Mock credentials path to not exist (so API key is the fallback)
                    nonexistent_path = tmp_path / "nonexistent"
                    with patch.object(
                        ClaudeCliAuthenticator, "_get_credentials_path", return_value=nonexistent_path
                    ):
                        executor = ClaudeCliExecutor()
                        result = executor.verify_authentication()

                        # Should succeed but with warning
                        assert result is True
                        assert executor.authenticator.get_auth_method() == "api_key"
                        # Check warning was logged
                        assert any("API billing" in record.message for record in caplog.records)
        finally:
            parent_logger.propagate = original_parent_propagate
            cli_logger.propagate = original_cli_propagate

    def test_verify_auth_no_api_key(self, monkeypatch):
        """Test auth check passes with OAuth token (no API key)."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "test-oauth-token")

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/claude"

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="1.0.0")

                executor = ClaudeCliExecutor()
                result = executor.verify_authentication()

                assert result is True
                assert executor.authenticator.get_auth_method() == "oauth_token"

    def test_verify_auth_cli_not_found(self, monkeypatch):
        """Test error when CLI not in PATH."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with patch("shutil.which") as mock_which:
            mock_which.return_value = None

            executor = ClaudeCliExecutor()

            with pytest.raises(CliExecutionError) as exc_info:
                executor.verify_authentication()

            assert "not found" in str(exc_info.value)

    def test_rate_limit_detection(self):
        """Test rate limit detection in responses."""
        executor = ClaudeCliExecutor()

        # Should detect rate limit
        assert executor._is_rate_limited("Rate limit exceeded", "") is True
        assert executor._is_rate_limited("", "usage limit reached") is True
        assert executor._is_rate_limited("Please try again later", "") is True

        # Should not detect rate limit
        assert executor._is_rate_limited("Normal response", "") is False

    def test_parse_json_response(self):
        """Test parsing JSON from CLI response."""
        executor = ClaudeCliExecutor()

        # Direct JSON
        result = executor._parse_response('{"key": "value"}')
        assert result == {"key": "value"}

        # JSON in markdown block
        result = executor._parse_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

        # JSON with surrounding text
        result = executor._parse_response('Here is the result:\n{"key": "value"}\nDone.')
        assert result == {"key": "value"}

    def test_parse_invalid_json(self):
        """Test error on invalid JSON."""
        executor = ClaudeCliExecutor()

        with pytest.raises(ParseError) as exc_info:
            executor._parse_response("This is not JSON at all")

        assert "Could not parse" in str(exc_info.value)
        assert exc_info.value.raw_output == "This is not JSON at all"

    def test_extract_reset_time(self):
        """Test extracting reset time from error messages."""
        executor = ClaudeCliExecutor()

        assert executor._extract_reset_time("try again in 30 minutes") == "30 minutes"
        assert executor._extract_reset_time("reset in 2 hours") == "2 hours"
        assert executor._extract_reset_time("no time mentioned") is None


class TestRateLimitHandler:
    """Test rate limit handling."""

    def test_session_tracking(self):
        """Test session request counting."""
        handler = RateLimitHandler()
        handler.start_session()

        assert handler.get_session_request_count() == 0

        handler.record_request()
        handler.record_request()

        assert handler.get_session_request_count() == 2

    def test_approaching_limit_detection(self):
        """Test detection when approaching rate limit."""
        handler = RateLimitHandler(conservative_limit=10)
        handler.start_session()

        # Not approaching initially
        assert handler.is_approaching_limit() is False

        # Simulate requests
        for _ in range(10):
            handler.record_request()

        assert handler.is_approaching_limit() is True

    def test_response_limit_detection(self):
        """Test detecting rate limit in responses."""
        handler = RateLimitHandler()

        assert handler.check_response_for_limit("rate limit exceeded") is True
        assert handler.check_response_for_limit("normal response") is False

    def test_handle_limit_raises_when_not_pausing(self):
        """Test that limit raises error when pause_on_limit is False."""
        handler = RateLimitHandler(pause_on_limit=False, auto_resume=False)
        handler.start_session()

        with pytest.raises(RateLimitExceededError):
            handler.handle_limit_hit()

    def test_handle_limit_returns_false_when_pausing(self):
        """Test that limit returns False when pausing."""
        handler = RateLimitHandler(pause_on_limit=True, auto_resume=False)
        handler.start_session()

        result = handler.handle_limit_hit()
        assert result is False

    def test_session_stats(self):
        """Test session statistics."""
        handler = RateLimitHandler()
        handler.start_session()
        handler.record_request()
        handler.record_request()

        stats = handler.get_session_stats()

        assert stats["request_count"] == 2
        assert stats["limit_hit"] is False
        assert stats["session_start"] is not None


class TestProgressTracker:
    """Test progress tracking functionality."""

    def test_initialize_and_load(self, tmp_path):
        """Test initializing and loading progress."""
        tracker = ProgressTracker(tmp_path)
        tracker.initialize(100)

        # Load in new instance
        tracker2 = ProgressTracker(tmp_path)
        state = tracker2.load()

        assert state is not None
        assert state.total_papers == 100
        assert len(state.completed) == 0
        assert len(state.failed) == 0

    def test_mark_completed(self, tmp_path):
        """Test marking papers as completed."""
        tracker = ProgressTracker(tmp_path)
        tracker.initialize(10)

        tracker.mark_completed("paper1")
        tracker.mark_completed("paper2")

        state = tracker.load()
        assert "paper1" in state.completed
        assert "paper2" in state.completed

    def test_mark_failed(self, tmp_path):
        """Test marking papers as failed."""
        tracker = ProgressTracker(tmp_path)
        tracker.initialize(10)

        tracker.mark_failed("paper1", "Some error")

        state = tracker.load()
        assert len(state.failed) == 1
        assert state.failed[0].paper_id == "paper1"
        assert state.failed[0].error == "Some error"

    def test_get_pending_papers(self, tmp_path):
        """Test getting pending papers."""
        tracker = ProgressTracker(tmp_path)
        tracker.initialize(5)

        all_papers = ["p1", "p2", "p3", "p4", "p5"]

        tracker.mark_completed("p1")
        tracker.mark_failed("p2", "error")

        pending = tracker.get_pending_papers(all_papers)

        assert pending == ["p3", "p4", "p5"]

    def test_reset(self, tmp_path):
        """Test resetting progress."""
        tracker = ProgressTracker(tmp_path)
        tracker.initialize(10)
        tracker.mark_completed("paper1")

        tracker.reset()

        state = tracker.load()
        assert state is None

    def test_get_summary(self, tmp_path):
        """Test progress summary."""
        tracker = ProgressTracker(tmp_path)
        tracker.initialize(10)

        tracker.mark_completed("p1")
        tracker.mark_completed("p2")
        tracker.mark_failed("p3", "error")

        summary = tracker.get_summary()

        assert summary["total"] == 10
        assert summary["completed"] == 2
        assert summary["failed"] == 1
        assert summary["pending"] == 7
        assert summary["progress_percent"] == 20.0

    def test_handles_corrupted_file(self, tmp_path):
        """Test handling of corrupted progress file."""
        tracker = ProgressTracker(tmp_path)

        # Write invalid JSON
        progress_file = tmp_path / "cli_progress.json"
        progress_file.write_text("not valid json")

        state = tracker.load()
        assert state is None

        # Original file should be backed up
        backup = tmp_path / "cli_progress.json.bak"
        assert backup.exists()

    def test_completed_removes_from_failed(self, tmp_path):
        """Test that completing a paper removes it from failed list."""
        tracker = ProgressTracker(tmp_path)
        tracker.initialize(5)

        # First mark as failed
        tracker.mark_failed("paper1", "error")

        # Then complete it (retry succeeded)
        tracker.mark_completed("paper1")

        state = tracker.load()
        assert "paper1" in state.completed
        assert all(fp.paper_id != "paper1" for fp in state.failed)


class TestCliSectionExtractor:
    """Test CLI section extractor integration."""

    def test_prompt_uses_full_name_and_year(self, tmp_path, monkeypatch):
        """Ensure prompt uses author full names and publication_year."""
        from src.analysis.cli_section_extractor import CliSectionExtractor
        from src.zotero.models import PaperMetadata

        captured = {}

        class DummyExecutor:
            def extract(self, prompt, input_text):
                captured["prompt"] = prompt
                captured["input_text"] = input_text
                # Minimal valid response
                return {"thesis_statement": "Thesis", "research_questions": []}

        dummy_rate = MagicMock()
        dummy_rate.record_request = MagicMock()

        metadata = PaperMetadata(
            zotero_key="KEY12345",
            zotero_item_id=1,
            title="Test Paper",
            item_type="journalArticle",
            publication_year=2020,
            authors=[Author(first_name="Alice", last_name="Smith"), Author(first_name="Bob", last_name="Lee")],
            date_added=datetime.now(),
            date_modified=datetime.now(),
        )

        extractor = CliSectionExtractor(
            cache_dir=tmp_path,
            executor=DummyExecutor(),
            rate_handler=dummy_rate,
        )

        extractor.extract_single("paper1", "Some text body", metadata)

        assert "Alice Smith" in captured["prompt"]
        assert "Bob Lee" in captured["prompt"]
        assert "2020" in captured["prompt"]

    def test_parse_response_full(self, tmp_path):
        """Test parsing a full extraction response."""
        from src.analysis.cli_section_extractor import CliSectionExtractor
        from src.zotero.models import PaperMetadata

        extractor = CliSectionExtractor(cache_dir=tmp_path)

        response = {
            "thesis_statement": "This is the thesis",
            "research_questions": ["RQ1", "RQ2"],
            "methodology": {
                "approach": "qualitative",
                "design": "case study",
            },
            "key_findings": [
                {"finding": "Finding 1", "evidence_type": "qualitative", "significance": "high"},
                "Finding 2 as string",
            ],
            "key_claims": [
                {"claim": "Claim 1", "support_type": "data"},
            ],
            "limitations": ["Limitation 1"],
            "future_directions": ["Direction 1"],
            "extraction_confidence": 0.85,
        }

        metadata = PaperMetadata(
            zotero_key="ABC123",
            zotero_item_id=1,
            title="Test Paper",
            item_type="journalArticle",
            date_added=datetime.now(),
            date_modified=datetime.now(),
        )

        extraction = extractor._parse_response(response, "paper1", metadata)

        # PaperExtraction doesn't have paper_id - that's in ExtractionResult wrapper
        assert extraction.thesis_statement == "This is the thesis"
        assert len(extraction.research_questions) == 2
        assert extraction.methodology.approach == "qualitative"
        assert len(extraction.key_findings) == 2
        assert extraction.extraction_confidence == 0.85

    def test_parse_response_minimal(self, tmp_path):
        """Test parsing minimal response."""
        from src.analysis.cli_section_extractor import CliSectionExtractor
        from src.zotero.models import PaperMetadata

        extractor = CliSectionExtractor(cache_dir=tmp_path)

        response = {
            "thesis_statement": "Minimal thesis",
        }

        metadata = PaperMetadata(
            zotero_key="ABC123",
            zotero_item_id=1,
            title="Test Paper",
            item_type="journalArticle",
            date_added=datetime.now(),
            date_modified=datetime.now(),
        )

        extraction = extractor._parse_response(response, "paper1", metadata)

        # PaperExtraction doesn't have paper_id - that's in ExtractionResult wrapper
        assert extraction.thesis_statement == "Minimal thesis"
        assert extraction.key_findings == []
        assert extraction.extraction_confidence == 0.7  # Default


class TestCliIntegration:
    """Integration tests for CLI extraction (require CLI to be installed)."""

    @pytest.mark.skipif(
        not os.path.exists("/usr/bin/claude") and not os.path.exists("C:\\Program Files\\claude\\claude.exe"),
        reason="Claude CLI not installed",
    )
    def test_cli_available(self, monkeypatch):
        """Test that CLI is available (if installed)."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        _executor = ClaudeCliExecutor()
        # This would pass if CLI is installed
        # In CI, this test is skipped
