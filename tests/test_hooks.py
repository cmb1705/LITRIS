"""Tests for cross-platform hook scripts in scripts/hooks/."""

import os
import subprocess
import sys

import pytest

HOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "hooks")


def _ruff_available() -> bool:
    """Check if ruff is on PATH without crashing on FileNotFoundError."""
    try:
        return subprocess.run(["ruff", "--version"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


class TestZoteroGuard:
    """Tests for zotero_guard.py -- blocks writes to Zotero directories."""

    def _run_guard(self, file_path: str, zotero_path: str | None = None) -> int:
        env = os.environ.copy()
        env["CLAUDE_FILE_PATH"] = file_path
        if zotero_path is not None:
            env["ZOTERO_STORAGE_PATH"] = zotero_path
        else:
            env.pop("ZOTERO_STORAGE_PATH", None)
        result = subprocess.run(
            [sys.executable, os.path.join(HOOKS_DIR, "zotero_guard.py")],
            env=env,
            capture_output=True,
            text=True,
        )
        return result.returncode

    def test_blocks_zotero_path(self):
        """Paths containing 'Zotero' are blocked (exit 1) when no custom path set."""
        assert self._run_guard("/home/user/Zotero/storage/file.pdf") == 1

    def test_blocks_custom_zotero_path(self):
        """Custom ZOTERO_STORAGE_PATH is matched."""
        assert (
            self._run_guard(
                "D:/MyLibrary/papers/file.pdf",
                zotero_path="D:/MyLibrary",
            )
            == 1
        )

    def test_allows_safe_path(self):
        """Non-Zotero paths pass (exit 0)."""
        assert self._run_guard("src/analysis/gap_detection.py") == 0

    def test_allows_empty_path(self):
        """Empty CLAUDE_FILE_PATH exits cleanly."""
        assert self._run_guard("") == 0


class TestRuffLint:
    """Tests for ruff_lint.py -- runs ruff on Python files."""

    def _run_lint(self, file_path: str) -> int:
        env = os.environ.copy()
        env["CLAUDE_FILE_PATH"] = file_path
        result = subprocess.run(
            [sys.executable, os.path.join(HOOKS_DIR, "ruff_lint.py")],
            env=env,
            capture_output=True,
            text=True,
        )
        return result.returncode

    def test_skips_non_python(self):
        """Non-.py files are skipped (exit 0)."""
        assert self._run_lint("README.md") == 0

    def test_skips_empty_path(self):
        """Empty path is skipped (exit 0)."""
        assert self._run_lint("") == 0

    @pytest.mark.skipif(
        not _ruff_available(),
        reason="ruff not on PATH",
    )
    def test_passes_clean_python_file(self, tmp_path):
        """A clean Python file passes ruff check."""
        clean_file = tmp_path / "clean.py"
        clean_file.write_text('"""Module."""\n\nx = 1\n')
        assert self._run_lint(str(clean_file)) == 0
