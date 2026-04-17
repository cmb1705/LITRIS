"""Tests for logging configuration helpers."""

from __future__ import annotations

from pathlib import Path

from src.utils import logging_config


class _EncodingSensitiveStream:
    """Test stream that fails on characters unsupported by its console encoding."""

    encoding = "cp1252"

    def __init__(self) -> None:
        self.messages: list[str] = []

    def write(self, message: str) -> int:
        if "\u2002" in message:
            raise UnicodeEncodeError(
                "charmap",
                message,
                message.index("\u2002"),
                message.index("\u2002") + 1,
                "character maps to <undefined>",
            )
        self.messages.append(message)
        return len(message)

    def flush(self) -> None:
        return None


def test_find_project_root_prefers_cwd_hierarchy(monkeypatch, tmp_path):
    project_root = tmp_path / "sample_project"
    nested_dir = project_root / "nested" / "child"
    nested_dir.mkdir(parents=True)
    (project_root / "config.yaml").write_text("version: 1.0.0\n", encoding="utf-8")

    monkeypatch.chdir(nested_dir)

    assert logging_config._find_project_root() == project_root


def test_find_project_root_falls_back_to_module_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    expected_root = Path(logging_config.__file__).resolve().parents[2]

    assert logging_config._find_project_root() == expected_root


def test_log_context_escapes_console_unsafe_characters(monkeypatch, tmp_path):
    stream = _EncodingSensitiveStream()
    monkeypatch.setattr(logging_config.sys, "stdout", stream)

    logger = logging_config.setup_logging(
        level="INFO",
        log_dir=tmp_path,
        log_file="test.log",
        console=True,
        file_logging=False,
    )

    with logging_config.LogContext(logger, "Extracting paper: Title\u2002with en space"):
        pass

    assert any("\\u2002" in message for message in stream.messages)
