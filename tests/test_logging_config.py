"""Tests for logging project root detection."""

from __future__ import annotations

from pathlib import Path

from src.utils import logging_config


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
