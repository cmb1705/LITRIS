"""Tests for OpenDataLoader hybrid backend helpers."""

from pathlib import Path
from unittest.mock import patch

from src.extraction import opendataloader_extractor


def test_hybrid_server_executable_prefers_current_interpreter_dir(tmp_path):
    """Prefer the launcher installed in the active virtualenv."""
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()
    local_exe = scripts_dir / "opendataloader-pdf-hybrid.exe"
    local_exe.write_text("")

    with (
        patch.object(
            opendataloader_extractor.sys,
            "executable",
            str(scripts_dir / "python.exe"),
        ),
        patch.object(opendataloader_extractor.sys, "prefix", str(tmp_path)),
        patch(
            "src.extraction.opendataloader_extractor.shutil.which",
            return_value=r"C:\global\opendataloader-pdf-hybrid.exe",
        ),
    ):
        assert opendataloader_extractor._hybrid_server_executable() == str(local_exe)


def test_hybrid_server_executable_falls_back_to_path(tmp_path):
    """Use PATH lookup when the active virtualenv has no launcher."""
    with (
        patch.object(
            opendataloader_extractor.sys,
            "executable",
            str(tmp_path / "Scripts" / "python.exe"),
        ),
        patch.object(opendataloader_extractor.sys, "prefix", str(tmp_path)),
        patch(
            "src.extraction.opendataloader_extractor.shutil.which",
            side_effect=[
                r"C:\global\opendataloader-pdf-hybrid.exe",
                None,
            ],
        ),
    ):
        assert opendataloader_extractor._hybrid_server_executable() == (
            r"C:\global\opendataloader-pdf-hybrid.exe"
        )
