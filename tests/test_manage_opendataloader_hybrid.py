"""Tests for the managed OpenDataLoader hybrid server controller."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from src.config import ManagedHybridServerConfig, ProcessingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "manage_opendataloader_hybrid.py"


def _load_manage_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "litris_manage_hybrid_script",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_server_commands_uses_configured_python_and_fixed_ports(tmp_path):
    """Launch commands should come from the configured Python install and URLs."""
    module = _load_manage_script()
    python_dir = tmp_path / "Python310"
    scripts_dir = python_dir / "Scripts"
    scripts_dir.mkdir(parents=True)
    python_exe = python_dir / "python.exe"
    python_exe.write_text("")
    hybrid_exe = scripts_dir / "opendataloader-pdf-hybrid.exe"
    hybrid_exe.write_text("")

    processing = ProcessingConfig(
        opendataloader_hybrid_python_executable=str(python_exe),
        opendataloader_hybrid_device="cuda",
        opendataloader_hybrid_picture_description_prompt="Describe the chart.",
        opendataloader_hybrid_servers={
            "formula_picture": ManagedHybridServerConfig(
                url="http://127.0.0.1:5008",
                enrich_formula=True,
                enrich_picture_description=True,
            )
        },
    )

    targets = module.build_server_commands(processing, project_root=tmp_path)

    target = targets["formula_picture"]
    assert target.url == "http://127.0.0.1:5008"
    assert target.log_path == (
        tmp_path / "data" / "logs" / "opendataloader_hybrid_formula_picture.log"
    )
    assert target.command == (
        str(hybrid_exe),
        "--host",
        "127.0.0.1",
        "--port",
        "5008",
        "--device",
        "cuda",
        "--enrich-formula",
        "--enrich-picture-description",
        "--picture-description-prompt",
        "Describe the chart.",
    )
