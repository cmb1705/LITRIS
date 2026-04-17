"""Tests for the extraction preflight script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from src.config import ManagedHybridServerConfig, ProcessingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "preflight.py"


def _load_preflight_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "litris_preflight_script",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_check_managed_hybrid_pool_reports_endpoint_health(monkeypatch):
    """Preflight should check every configured managed hybrid endpoint."""
    module = _load_preflight_script()
    processing = ProcessingConfig(
        opendataloader_hybrid_python_executable=r"C:\Python310\python.exe",
        opendataloader_hybrid_servers={
            "base": ManagedHybridServerConfig(url="http://127.0.0.1:5002"),
            "formula": ManagedHybridServerConfig(
                url="http://127.0.0.1:5004",
                enrich_formula=True,
            ),
        },
    )
    ok_messages: list[str] = []
    warn_messages: list[str] = []

    monkeypatch.setattr(module, "ok", ok_messages.append)
    monkeypatch.setattr(
        module,
        "warn",
        lambda msg, fix="": (warn_messages.append(msg), (msg, fix))[1],
    )
    monkeypatch.setattr(
        "src.extraction.opendataloader_extractor.hybrid_server_executable_for_python",
        lambda python_executable, allow_path_fallback=False: (
            r"C:\Python310\Scripts\opendataloader-pdf-hybrid.exe"
        ),
    )
    monkeypatch.setattr(
        "src.extraction.opendataloader_extractor._hybrid_server_executable",
        lambda: r"C:\fallback\opendataloader-pdf-hybrid.exe",
    )
    monkeypatch.setattr(
        "src.extraction.opendataloader_extractor.is_hybrid_server_reachable",
        lambda url=None, timeout_seconds=1.0: url == "http://127.0.0.1:5002",
    )

    issues = module.check_managed_hybrid_pool(processing)

    assert len(issues) == 1
    assert any("managed pool configured (2 endpoints)" in msg for msg in ok_messages)
    assert any("base responding" in msg for msg in ok_messages)
    assert warn_messages == ["Hybrid backend formula not running on http://127.0.0.1:5004"]
