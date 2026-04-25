#!/usr/bin/env python
"""Preflight check for LITRIS extraction pipeline.

Validates all dependencies, tools, and models are installed and accessible
before committing to a long index build.

Usage:
    python scripts/preflight.py
    python scripts/preflight.py --fix    # Show install commands for missing deps
"""

import argparse
import os
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ANSI colors (Windows 10+ supports these)
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {GREEN}[OK]{RESET}   {msg}")


def fail(msg: str, fix: str = "") -> tuple[str, str]:
    print(f"  {RED}[FAIL]{RESET} {msg}")
    return (f"FAIL: {msg}", fix)


def warn(msg: str, fix: str = "") -> tuple[str, str]:
    print(f"  {YELLOW}[MISS]{RESET} {msg}")
    return (f"MISS: {msg}", fix)


def info(msg: str) -> None:
    print(f"  {CYAN}[INFO]{RESET} {msg}")


def section(title: str) -> None:
    print(f"\n{BOLD}=== {title} ==={RESET}\n")


def load_runtime_config(config_path: Path) -> tuple[Any | None, Exception | None]:
    """Load config.yaml once so other checks can reuse the same config."""
    if not config_path.exists():
        return None, FileNotFoundError(f"{config_path} not found")

    try:
        from src.config import Config

        return Config.load(config_path), None
    except Exception as exc:  # pragma: no cover - exercised in integration
        return None, exc


def check_managed_hybrid_pool(processing: Any | None) -> list[tuple[str, str]]:
    """Check the configured managed hybrid pool, if any."""
    issues: list[tuple[str, str]] = []
    from src.extraction.opendataloader_extractor import (
        _hybrid_server_executable,
        build_managed_server_specs,
        hybrid_server_executable_for_python,
        is_hybrid_server_reachable,
    )

    python_executable = (
        getattr(processing, "opendataloader_hybrid_python_executable", None)
        if processing is not None
        else None
    )
    hybrid_required = bool(
        processing is not None
        and (
            getattr(processing, "opendataloader_hybrid_enabled", False)
            or getattr(processing, "opendataloader_mode", "fast") == "hybrid"
        )
    )
    issue = fail if hybrid_required else warn
    if python_executable:
        ok(f"OpenDataLoader hybrid Python: {python_executable}")

    hybrid_exe = None
    if python_executable:
        hybrid_exe = hybrid_server_executable_for_python(
            python_executable,
            allow_path_fallback=False,
        )
    if hybrid_exe is None:
        hybrid_exe = _hybrid_server_executable()

    if hybrid_exe:
        ok(f"OpenDataLoader hybrid executable: {hybrid_exe}")
    else:
        issues.append(
            issue(
                "OpenDataLoader hybrid executable not found",
                'pip install "opendataloader-pdf[hybrid]"',
            )
        )

    managed_specs = build_managed_server_specs(
        getattr(processing, "opendataloader_hybrid_servers", None)
        if processing is not None
        else None
    )
    if managed_specs:
        ok(f"OpenDataLoader hybrid managed pool configured ({len(managed_specs)} endpoints)")
        for spec in managed_specs:
            if is_hybrid_server_reachable(spec.url):
                ok(f"Hybrid backend {spec.name} responding on {spec.url}")
            else:
                issues.append(
                    issue(
                        f"Hybrid backend {spec.name} not running on {spec.url}",
                        "python scripts/manage_opendataloader_hybrid.py start",
                    )
                )
        return issues

    if is_hybrid_server_reachable():
        ok("OpenDataLoader hybrid backend responding on http://127.0.0.1:5002")
    else:
        issues.append(
            issue(
                "OpenDataLoader hybrid backend not running on http://127.0.0.1:5002",
                "opendataloader-pdf-hybrid --port 5002",
            )
        )
    return issues


def check_extraction_pipeline(config: Any | None = None) -> list[tuple[str, str]]:
    """Check PDF extraction dependencies."""
    issues: list[tuple[str, str]] = []
    processing = getattr(config, "processing", None)

    section("PDF Extraction Pipeline")

    # PyMuPDF
    try:
        import fitz

        ok(f"PyMuPDF {fitz.version[0]}")
    except ImportError:
        issues.append(fail("PyMuPDF not installed", "pip install pymupdf"))

    # Marker
    try:
        import marker
        from marker.converters.pdf import PdfConverter  # noqa: F401

        ver = getattr(marker, "__version__", "unknown")
        ok(f"Marker {ver} (ML-based PDF parser: tables, equations, layouts)")
    except ImportError:
        issues.append(
            warn(
                "Marker not installed (ML PDF parser for complex layouts)",
                "pip install marker-pdf",
            )
        )

    # OpenDataLoader PDF
    try:
        import opendataloader_pdf  # noqa: F401

        ver = getattr(opendataloader_pdf, "__version__", "unknown")
        ok(f"OpenDataLoader PDF {ver} (Java-based layout analysis)")

        # Check Java 11+
        from src.extraction.opendataloader_extractor import _find_java

        java_path = _find_java()
        if java_path:
            ok(f"Java 11+ found: {java_path}")
        else:
            issues.append(
                warn(
                    "Java 11+ not found (required for OpenDataLoader PDF)",
                    "winget install Microsoft.OpenJDK.21",
                )
            )

        hybrid_specs = (
            find_spec("docling"),
            find_spec("fastapi"),
            find_spec("uvicorn"),
        )
        if all(spec is not None for spec in hybrid_specs):
            ok("OpenDataLoader hybrid extra installed")
        else:
            issues.append(
                warn(
                    "OpenDataLoader hybrid extra not installed",
                    'pip install "opendataloader-pdf[hybrid]"',
                )
            )

        issues.extend(check_managed_hybrid_pool(processing))
    except ImportError:
        issues.append(
            warn(
                "OpenDataLoader PDF not installed (Java PDF layout analysis)",
                "pip install opendataloader-pdf",
            )
        )

    # Tesseract
    try:
        import pytesseract

        ver = pytesseract.get_tesseract_version()
        ok(f"Tesseract {ver}")
    except Exception as exc:
        issues.append(
            fail(
                f"Tesseract: {exc}",
                "winget install --id UB-Mannheim.TesseractOCR (then add to PATH)",
            )
        )

    # Poppler
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        try:
            result = subprocess.run(
                [pdftoppm, "-v"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            ver = (result.stdout.strip() or result.stderr.strip()).split("\n")[0]
            ok(f"Poppler ({ver})")
        except Exception:
            ok(f"Poppler (pdftoppm found at {pdftoppm})")
    else:
        issues.append(
            fail(
                "Poppler: pdftoppm not on PATH (needed for OCR page rendering)",
                "winget install --id oschwartz10612.Poppler (then add bin/ to PATH)",
            )
        )

    # pdf2image
    try:
        from pdf2image import convert_from_path  # noqa: F401

        ok("pdf2image (Python package)")
    except ImportError:
        issues.append(fail("pdf2image not installed", "pip install pdf2image"))

    # Pillow
    try:
        import PIL

        ok(f"Pillow {PIL.__version__}")
    except ImportError:
        issues.append(fail("Pillow not installed", "pip install Pillow"))

    # arXiv extractor
    try:
        from src.extraction import arxiv_extractor  # noqa: F401

        ok("arXiv HTML extractor")
    except ImportError as exc:
        issues.append(fail(f"arXiv extractor: {exc}"))

    return issues


def check_llm_providers() -> list[tuple[str, str]]:
    """Check LLM provider availability."""
    issues: list[tuple[str, str]] = []

    section("LLM Providers")

    # Claude CLI
    claude = shutil.which("claude")
    if claude:
        ok(f"Claude CLI: {claude}")
    else:
        issues.append(warn("Claude CLI not found", "npm i -g @anthropic-ai/claude-code"))

    # Codex CLI
    codex = shutil.which("codex")
    if codex:
        ok(f"Codex CLI: {codex}")
    else:
        issues.append(warn("Codex CLI not found", "npm i -g @openai/codex"))

    # Anthropic SDK
    try:
        import anthropic

        ok(f"anthropic SDK {anthropic.__version__}")
    except ImportError:
        issues.append(warn("anthropic SDK not installed", "pip install anthropic"))

    # OpenAI SDK
    try:
        import openai

        ok(f"openai SDK {openai.__version__}")
    except ImportError:
        issues.append(warn("openai SDK not installed", "pip install openai"))

    # Google SDK
    try:
        import google.generativeai

        ok(f"google-generativeai SDK {google.generativeai.__version__}")
    except ImportError:
        info("google-generativeai SDK not installed (optional)")

    return issues


def check_embeddings() -> list[tuple[str, str]]:
    """Check embedding infrastructure."""
    issues: list[tuple[str, str]] = []

    section("Embeddings")

    # Ollama
    ollama = shutil.which("ollama")
    if not ollama:
        ollama_local = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
        if os.path.exists(ollama_local):
            ollama = ollama_local

    if ollama:
        ok(f"Ollama: {ollama}")

        # Check embedding model
        try:
            result = subprocess.run(
                [ollama, "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if "qwen3-embedding" in result.stdout:
                ok("Qwen3-Embedding model loaded")
            else:
                issues.append(
                    warn(
                        "Qwen3-Embedding not pulled",
                        "ollama pull qwen3-embedding:8b-q8_0",
                    )
                )
        except Exception:
            info("Could not check Ollama models (server running?)")
    else:
        issues.append(fail("Ollama not found", "https://ollama.com/download"))

    # ChromaDB
    try:
        import chromadb

        ok(f"ChromaDB {chromadb.__version__}")
    except ImportError:
        issues.append(fail("ChromaDB not installed", "pip install chromadb"))

    return issues


def check_optional() -> list[tuple[str, str]]:
    """Check optional/future dependencies."""
    issues: list[tuple[str, str]] = []

    section("Optional / Future")

    # GLM-OCR
    ollama = shutil.which("ollama")
    if not ollama:
        ollama_local = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
        if os.path.exists(ollama_local):
            ollama = ollama_local

    if ollama:
        try:
            result = subprocess.run(
                [ollama, "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if "glm-ocr" in result.stdout:
                ok("GLM-OCR model available in Ollama")
            else:
                info("GLM-OCR not pulled yet (ollama pull glm-ocr)")
        except Exception:
            info("Could not check Ollama models")

    return issues


def check_config(
    config: Any | None = None,
    config_error: Exception | None = None,
    config_path: Path | None = None,
) -> list[tuple[str, str]]:
    """Check LITRIS configuration."""
    issues: list[tuple[str, str]] = []

    section("Configuration")

    config_path = config_path or Path("config.yaml")
    if config_path.exists():
        ok(f"Config file found: {config_path} ({config_path.stat().st_size} bytes)")

        if config is not None and config_error is None:
            ok(
                f"Config loaded (provider: {config.extraction.provider}, "
                f"mode: {config.extraction.mode})"
            )
            if config.extraction.reasoning_effort:
                ok(f"Reasoning effort: {config.extraction.reasoning_effort}")
        else:
            issues.append(fail(f"Config load error: {config_error}"))
    else:
        issues.append(
            fail(
                "config.yaml not found",
                "cp config.example.yaml config.yaml",
            )
        )

    # Check index directory
    project_root = Path(__file__).resolve().parents[1]
    if config is not None and hasattr(config, "get_index_path"):
        index_dir = config.get_index_path(project_root)
    else:
        index_dir = Path("data/index")
    if index_dir.exists():
        papers_json = index_dir / "papers.json"
        extractions_json = index_dir / "semantic_analyses.json"
        papers_size = papers_json.stat().st_size if papers_json.exists() else 0
        extractions_size = extractions_json.stat().st_size if extractions_json.exists() else 0
        ok(
            f"Index directory: {index_dir} | papers.json ({papers_size:,} bytes), "
            f"semantic_analyses.json ({extractions_size:,} bytes)"
        )
    else:
        info(f"No existing index at {index_dir} (fresh build)")

    return issues


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse preflight arguments."""
    parser = argparse.ArgumentParser(description="Preflight check for LITRIS extraction pipeline")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--fix", action="store_true", help="Show fix commands for missing deps")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    show_fixes = args.fix
    config, config_error = load_runtime_config(args.config)

    print(f"\n{BOLD}LITRIS Extraction Pipeline Preflight{RESET}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Working directory: {os.getcwd()}")
    print(f"Config: {args.config}")

    all_issues: list[tuple[str, str]] = []
    all_issues.extend(check_extraction_pipeline(config))
    all_issues.extend(check_llm_providers())
    all_issues.extend(check_embeddings())
    all_issues.extend(check_optional())
    all_issues.extend(check_config(config, config_error, args.config))

    # Summary
    section("Summary")

    if not all_issues:
        print(f"  {GREEN}{BOLD}All checks passed. Ready to build.{RESET}")
        return 0

    fails = [i for i in all_issues if i[0].startswith("FAIL:")]
    warns = [i for i in all_issues if i[0].startswith("MISS:")]

    if fails:
        print(f"  {RED}{len(fails)} critical issue(s){RESET} (build may fail)")
    if warns:
        print(f"  {YELLOW}{len(warns)} optional missing{RESET} (degraded quality)")

    if show_fixes or fails:
        print(f"\n{BOLD}Fix commands:{RESET}\n")
        for _msg, fix in all_issues:
            if fix:
                print(f"  {fix}")

    if fails:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
