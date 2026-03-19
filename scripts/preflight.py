#!/usr/bin/env python
"""Preflight check for LITRIS extraction pipeline.

Validates all dependencies, tools, and models are installed and accessible
before committing to a long index build.

Usage:
    python scripts/preflight.py
    python scripts/preflight.py --fix    # Show install commands for missing deps
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

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
    return (msg, fix)


def warn(msg: str, fix: str = "") -> tuple[str, str]:
    print(f"  {YELLOW}[MISS]{RESET} {msg}")
    return (msg, fix)


def info(msg: str) -> None:
    print(f"  {CYAN}[INFO]{RESET} {msg}")


def section(title: str) -> None:
    print(f"\n{BOLD}=== {title} ==={RESET}\n")


def check_extraction_pipeline() -> list[tuple[str, str]]:
    """Check PDF extraction dependencies."""
    issues: list[tuple[str, str]] = []

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
        issues.append(warn(
            "Marker not installed (ML PDF parser for complex layouts)",
            "pip install marker-pdf",
        ))

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
            issues.append(warn(
                "Java 11+ not found (required for OpenDataLoader PDF)",
                "winget install Microsoft.OpenJDK.21",
            ))
    except ImportError:
        issues.append(warn(
            "OpenDataLoader PDF not installed (Java PDF layout analysis)",
            "pip install opendataloader-pdf",
        ))

    # Tesseract
    try:
        import pytesseract
        ver = pytesseract.get_tesseract_version()
        ok(f"Tesseract {ver}")
    except Exception as exc:
        issues.append(fail(
            f"Tesseract: {exc}",
            "winget install --id UB-Mannheim.TesseractOCR (then add to PATH)",
        ))

    # Poppler
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        try:
            result = subprocess.run(
                [pdftoppm, "-v"], capture_output=True, text=True, timeout=5,
            )
            ver = (result.stdout.strip() or result.stderr.strip()).split("\n")[0]
            ok(f"Poppler ({ver})")
        except Exception:
            ok(f"Poppler (pdftoppm found at {pdftoppm})")
    else:
        issues.append(fail(
            "Poppler: pdftoppm not on PATH (needed for OCR page rendering)",
            "winget install --id oschwartz10612.Poppler (then add bin/ to PATH)",
        ))

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
        ollama_local = os.path.expandvars(
            r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
        )
        if os.path.exists(ollama_local):
            ollama = ollama_local

    if ollama:
        ok(f"Ollama: {ollama}")

        # Check embedding model
        try:
            result = subprocess.run(
                [ollama, "list"], capture_output=True, text=True, timeout=10,
            )
            if "qwen3-embedding" in result.stdout:
                ok("Qwen3-Embedding model loaded")
            else:
                issues.append(warn(
                    "Qwen3-Embedding not pulled",
                    "ollama pull qwen3-embedding:8b-q8_0",
                ))
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
        ollama_local = os.path.expandvars(
            r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
        )
        if os.path.exists(ollama_local):
            ollama = ollama_local

    if ollama:
        try:
            result = subprocess.run(
                [ollama, "list"], capture_output=True, text=True, timeout=10,
            )
            if "glm-ocr" in result.stdout:
                ok("GLM-OCR model available in Ollama")
            else:
                info("GLM-OCR not pulled yet (ollama pull glm-ocr)")
        except Exception:
            info("Could not check Ollama models")

    return issues


def check_config() -> list[tuple[str, str]]:
    """Check LITRIS configuration."""
    issues: list[tuple[str, str]] = []

    section("Configuration")

    config_path = Path("config.yaml")
    if config_path.exists():
        ok(f"config.yaml found ({config_path.stat().st_size} bytes)")

        try:
            from src.config import Config
            config = Config.load()
            ok(f"Config loaded (provider: {config.extraction.provider}, "
               f"mode: {config.extraction.mode})")
            if config.extraction.reasoning_effort:
                ok(f"Reasoning effort: {config.extraction.reasoning_effort}")
        except Exception as exc:
            issues.append(fail(f"Config load error: {exc}"))
    else:
        issues.append(fail(
            "config.yaml not found",
            "cp config.example.yaml config.yaml",
        ))

    # Check index directory
    index_dir = Path("data/index")
    if index_dir.exists():
        papers_json = index_dir / "papers.json"
        extractions_json = index_dir / "extractions.json"
        papers_size = papers_json.stat().st_size if papers_json.exists() else 0
        extractions_size = extractions_json.stat().st_size if extractions_json.exists() else 0
        ok(f"Index directory: papers.json ({papers_size:,} bytes), "
           f"extractions.json ({extractions_size:,} bytes)")
    else:
        info("No existing index (fresh build)")

    return issues


def main() -> int:
    show_fixes = "--fix" in sys.argv

    print(f"\n{BOLD}LITRIS Extraction Pipeline Preflight{RESET}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Working directory: {os.getcwd()}")

    all_issues: list[tuple[str, str]] = []
    all_issues.extend(check_extraction_pipeline())
    all_issues.extend(check_llm_providers())
    all_issues.extend(check_embeddings())
    all_issues.extend(check_optional())
    all_issues.extend(check_config())

    # Summary
    section("Summary")

    if not all_issues:
        print(f"  {GREEN}{BOLD}All checks passed. Ready to build.{RESET}")
        return 0

    fails = [i for i in all_issues if "[FAIL]" in str(i)]
    warns = [i for i in all_issues if "[MISS]" in str(i)]

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
