"""OpenDataLoader PDF extraction.

OpenDataLoader PDF (https://github.com/opendataloader/opendataloader-pdf)
converts PDFs to markdown using Java-based layout analysis. It handles:
- Complex layouts (multi-column, tables, figures)
- Reading order detection (xycut algorithm)
- Tagged PDF structure trees
- Hybrid mode with AI-assisted OCR (optional)

OpenDataLoader PDF is an optional dependency; this module gracefully
degrades if it is not installed or Java is not available.

Requires Java 11+ at runtime (JVM spawned per convert() call).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from src.utils.logging_config import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

try:
    import opendataloader_pdf

    OPENDATALOADER_AVAILABLE = True
except ImportError:
    OPENDATALOADER_AVAILABLE = False
    opendataloader_pdf = None

# Minimum Java version required by opendataloader-pdf
_MIN_JAVA_VERSION = 11


def _find_java() -> str | None:
    """Find a Java 11+ executable.

    Checks PATH first, then common Windows install locations for
    Microsoft OpenJDK.

    Returns:
        Path to java executable, or None if not found or too old.
    """
    candidates: list[str] = []

    # Check PATH
    java_on_path = shutil.which("java")
    if java_on_path:
        candidates.append(java_on_path)

    # Common Windows OpenJDK locations
    import os

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    ms_jdk_dir = Path(program_files) / "Microsoft"
    if ms_jdk_dir.is_dir():
        for jdk_dir in sorted(ms_jdk_dir.glob("jdk-*"), reverse=True):
            java_exe = jdk_dir / "bin" / "java.exe"
            if java_exe.is_file():
                candidates.append(str(java_exe))

    for java_path in candidates:
        version = _get_java_version(java_path)
        if version is not None and version >= _MIN_JAVA_VERSION:
            return java_path

    return None


def _get_java_version(java_path: str) -> int | None:
    """Parse major Java version from `java -version` output.

    Returns:
        Major version number (e.g. 21), or None on failure.
    """
    try:
        result = subprocess.run(
            [java_path, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
        )
        # java -version outputs to stderr
        output = result.stderr or result.stdout
        for line in output.splitlines():
            line = line.strip().strip('"')
            # Patterns: "21.0.10" or "1.8.0_361"
            if "version" in line.lower():
                # Extract version string between quotes
                parts = line.split('"')
                if len(parts) >= 2:
                    ver_str = parts[1]
                else:
                    ver_str = line.split()[-1]
                # Parse major version
                ver_parts = ver_str.split(".")
                major = int(ver_parts[0])
                # Java 1.x convention: 1.8 means Java 8
                if major == 1 and len(ver_parts) > 1:
                    return int(ver_parts[1])
                return major
    except Exception:
        pass
    return None


# Cache the Java path at module load time
_JAVA_PATH: str | None = _find_java() if OPENDATALOADER_AVAILABLE else None


def is_available() -> bool:
    """Check if OpenDataLoader PDF is installed and Java 11+ is available.

    Returns:
        True if OpenDataLoader PDF can be used for extraction.
    """
    return OPENDATALOADER_AVAILABLE and _JAVA_PATH is not None


def get_java_path() -> str | None:
    """Return the detected Java path, for use in PATH overrides.

    Returns:
        Path to java executable, or None.
    """
    return _JAVA_PATH


def extract_with_opendataloader(
    pdf_path: Path,
    mode: str = "fast",
    use_struct_tree: bool = True,
) -> str | None:
    """Extract text from PDF using OpenDataLoader PDF.

    OpenDataLoader produces markdown output preserving document structure
    (headers, tables, reading order).

    Args:
        pdf_path: Path to PDF file.
        mode: Extraction mode. "fast" uses Java-only layout analysis.
            "hybrid" enables AI-assisted processing (requires hybrid server).
        use_struct_tree: Use PDF structure tree for tagged PDFs. Can
            improve output quality on publisher PDFs with accessibility tags.

    Returns:
        Extracted markdown text, or None if extraction fails.
    """
    if not OPENDATALOADER_AVAILABLE:
        logger.debug("opendataloader-pdf not installed, skipping")
        return None

    if _JAVA_PATH is None:
        logger.debug("Java 11+ not found, skipping OpenDataLoader")
        return None

    if not pdf_path.exists():
        logger.warning(f"PDF not found: {pdf_path}")
        return None

    logger.info(f"OpenDataLoader extraction ({mode}): {pdf_path.name}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Ensure Java 11+ is on PATH for the subprocess
            import os

            env = os.environ.copy()
            java_dir = str(Path(_JAVA_PATH).parent)
            env["PATH"] = java_dir + os.pathsep + env.get("PATH", "")

            # Build convert kwargs
            kwargs: dict = {
                "input_path": [str(pdf_path)],
                "output_dir": tmpdir,
                "format": "markdown",
                "quiet": True,
                "image_output": "off",
                "use_struct_tree": use_struct_tree,
            }

            if mode == "hybrid":
                kwargs["hybrid"] = "docling-fast"
                kwargs["hybrid_mode"] = "auto"

            # Temporarily override PATH so JVM subprocess finds Java 11+
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = java_dir + os.pathsep + old_path
            try:
                opendataloader_pdf.convert(**kwargs)
            finally:
                os.environ["PATH"] = old_path

            # Read the output markdown file
            md_files = list(Path(tmpdir).rglob("*.md"))
            if not md_files:
                logger.debug(
                    f"OpenDataLoader produced no markdown for {pdf_path.name}"
                )
                return None

            text = md_files[0].read_text(encoding="utf-8")

            if text and len(text.split()) > 100:
                logger.info(
                    f"OpenDataLoader ({mode}): extracted {len(text.split())} "
                    f"words from {pdf_path.name}"
                )
                return text

            logger.debug(
                f"OpenDataLoader: insufficient text from {pdf_path.name}"
            )
            return None

    except FileNotFoundError as e:
        if "java" in str(e).lower():
            logger.warning(
                f"Java not found during OpenDataLoader extraction: {e}"
            )
        else:
            logger.warning(
                f"OpenDataLoader extraction failed for {pdf_path.name}: {e}"
            )
        return None
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"OpenDataLoader CLI failed for {pdf_path.name}: "
            f"exit code {e.returncode}"
        )
        return None
    except Exception as e:
        logger.warning(
            f"OpenDataLoader extraction failed for {pdf_path.name}: {e}"
        )
        return None
