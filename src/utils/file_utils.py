"""File utilities for the Literature Review system."""

import hashlib
import json
from pathlib import Path
from typing import Any

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def ensure_directory(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists.

    Returns:
        The same path, for chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Calculate hash of a file.

    Args:
        path: Path to file.
        algorithm: Hash algorithm (sha256, md5, etc.).

    Returns:
        Hex digest of file hash.
    """
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_read_json(path: Path, default: Any = None) -> Any:
    """Safely read a JSON file.

    Args:
        path: Path to JSON file.
        default: Default value if file doesn't exist or is invalid.

    Returns:
        Parsed JSON data or default value.
    """
    if not path.exists():
        logger.debug(f"JSON file not found: {path}")
        return default

    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in {path}: {e}")
        return default
    except OSError as e:
        logger.warning(f"Error reading {path}: {e}")
        return default


def safe_write_json(path: Path, data: Any, indent: int = 2) -> bool:
    """Safely write data to a JSON file.

    Args:
        path: Path to JSON file.
        data: Data to serialize to JSON.
        indent: Indentation level for pretty printing.

    Returns:
        True if successful, False otherwise.
    """
    try:
        ensure_directory(path.parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        return True
    except OSError as e:
        logger.error(f"Error writing {path}: {e}")
        return False


def find_pdf_files(directory: Path, recursive: bool = True) -> list[Path]:
    """Find all PDF files in a directory.

    Args:
        directory: Directory to search.
        recursive: Whether to search subdirectories.

    Returns:
        List of paths to PDF files.
    """
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []

    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(directory.glob(pattern))


def get_relative_path(path: Path, base: Path) -> Path:
    """Get path relative to base directory.

    Args:
        path: Absolute or relative path.
        base: Base directory for relative path.

    Returns:
        Path relative to base, or original path if not relative.
    """
    try:
        return path.relative_to(base)
    except ValueError:
        return path


def sanitize_filename(name: str, max_length: int = 200) -> str:
    """Sanitize a string for use as a filename.

    Args:
        name: Original string.
        max_length: Maximum filename length.

    Returns:
        Sanitized filename.
    """
    # Replace problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "_")

    # Remove leading/trailing whitespace and dots
    name = name.strip(". ")

    # Truncate if too long
    if len(name) > max_length:
        name = name[:max_length]

    # Ensure not empty
    if not name:
        name = "unnamed"

    return name


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable form.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable size string (e.g., "1.5 MB").
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"
