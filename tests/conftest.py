"""Shared pytest fixtures and configuration."""

import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest
import yaml


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_config_dict(tmp_path: Path) -> dict:
    """Provide a sample configuration dictionary for testing.

    Uses tmp_path for platform-agnostic paths that work on Windows, Mac, and Linux.
    """
    # Create mock Zotero structure in temp directory
    zotero_dir = tmp_path / "zotero"
    zotero_dir.mkdir(parents=True, exist_ok=True)
    db_path = zotero_dir / "zotero.sqlite"
    db_path.touch()
    storage_path = zotero_dir / "storage"
    storage_path.mkdir(parents=True, exist_ok=True)

    return {
        "zotero": {
            "database_path": str(db_path),
            "storage_path": str(storage_path),
        },
        "extraction": {
            "mode": "cli",
            "model": "claude-opus-4-5-20251101",
            "max_tokens": 100000,
            "timeout": 120,
        },
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
        },
        "storage": {
            "chroma_path": str(tmp_path / "data" / "chroma"),
            "cache_path": str(tmp_path / "data" / "cache"),
            "collection_name": "literature_review_test",
        },
        "processing": {
            "batch_size": 10,
            "ocr_enabled": False,
            "min_text_length": 100,
        },
    }


@pytest.fixture
def temp_config_file(
    tmp_path: Path, sample_config_dict: dict
) -> Generator[Path, None, None]:
    """Create a temporary config.yaml file for testing."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sample_config_dict, f)
    yield config_path


@pytest.fixture
def temp_index_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary directory for test index outputs."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    yield index_dir


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary directory for test cache."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yield cache_dir


@pytest.fixture
def mock_zotero_db(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a mock Zotero database file for testing."""
    db_path = tmp_path / "zotero.sqlite"
    db_path.touch()
    yield db_path


@pytest.fixture
def mock_zotero_storage(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a mock Zotero storage directory for testing."""
    storage_path = tmp_path / "storage"
    storage_path.mkdir(parents=True, exist_ok=True)
    yield storage_path


@pytest.fixture
def sample_pdf_path(project_root: Path) -> Path:
    """Get path to sample PDF for testing."""
    return project_root / "tests" / "fixtures" / "sample_papers" / "sample.pdf"


@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    """Create a mock Anthropic client for testing without API calls."""
    client = MagicMock()
    client.messages.create.return_value = MagicMock(
        content=[MagicMock(text='{"summary": "Test summary"}')]
    )
    return client


@pytest.fixture(autouse=True)
def clean_env() -> Generator[None, None, None]:
    """Clean environment variables before each test."""
    # Store original values
    original_env = {
        key: os.environ.get(key)
        for key in ["ANTHROPIC_API_KEY", "ZOTERO_DATABASE_PATH", "ZOTERO_STORAGE_PATH"]
    }

    yield

    # Restore original values
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
