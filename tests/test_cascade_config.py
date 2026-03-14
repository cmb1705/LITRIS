"""Tests for cascade configuration fields and schema additions."""

from pathlib import Path

from src.analysis.schemas import ExtractionResult
from src.config import ProcessingConfig


def test_processing_config_cascade_defaults():
    """ProcessingConfig has cascade fields with correct defaults."""
    config = ProcessingConfig()
    assert config.cascade_enabled is True
    assert config.companion_dir is None
    assert config.arxiv_enabled is True
    assert config.marker_enabled is True


def test_processing_config_cascade_overrides():
    """Cascade fields can be set via constructor."""
    config = ProcessingConfig(
        cascade_enabled=False,
        companion_dir=Path("/tmp/companions"),
        arxiv_enabled=False,
        marker_enabled=False,
    )
    assert config.cascade_enabled is False
    assert config.companion_dir == Path("/tmp/companions")
    assert config.arxiv_enabled is False
    assert config.marker_enabled is False


def test_extraction_result_has_extraction_method():
    """ExtractionResult accepts optional extraction_method field."""
    result = ExtractionResult(
        paper_id="test-001",
        success=True,
        extraction_method="companion",
    )
    assert result.extraction_method == "companion"


def test_extraction_result_extraction_method_default_none():
    """ExtractionResult defaults extraction_method to None."""
    result = ExtractionResult(paper_id="test-002", success=False)
    assert result.extraction_method is None
