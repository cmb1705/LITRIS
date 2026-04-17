"""Tests for Anthropic batch extraction helpers."""

import re

from src.analysis.batch_client import BatchExtractionClient, _build_extraction
from src.analysis.dimensions import DimensionProfile
from src.analysis.schemas import DimensionedExtraction


def test_batch_custom_id_is_anthropic_safe() -> None:
    """Anthropic batch custom IDs should use only allowed characters."""
    custom_id = BatchExtractionClient._build_custom_id(
        "2BN8YN3J_RF3USQ22",
        6,
    )

    assert custom_id == "2BN8YN3J_RF3USQ22__pass6"
    assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", custom_id)
    assert BatchExtractionClient._parse_custom_id(custom_id) == (
        "2BN8YN3J_RF3USQ22",
        6,
    )


def test_batch_custom_id_parser_accepts_legacy_colon_separator() -> None:
    """Older saved batch state should still parse after the separator change."""
    assert BatchExtractionClient._parse_custom_id("2BN8YN3J_RF3USQ22:pass3") == (
        "2BN8YN3J_RF3USQ22",
        3,
    )


def test_build_extraction_maps_frozen_profile_fields_to_canonical_dimensions() -> None:
    """Explicit batch profiles should reassemble into canonical dimension ids."""

    profile = DimensionProfile(
        profile_id="custom_semantic_v1",
        version="1.0.0",
        title="Custom Semantic Profile",
        sections=[{"id": "research_core", "label": "Research Core", "order": 1}],
        dimensions=[
            {
                "id": "custom_dimension",
                "label": "Custom Dimension",
                "question": "What is the custom semantic finding?",
                "section": "research_core",
                "order": 1,
                "core": True,
                "legacy_field_name": "q01_research_question",
                "legacy_short_name": "q01",
            }
        ],
    )

    extraction = _build_extraction(
        paper_id="paper_001",
        answers={"q01_research_question": "Mapped answer"},
        model="claude-test",
        field_to_dimension={"q01_research_question": "custom_dimension"},
        profile=profile,
        prompt_version="2.0.0",
    )

    assert isinstance(extraction, DimensionedExtraction)
    assert extraction.dimensions["custom_dimension"] == "Mapped answer"
    assert extraction.profile_fingerprint == profile.fingerprint
