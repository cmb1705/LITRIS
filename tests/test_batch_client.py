"""Tests for Anthropic batch extraction helpers."""

import re

from src.analysis.batch_client import BatchExtractionClient


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
