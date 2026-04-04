"""Tests for dimension registry compatibility behavior."""

from src.analysis.dimensions import (
    LEGACY_PROFILE_ID,
    DimensionRegistry,
    build_legacy_dimension_profile,
)


def test_get_profile_falls_back_to_active_profile_for_unknown_ids() -> None:
    """Unknown stored profile IDs should not break downstream lookups."""
    registry = DimensionRegistry(
        profiles={LEGACY_PROFILE_ID: build_legacy_dimension_profile()},
        active_profile_id=LEGACY_PROFILE_ID,
    )

    assert registry.get_profile("stp_cas_v1").profile_id == LEGACY_PROFILE_ID
    assert registry.resolve_role("thesis", profile_id="stp_cas_v1") is not None
