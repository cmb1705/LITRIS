"""Dimension-specific and group-based search for SemanticAnalysis embeddings.

Provides convenience wrappers around SearchEngine.search() that filter
by specific dimension chunk types (dim_q01..dim_q40) or dimension groups
(research_core, methodology, context, meta, scholarly, impact).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.analysis.dimensions import (
    LEGACY_PROFILE_ID,
    DimensionRegistry,
    get_default_dimension_registry,
)
from src.analysis.schemas import SemanticAnalysis

if TYPE_CHECKING:
    from src.query.search import EnrichedResult, SearchEngine


def _get_registry(engine: SearchEngine | None = None):
    if engine is not None and isinstance(getattr(engine, "dimension_registry", None), DimensionRegistry):
        return engine.dimension_registry
    return get_default_dimension_registry()


_LEGACY_REGISTRY = get_default_dimension_registry()
VALID_DIMENSIONS = [
    dimension.legacy_short_name or dimension.id
    for dimension in _LEGACY_REGISTRY.get_profile(LEGACY_PROFILE_ID).ordered_dimensions
]
VALID_GROUPS = _LEGACY_REGISTRY.get_group_names(profile_id=LEGACY_PROFILE_ID)


def search_dimension(
    engine: SearchEngine,
    query: str,
    dimension: str,
    top_k: int = 10,
    **kwargs,
) -> list[EnrichedResult]:
    """Search within a specific dimension's embeddings.

    Args:
        engine: SearchEngine instance.
        query: Natural language search query.
        dimension: Dimension identifier, e.g. "q01" or "q01_research_question".
            Accepts either short form ("q07") or full field name ("q07_methods").
        top_k: Number of results to return.
        **kwargs: Additional filters passed to SearchEngine.search()
            (year_min, year_max, collections, item_types, quality_min, etc.)

    Returns:
        List of EnrichedResult objects from that dimension only.

    Raises:
        ValueError: If the dimension identifier is invalid.
    """
    registry = _get_registry(engine)
    definition = _resolve_dimension_definition(dimension, engine=engine)
    chunk_type = definition.chunk_type

    return engine.search(
        query=query,
        top_k=top_k,
        chunk_types=[chunk_type],
        **kwargs,
    )


def search_group(
    engine: SearchEngine,
    query: str,
    group: str,
    top_k: int = 10,
    **kwargs,
) -> list[EnrichedResult]:
    """Search across all dimensions in a thematic group.

    Args:
        engine: SearchEngine instance.
        query: Natural language search query.
        group: Group name: research_core, methodology, context, meta,
            scholarly, or impact.
        top_k: Number of results to return.
        **kwargs: Additional filters passed to SearchEngine.search().

    Returns:
        List of EnrichedResult objects from all dimensions in the group.

    Raises:
        ValueError: If the group name is invalid.
    """
    registry = _get_registry(engine)
    if group not in registry.get_group_names():
        raise ValueError(
            f"Invalid group '{group}'. Valid groups: {', '.join(registry.get_group_names())}"
        )

    chunk_types = [
        definition.chunk_type
        for definition in registry.get_section_dimensions(group)
    ]

    return engine.search(
        query=query,
        top_k=top_k,
        chunk_types=chunk_types,
        **kwargs,
    )


def _resolve_dimension_definition(dimension: str, engine: SearchEngine | None = None):
    """Resolve a dimension identifier to the active profile definition.

    Accepts:
        - Canonical IDs: ``"thesis"``
        - Legacy short form: ``"q07"``
        - Legacy full form: ``"q07_methods"``
        - Role aliases: ``"methods"``

    Raises:
        ValueError: If the dimension identifier is invalid.
    """
    registry = _get_registry(engine)
    definition = registry.resolve_optional_dimension(dimension)
    if definition:
        return definition
    role_match = registry.resolve_role(dimension)
    if role_match:
        return role_match
    raise ValueError(
        f"Invalid dimension '{dimension}'. Use a canonical ID, legacy qNN alias, "
        "or a supported role alias."
    )


def _normalize_dimension(dimension: str) -> str:
    """Normalize a legacy dimension identifier to its short ``qNN`` alias."""

    if not dimension.startswith("q"):
        raise ValueError(f"Invalid dimension '{dimension}'. Use a legacy qNN alias.")
    definition = _resolve_dimension_definition(dimension)
    if not definition.legacy_short_name:
        raise ValueError(f"Invalid dimension '{dimension}'. Use a legacy qNN alias.")
    return definition.legacy_short_name
