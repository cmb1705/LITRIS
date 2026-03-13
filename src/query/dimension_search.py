"""Dimension-specific and group-based search for SemanticAnalysis embeddings.

Provides convenience wrappers around SearchEngine.search() that filter
by specific dimension chunk types (dim_q01..dim_q40) or dimension groups
(research_core, methodology, context, meta, scholarly, impact).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.analysis.schemas import SemanticAnalysis

if TYPE_CHECKING:
    from src.query.search import EnrichedResult, SearchEngine


# Valid dimension names for search (e.g., "q01", "q02", ..., "q40")
VALID_DIMENSIONS = [f"q{i:02d}" for i in range(1, 41)]

# Valid group names
VALID_GROUPS = list(SemanticAnalysis.DIMENSION_GROUPS.keys())


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
    dim_key = _normalize_dimension(dimension)
    chunk_type = f"dim_{dim_key}"

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
    if group not in SemanticAnalysis.DIMENSION_GROUPS:
        raise ValueError(
            f"Invalid group '{group}'. Valid groups: {', '.join(VALID_GROUPS)}"
        )

    # Get dimension field names for this group and convert to chunk types
    field_names = SemanticAnalysis.DIMENSION_GROUPS[group]
    chunk_types = [f"dim_{name[:3]}" for name in field_names]

    return engine.search(
        query=query,
        top_k=top_k,
        chunk_types=chunk_types,
        **kwargs,
    )


def _normalize_dimension(dimension: str) -> str:
    """Normalize a dimension identifier to short form (e.g., "q07").

    Accepts:
        - Short form: "q07", "q01"
        - Full field name: "q07_methods", "q01_research_question"

    Returns:
        Short dimension key like "q07".

    Raises:
        ValueError: If the dimension identifier is invalid.
    """
    # If it's already short form
    if dimension in VALID_DIMENSIONS:
        return dimension

    # Try extracting short form from full field name (e.g., "q07_methods" -> "q07")
    short = dimension[:3]
    if short in VALID_DIMENSIONS:
        # Verify the full name matches a real field
        matching = [f for f in SemanticAnalysis.DIMENSION_FIELDS if f.startswith(short)]
        if matching and (dimension == short or dimension in matching):
            return short

    raise ValueError(
        f"Invalid dimension '{dimension}'. Use short form (q01-q40) "
        f"or full field name (e.g., q07_methods)."
    )
