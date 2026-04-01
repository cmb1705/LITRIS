"""Input validation for MCP tool parameters."""

from src.indexing.embeddings import CHUNK_TYPES

# Constants
MAX_QUERY_LENGTH = 1000
MAX_TOP_K = 50
MIN_YEAR = 1800
MAX_YEAR = 2100


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


def validate_query(query: str) -> str:
    """Validate search query parameter.

    Args:
        query: Search query string.

    Returns:
        Validated query string.

    Raises:
        ValidationError: If query is empty or too long.
    """
    if not query or not query.strip():
        raise ValidationError("Query cannot be empty")

    query = query.strip()

    if len(query) > MAX_QUERY_LENGTH:
        raise ValidationError(
            f"Query too long: {len(query)} characters (max: {MAX_QUERY_LENGTH})"
        )

    return query


def validate_paper_id(paper_id: str) -> str:
    """Validate paper ID parameter.

    Args:
        paper_id: Paper identifier string.

    Returns:
        Validated paper ID.

    Raises:
        ValidationError: If paper_id is empty or invalid.
    """
    if not paper_id or not paper_id.strip():
        raise ValidationError("Paper ID cannot be empty")

    paper_id = paper_id.strip()

    # Paper IDs should be alphanumeric with possible underscores/hyphens
    if not all(c.isalnum() or c in "_-" for c in paper_id):
        raise ValidationError(
            f"Invalid paper ID format: {paper_id}. "
            "Paper IDs should contain only alphanumeric characters, underscores, or hyphens."
        )

    return paper_id


def validate_top_k(top_k: int) -> int:
    """Validate top_k parameter.

    Args:
        top_k: Number of results to return.

    Returns:
        Validated top_k value (clamped to valid range).
    """
    if top_k < 1:
        return 1
    if top_k > MAX_TOP_K:
        return MAX_TOP_K
    return top_k


def validate_year(year: int, param_name: str = "year") -> int:
    """Validate year parameter.

    Args:
        year: Publication year.
        param_name: Name of parameter for error messages.

    Returns:
        Validated year value.

    Raises:
        ValidationError: If year is outside valid range.
    """
    if year < MIN_YEAR or year > MAX_YEAR:
        raise ValidationError(
            f"Invalid {param_name}: {year}. "
            f"Year must be between {MIN_YEAR} and {MAX_YEAR}."
        )
    return year


def validate_chunk_types(chunk_types: list[str]) -> list[str]:
    """Validate chunk_types parameter.

    Valid chunk types are: dim_q01 through dim_q40 (one per SemanticAnalysis
    dimension), raptor_overview, raptor_core, and abstract.

    Args:
        chunk_types: List of chunk type strings.

    Returns:
        Validated list of chunk types.

    Raises:
        ValidationError: If any chunk type is invalid.
    """
    if not chunk_types:
        return []

    valid_static = {"abstract", "raptor_overview", "raptor_core"}
    invalid_types = [
        ct
        for ct in chunk_types
        if ct not in valid_static and not ct.startswith("dim_")
    ]
    if invalid_types:
        raise ValidationError(
            f"Invalid chunk types: {invalid_types}. "
            f"Valid types are: {CHUNK_TYPES} plus profile-defined dim_* chunk types"
        )

    return chunk_types


def validate_quality_min(quality_min: int) -> int:
    """Validate quality_min parameter.

    Args:
        quality_min: Minimum quality rating (1-5).

    Returns:
        Validated quality_min value.

    Raises:
        ValidationError: If quality_min is outside valid range.
    """
    if quality_min < 1 or quality_min > 5:
        raise ValidationError(
            f"Invalid quality_min: {quality_min}. Must be between 1 and 5."
        )
    return quality_min


def validate_n_variants(n_variants: int) -> int:
    """Validate n_variants parameter for RRF search.

    Args:
        n_variants: Number of query reformulations (1-10).

    Returns:
        Validated n_variants value (clamped to valid range).
    """
    if n_variants < 1:
        return 1
    if n_variants > 10:
        return 10
    return n_variants


def validate_max_rounds(max_rounds: int) -> int:
    """Validate max_rounds parameter for agentic search.

    Args:
        max_rounds: Maximum gap-analysis rounds (1-5).

    Returns:
        Validated max_rounds value (clamped to valid range).
    """
    if max_rounds < 1:
        return 1
    if max_rounds > 5:
        return 5
    return max_rounds


def validate_recency_boost(recency_boost: float) -> float:
    """Validate recency_boost parameter.

    Args:
        recency_boost: Boost factor for recent papers (0.0-1.0).

    Returns:
        Validated recency_boost value (clamped to valid range).
    """
    if recency_boost < 0.0:
        return 0.0
    if recency_boost > 1.0:
        return 1.0
    return recency_boost
