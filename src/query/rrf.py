"""Reciprocal Rank Fusion (RRF) for multi-query search.

Generates query reformulations via LLM, runs each through vector search,
and fuses results using the RRF formula: score(d) = sum(1 / (k + rank_i(d))).
"""

import json

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Prompt for query reformulation
_REFORMULATION_PROMPT = """\
You are a search query reformulation assistant for an academic literature database.

Given the user's search query, generate {n_variants} alternative phrasings that would \
help find relevant papers. Each variant should approach the topic from a different angle:
- Use synonyms or related terminology
- Rephrase as a research question
- Focus on methodology or application
- Broaden or narrow the scope

Return ONLY a JSON array of strings. Do not include the original query.

User query: {query}"""


def generate_query_variants(
    query: str,
    n_variants: int = 4,
    provider: str = "anthropic",
    model: str | None = None,
) -> list[str]:
    """Generate query reformulations via LLM.

    Args:
        query: Original user search query.
        n_variants: Number of variant queries to generate.
        provider: LLM provider for reformulation.
        model: Model to use. Defaults to provider's default.

    Returns:
        List with original query first, followed by variants.
    """
    from src.analysis.llm_factory import create_llm_client

    client = create_llm_client(
        provider=provider,
        mode="api",
        model=model,
        max_tokens=1024,
        timeout=30,
    )

    prompt = _REFORMULATION_PROMPT.format(query=query, n_variants=n_variants)

    try:
        response_text, _, _ = client._call_api(prompt)

        # Parse JSON array from response
        # Strip markdown code fences if present
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        variants = json.loads(text)
        if not isinstance(variants, list):
            logger.warning("LLM returned non-list for query variants, using original only")
            return [query]

        # Ensure all variants are strings
        variants = [str(v) for v in variants if v][:n_variants]

    except Exception as e:
        logger.warning(f"Query reformulation failed: {e}. Using original query only.")
        return [query]

    # Original query always first
    return [query] + variants


def rrf_score(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Compute Reciprocal Rank Fusion scores from multiple ranked lists.

    For each document d appearing in any ranking:
        score(d) = sum(1 / (k + rank_i(d))) across all rankings containing d

    Args:
        rankings: List of ranked paper_id lists (one per query variant).
        k: RRF constant (default 60, from the original RRF paper).

    Returns:
        List of (paper_id, rrf_score) sorted by score descending.
    """
    scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, paper_id in enumerate(ranking, start=1):
            scores[paper_id] = scores.get(paper_id, 0.0) + 1.0 / (k + rank)

    # Sort by score descending, then by paper_id for stable tie-breaking
    sorted_results = sorted(
        scores.items(),
        key=lambda x: (-x[1], x[0]),
    )

    return sorted_results
