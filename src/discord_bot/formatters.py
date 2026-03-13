"""Discord embed formatters for LITRIS search results."""

from __future__ import annotations

from typing import Any

MAX_EMBED_DESCRIPTION = 4096
MAX_FIELD_VALUE = 1024
RESULTS_PER_PAGE = 5


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def format_paper_embed(paper: dict[str, Any]) -> dict[str, Any]:
    """Format a single paper as a Discord embed dict.

    Args:
        paper: Paper data from LitrisAdapter.get_paper().

    Returns:
        Dict suitable for discord.Embed.from_dict().
    """
    paper_info = paper.get("paper", {})
    extraction = paper.get("extraction", {})

    title = paper_info.get("title", "Unknown Title")
    authors = paper_info.get("author_string", "Unknown")
    year = paper_info.get("publication_year", "n/a")
    doi = paper_info.get("doi")
    journal = paper_info.get("journal", "")
    abstract = paper_info.get("abstract", "")
    collections = paper_info.get("collections", [])

    description_parts = []
    if authors:
        description_parts.append(f"**Authors:** {authors}")
    if journal:
        description_parts.append(f"**Journal:** {journal}")
    if doi:
        description_parts.append(f"**DOI:** {doi}")
    if collections:
        description_parts.append(f"**Collections:** {', '.join(collections)}")

    description = "\n".join(description_parts)

    fields = []

    if abstract:
        fields.append({
            "name": "Abstract",
            "value": _truncate(abstract, MAX_FIELD_VALUE),
            "inline": False,
        })

    thesis = extraction.get("q02_thesis", "") if extraction else ""
    if thesis:
        fields.append({
            "name": "Thesis",
            "value": _truncate(thesis, MAX_FIELD_VALUE),
            "inline": False,
        })

    methods = extraction.get("q07_methods", "") if extraction else ""
    if methods:
        fields.append({
            "name": "Methods",
            "value": _truncate(methods, MAX_FIELD_VALUE),
            "inline": False,
        })

    key_claims = extraction.get("q03_key_claims", "") if extraction else ""
    if key_claims:
        fields.append({
            "name": "Key Claims",
            "value": _truncate(key_claims, MAX_FIELD_VALUE),
            "inline": False,
        })

    quality = extraction.get("quality_rating") if extraction else None
    confidence = extraction.get("extraction_confidence", 0) if extraction else 0
    if quality or confidence:
        meta_parts = []
        if quality:
            meta_parts.append(f"Quality: {quality}/5")
        if confidence:
            meta_parts.append(f"Confidence: {confidence:.0%}")
        fields.append({
            "name": "Extraction Quality",
            "value": " | ".join(meta_parts),
            "inline": True,
        })

    return {
        "title": _truncate(title, 256),
        "description": _truncate(description, MAX_EMBED_DESCRIPTION),
        "color": 0x3498DB,
        "fields": fields,
        "footer": {"text": f"Year: {year} | ID: {paper.get('paper_id', 'unknown')}"},
    }


def format_search_result_embed(
    result: dict[str, Any],
    rank: int | None = None,
) -> dict[str, Any]:
    """Format a single search result as a compact embed dict.

    Args:
        result: Single result from search results list.
        rank: Optional rank number for display.

    Returns:
        Dict suitable for discord.Embed.from_dict().
    """
    title = result.get("title", "Unknown")
    authors = result.get("authors", "Unknown")
    year = result.get("year", "n/a")
    score = result.get("score", 0)
    paper_id = result.get("paper_id", "")
    matched_text = result.get("matched_text", "")

    rank_str = f"#{rank} " if rank else ""
    header = f"{rank_str}{title}"

    description_parts = [
        f"**Authors:** {authors}",
        f"**Year:** {year} | **Score:** {score:.4f}",
    ]

    if matched_text:
        description_parts.append(
            f"**Match:** {_truncate(matched_text, 300)}"
        )

    extraction = result.get("extraction", {})
    if extraction:
        thesis = extraction.get("q02_thesis", "")
        if thesis:
            description_parts.append(
                f"**Thesis:** {_truncate(thesis, 200)}"
            )

    return {
        "title": _truncate(header, 256),
        "description": _truncate(
            "\n".join(description_parts), MAX_EMBED_DESCRIPTION
        ),
        "color": 0x2ECC71,
        "footer": {"text": f"Paper ID: {paper_id}"},
    }


def format_search_page(
    results: list[dict[str, Any]],
    query: str,
    page: int,
    total_results: int,
) -> list[dict[str, Any]]:
    """Format a page of search results as embed dicts.

    Args:
        results: Slice of results for this page.
        query: Original search query.
        page: Current page number (0-indexed).
        total_results: Total number of results.

    Returns:
        List of embed dicts for this page.
    """
    total_pages = (total_results + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE
    start_rank = page * RESULTS_PER_PAGE + 1

    # Header embed
    header = {
        "title": f"Search: {_truncate(query, 200)}",
        "description": (
            f"Found {total_results} results | "
            f"Page {page + 1}/{total_pages}"
        ),
        "color": 0x9B59B6,
    }

    embeds = [header]
    for i, result in enumerate(results):
        embeds.append(format_search_result_embed(result, rank=start_rank + i))

    return embeds


def format_summary_embed(summary: dict[str, Any]) -> dict[str, Any]:
    """Format index summary as a Discord embed dict.

    Args:
        summary: Summary data from LitrisAdapter.get_summary().

    Returns:
        Dict suitable for discord.Embed.from_dict().
    """
    total_papers = summary.get("total_papers", 0)
    total_extractions = summary.get("total_extractions", 0)
    vector_store = summary.get("vector_store", {})
    total_chunks = vector_store.get("total_documents", 0)

    fields = [
        {
            "name": "Papers",
            "value": str(total_papers),
            "inline": True,
        },
        {
            "name": "Extractions",
            "value": str(total_extractions),
            "inline": True,
        },
        {
            "name": "Chunks",
            "value": str(total_chunks),
            "inline": True,
        },
    ]

    # Top collections
    collections = summary.get("papers_by_collection", {})
    if collections:
        top_5 = sorted(collections.items(), key=lambda x: x[1], reverse=True)[:5]
        col_text = "\n".join(f"- {name}: {count}" for name, count in top_5)
        fields.append({
            "name": "Top Collections",
            "value": _truncate(col_text, MAX_FIELD_VALUE),
            "inline": False,
        })

    # Top disciplines
    disciplines = summary.get("top_disciplines", {})
    if disciplines:
        top_5 = sorted(disciplines.items(), key=lambda x: x[1], reverse=True)[:5]
        disc_text = "\n".join(f"- {name}: {count}" for name, count in top_5)
        fields.append({
            "name": "Top Disciplines",
            "value": _truncate(disc_text, MAX_FIELD_VALUE),
            "inline": False,
        })

    return {
        "title": "LITRIS Index Summary",
        "description": f"Literature Review Indexing System -- {total_papers} papers indexed",
        "color": 0xE67E22,
        "fields": fields,
        "footer": {"text": f"Generated: {summary.get('generated_at', 'unknown')}"},
    }
